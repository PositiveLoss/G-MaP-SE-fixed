import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import (
    DenseEncoder,
    MaskDecoder,
    PhaseDecoder,
    TSTransformerBlock,
    pesq_score,
    phase_losses,
)

__all__ = ["MPNet", "pesq_score", "phase_losses"]


class InstantiationModule(nn.Module):
    def __init__(self, h, tau=0.2):
        super().__init__()
        self.tau = tau
        self.register_buffer("mu_clean", self._load_prior(h))

    def _load_prior(self, h):
        embed_dim = getattr(h, "embed_dim", 192)
        num_components = getattr(h, "K_components", 192)
        prior_path = os.path.join(getattr(h, "gmm_embed_dir", "gmms"), "clean_gmm_mu_k.pt")
        fallback_path = os.path.join("gmms", "clean_gmm_mu_k.pt")

        for path in (prior_path, fallback_path):
            if os.path.isfile(path):
                prior = torch.load(path, map_location="cpu").float()
                if prior.shape == (num_components, embed_dim):
                    return prior

        return torch.zeros(num_components, embed_dim)

    def forward(self, embedding):
        embedding = F.normalize(embedding, dim=-1)
        mu_clean = F.normalize(self.mu_clean, dim=-1)
        logits = torch.matmul(embedding, mu_clean.transpose(0, 1)) / self.tau
        weights = torch.softmax(logits, dim=-1)
        return torch.matmul(weights, self.mu_clean)

    def default_prior(self, batch_size, device):
        prior = self.mu_clean.mean(dim=0, keepdim=True)
        return prior.to(device).expand(batch_size, -1)


class GatedFusion(nn.Module):
    def __init__(self, feature_dim=64, condition_dim=192):
        super().__init__()
        self.feature_proj = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU())
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, feature_dim), nn.ReLU()
        )
        self.gate_proj = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim))

    def forward(self, feature, condition):
        feature = feature.permute(0, 2, 3, 1)
        projected_feature = self.feature_proj(feature)
        projected_condition = self.condition_proj(condition).unsqueeze(1).unsqueeze(1)
        projected_condition = projected_condition.expand_as(projected_feature)
        gate = torch.sigmoid(
            self.gate_proj(torch.cat((projected_feature, projected_condition), dim=-1))
        )
        fused = (1 - gate) * projected_feature + gate * projected_condition
        return fused.permute(0, 3, 1, 2)


class MPNet(nn.Module):
    def __init__(self, h, num_tsblocks=4):
        super().__init__()
        self.h = h
        self.num_tscblocks = num_tsblocks
        self.dense_encoder = DenseEncoder(h, in_channel=2)
        self.instantiation_module = InstantiationModule(h)
        self.gated_fusion = GatedFusion(
            feature_dim=h.dense_channel,
            condition_dim=getattr(h, "embed_dim", 192),
        )

        self.TSTransformer = nn.ModuleList([])
        for _ in range(num_tsblocks):
            self.TSTransformer.append(TSTransformerBlock(h))

        self.mask_decoder = MaskDecoder(h, out_channel=1)
        self.phase_decoder = PhaseDecoder(h, out_channel=1)

    def forward(self, noisy_wav, noisy_amp, noisy_pha, prior_embedding=None):
        x = torch.stack((noisy_amp, noisy_pha), dim=-1).permute(0, 3, 2, 1)
        x = self.dense_encoder(x)

        if prior_embedding is None:
            prior_embedding = self.instantiation_module.default_prior(
                noisy_wav.size(0), x.device
            )
        else:
            prior_embedding = prior_embedding.to(x.device)
            prior_embedding = self.instantiation_module(prior_embedding)

        x = self.gated_fusion(x, prior_embedding)

        for i in range(self.num_tscblocks):
            x = self.TSTransformer[i](x)

        denoised_amp = noisy_amp * self.mask_decoder(x)
        denoised_pha = self.phase_decoder(x)
        denoised_com = torch.stack(
            (
                denoised_amp * torch.cos(denoised_pha),
                denoised_amp * torch.sin(denoised_pha),
            ),
            dim=-1,
        )

        return denoised_amp, denoised_pha, denoised_com
