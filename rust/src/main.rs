use std::{
    collections::HashMap,
    f32::consts::PI,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, bail, Context, Result};
use burn::{
    backend::wgpu::{Wgpu, WgpuDevice},
    prelude::*,
    tensor::TensorData,
};
use clap::Parser;
use num_complex::Complex32;
use rustfft::FftPlanner;
use serde::Deserialize;
use serde_json::Value;

mod g_map_se {
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/g_map_se.rs"));
}

mod voxceleb_ecapa512 {
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/voxceleb_ecapa512.rs"));
}

use g_map_se::Model as EnhancementModel;
use voxceleb_ecapa512::Model as EcapaModel;

type B = Wgpu;

#[derive(Parser, Debug)]
#[command(about = "Run G-MaP-SE inference with a Burn WGPU model")]
struct Args {
    #[arg(long)]
    input_noisy_wavs_dir: PathBuf,

    #[arg(long)]
    output_dir: PathBuf,

    #[arg(long, default_value = "../onnx/g_map_se.onnx.json")]
    metadata: PathBuf,

    #[arg(long, default_value = "../burn_models/g_map_se/g_map_se.bpk")]
    burnpack: PathBuf,

    #[arg(
        long,
        default_value = "../burn_models/voxceleb_ecapa512/voxceleb_ecapa512.bpk"
    )]
    ecapa_burnpack: PathBuf,

    #[arg(long)]
    prior_embedding_json: Option<PathBuf>,

    #[arg(long)]
    allow_zero_embedding_fallback: bool,

    #[arg(long, default_value = "default")]
    device: String,
}

#[derive(Debug, Deserialize)]
struct Metadata {
    chunk_size: usize,
    overlap_size: usize,
    sampling_rate: u32,
    n_fft: usize,
    hop_size: usize,
    win_size: usize,
    compress_factor: f32,
    embed_dim: usize,
}

const ECAPA_FRAMES: usize = 321;
const ECAPA_MEL_BINS: usize = 80;
const ECAPA_FRAME_LENGTH_MS: f32 = 25.0;
const ECAPA_FRAME_SHIFT_MS: f32 = 10.0;
const ECAPA_LOW_FREQ: f32 = 20.0;

enum Embeddings {
    One(Vec<f32>),
    ByFile(HashMap<String, Vec<f32>>),
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}

fn run(args: Args) -> Result<()> {
    let metadata_path = resolve_path(&args.metadata)?;
    let metadata: Metadata = serde_json::from_str(
        &fs::read_to_string(&metadata_path)
            .with_context(|| format!("failed to read {}", metadata_path.display()))?,
    )?;
    validate_metadata(&metadata)?;

    let burnpack = resolve_path(&args.burnpack)?;
    let ecapa_burnpack = resolve_path(&args.ecapa_burnpack)?;
    let device = parse_device(&args.device)?;
    let model = EnhancementModel::<B>::from_file(&burnpack, &device);
    let ecapa_model = EcapaModel::<B>::from_file(&ecapa_burnpack, &device);
    let embeddings = match &args.prior_embedding_json {
        Some(path) => Some(load_embeddings(path, metadata.embed_dim)?),
        None => None,
    };

    let input_files = wav_files(&args.input_noisy_wavs_dir)?;
    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("failed to create {}", args.output_dir.display()))?;

    println!("Burn backend: WGPU");
    println!("Device: {:?}", device);
    println!("Files: {}", input_files.len());

    for (index, input_file) in input_files.iter().enumerate() {
        println!(
            "[{}/{}] {}",
            index + 1,
            input_files.len(),
            input_file.display()
        );
        let wav = read_wav_mono(input_file, metadata.sampling_rate)?;
        let embedding = embedding_for_file(
            input_file,
            embeddings.as_ref(),
            &ecapa_model,
            &device,
            &wav,
            &metadata,
            metadata.embed_dim,
            args.allow_zero_embedding_fallback,
        )?;
        let norm_factor = rms_norm_factor(&wav);
        let normalized: Vec<f32> = wav.iter().map(|sample| sample * norm_factor).collect();
        let enhanced = enhance_audio(&model, &device, &normalized, &embedding, &metadata)?;
        let denormalized: Vec<f32> = enhanced
            .into_iter()
            .map(|sample| sample / norm_factor)
            .collect();

        let output_file = args.output_dir.join(
            input_file
                .file_name()
                .ok_or_else(|| anyhow!("invalid input filename {}", input_file.display()))?,
        );
        write_wav_mono(&output_file, &denormalized, metadata.sampling_rate)?;
    }

    Ok(())
}

fn validate_metadata(metadata: &Metadata) -> Result<()> {
    if metadata.chunk_size == 0 {
        bail!("metadata chunk_size must be positive");
    }
    if metadata.overlap_size >= metadata.chunk_size {
        bail!("metadata overlap_size must be smaller than chunk_size");
    }
    if metadata.n_fft == 0 || metadata.win_size == 0 || metadata.hop_size == 0 {
        bail!("metadata n_fft, win_size and hop_size must be positive");
    }
    if metadata.win_size > metadata.n_fft {
        bail!("metadata win_size must be <= n_fft");
    }
    if metadata.compress_factor <= 0.0 {
        bail!("metadata compress_factor must be positive");
    }
    Ok(())
}

fn resolve_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() || path.exists() {
        return Ok(path.to_path_buf());
    }

    let manifest_relative = Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
    if manifest_relative.exists() {
        return Ok(manifest_relative);
    }

    Ok(path.to_path_buf())
}

fn parse_device(value: &str) -> Result<WgpuDevice> {
    if value == "default" {
        return Ok(WgpuDevice::DefaultDevice);
    }
    if value == "cpu" {
        return Ok(WgpuDevice::Cpu);
    }
    if let Some(index) = value.strip_prefix("discrete:") {
        return Ok(WgpuDevice::DiscreteGpu(index.parse()?));
    }
    if let Some(index) = value.strip_prefix("integrated:") {
        return Ok(WgpuDevice::IntegratedGpu(index.parse()?));
    }
    bail!("unsupported --device '{value}', expected default, cpu, discrete:N or integrated:N")
}

fn wav_files(input_dir: &Path) -> Result<Vec<PathBuf>> {
    let pattern = input_dir.join("*.wav").to_string_lossy().into_owned();
    let mut files = glob::glob(&pattern)
        .with_context(|| format!("invalid glob pattern {pattern}"))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    files.sort();

    if files.is_empty() {
        bail!("no .wav files found in {}", input_dir.display());
    }
    Ok(files)
}

fn load_embeddings(path: &Path, embed_dim: usize) -> Result<Embeddings> {
    let value: Value = serde_json::from_str(
        &fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?,
    )?;

    if value.is_array() {
        return Ok(Embeddings::One(json_vec(&value, embed_dim)?));
    }

    let object = value
        .as_object()
        .ok_or_else(|| anyhow!("embedding JSON must be an array or object"))?;
    let mut by_file = HashMap::new();
    for (key, value) in object {
        by_file.insert(key.clone(), json_vec(value, embed_dim)?);
    }
    Ok(Embeddings::ByFile(by_file))
}

fn json_vec(value: &Value, expected_len: usize) -> Result<Vec<f32>> {
    let array = value
        .as_array()
        .ok_or_else(|| anyhow!("embedding value must be an array"))?;
    if array.len() != expected_len {
        bail!(
            "embedding length mismatch: got {}, expected {}",
            array.len(),
            expected_len
        );
    }
    array
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|value| value as f32)
                .ok_or_else(|| anyhow!("embedding contains a non-number"))
        })
        .collect()
}

fn embedding_for_file(
    input_file: &Path,
    embeddings: Option<&Embeddings>,
    ecapa_model: &EcapaModel<B>,
    device: &WgpuDevice,
    wav: &[f32],
    metadata: &Metadata,
    embed_dim: usize,
    allow_zero: bool,
) -> Result<Vec<f32>> {
    match embeddings {
        Some(Embeddings::One(values)) => Ok(values.clone()),
        Some(Embeddings::ByFile(map)) => {
            let basename = input_file
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| anyhow!("invalid filename {}", input_file.display()))?;
            map.get(basename)
                .cloned()
                .or_else(|| map.get(input_file.to_string_lossy().as_ref()).cloned())
                .ok_or_else(|| anyhow!("no embedding found for {basename}"))
        }
        None => extract_ecapa_embedding(ecapa_model, device, wav, metadata).or_else(|error| {
            if allow_zero {
                eprintln!(
                    "warning: ECAPA embedding failed for {}: {error}; using zero fallback",
                    input_file.display()
                );
                Ok(vec![0.0; embed_dim])
            } else {
                Err(error)
            }
        }),
    }
}

fn extract_ecapa_embedding(
    model: &EcapaModel<B>,
    device: &WgpuDevice,
    wav: &[f32],
    metadata: &Metadata,
) -> Result<Vec<f32>> {
    let features = kaldi_fbank_fixed(wav, metadata.sampling_rate, ECAPA_FRAMES, ECAPA_MEL_BINS)?;
    let feats = Tensor::<B, 3>::from_data(
        TensorData::new(features, [1, ECAPA_FRAMES, ECAPA_MEL_BINS]),
        device,
    );
    let embedding = model.forward(feats).into_data().into_vec::<f32>()?;
    if embedding.len() != metadata.embed_dim {
        bail!(
            "ECAPA embedding length mismatch: got {}, expected {}",
            embedding.len(),
            metadata.embed_dim
        );
    }
    Ok(embedding)
}

fn read_wav_mono(path: &Path, target_rate: u32) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let max_value = (1_i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|sample| sample as f32 / max_value))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };

    let mono = if channels == 1 {
        samples
    } else {
        samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
            .collect()
    };

    if spec.sample_rate == target_rate {
        Ok(mono)
    } else {
        Ok(resample_linear(&mono, spec.sample_rate, target_rate))
    }
}

fn write_wav_mono(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("failed to create {}", path.display()))?;
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        writer.write_sample(value)?;
    }
    writer.finalize()?;
    Ok(())
}

fn resample_linear(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if samples.is_empty() || source_rate == target_rate {
        return samples.to_vec();
    }

    let output_len =
        ((samples.len() as f64) * (target_rate as f64) / (source_rate as f64)).round() as usize;
    let scale = source_rate as f64 / target_rate as f64;
    (0..output_len)
        .map(|index| {
            let position = index as f64 * scale;
            let left = position.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let frac = (position - left as f64) as f32;
            samples[left] * (1.0 - frac) + samples[right] * frac
        })
        .collect()
}

fn rms_norm_factor(samples: &[f32]) -> f32 {
    let energy = samples.iter().map(|sample| sample * sample).sum::<f32>();
    ((samples.len() as f32) / energy.max(1e-12)).sqrt()
}

fn kaldi_fbank_fixed(
    wav: &[f32],
    sample_rate: u32,
    target_frames: usize,
    mel_bins: usize,
) -> Result<Vec<f32>> {
    if wav.is_empty() {
        bail!("cannot extract ECAPA features from empty audio");
    }

    let frame_length = ((sample_rate as f32) * ECAPA_FRAME_LENGTH_MS / 1000.0).round() as usize;
    let frame_shift = ((sample_rate as f32) * ECAPA_FRAME_SHIFT_MS / 1000.0).round() as usize;
    let fft_size = frame_length.next_power_of_two();
    let frames = if wav.len() >= frame_length {
        1 + (wav.len() - frame_length) / frame_shift
    } else {
        1
    };
    let hamming = hamming_window(frame_length);
    let mel_filters = mel_filterbank(mel_bins, fft_size, sample_rate);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let mut buffer = vec![Complex32::new(0.0, 0.0); fft_size];
    let mut features = vec![0.0; frames * mel_bins];

    for frame in 0..frames {
        buffer.fill(Complex32::new(0.0, 0.0));
        let start = frame * frame_shift;
        let mut raw = vec![0.0; frame_length];

        for (index, sample) in raw.iter_mut().enumerate() {
            let source = (start + index).min(wav.len() - 1);
            *sample = wav[source] * 32768.0;
        }

        let mean = raw.iter().sum::<f32>() / raw.len() as f32;
        for sample in &mut raw {
            *sample -= mean;
        }

        let first = raw[0];
        for index in (1..raw.len()).rev() {
            raw[index] -= 0.97 * raw[index - 1];
        }
        raw[0] -= 0.97 * first;

        for index in 0..frame_length {
            buffer[index].re = raw[index] * hamming[index];
        }
        fft.process(&mut buffer);

        let power_bins = fft_size / 2 + 1;
        let mut power = vec![0.0; power_bins];
        for bin in 0..power_bins {
            let value = buffer[bin];
            power[bin] = value.re.mul_add(value.re, value.im * value.im);
        }

        for mel in 0..mel_bins {
            let energy = mel_filters[mel]
                .iter()
                .map(|(bin, weight)| power[*bin] * *weight)
                .sum::<f32>()
                .max(f32::EPSILON);
            features[frame * mel_bins + mel] = energy.ln();
        }
    }

    mean_normalize_features(&mut features, frames, mel_bins);

    let mut fixed = vec![0.0; target_frames * mel_bins];
    let copy_frames = frames.min(target_frames);
    for frame in 0..copy_frames {
        let source = frame * mel_bins;
        let destination = frame * mel_bins;
        fixed[destination..destination + mel_bins]
            .copy_from_slice(&features[source..source + mel_bins]);
    }

    Ok(fixed)
}

fn mean_normalize_features(features: &mut [f32], frames: usize, mel_bins: usize) {
    for mel in 0..mel_bins {
        let mean = (0..frames)
            .map(|frame| features[frame * mel_bins + mel])
            .sum::<f32>()
            / frames as f32;
        for frame in 0..frames {
            features[frame * mel_bins + mel] -= mean;
        }
    }
}

fn mel_filterbank(mel_bins: usize, fft_size: usize, sample_rate: u32) -> Vec<Vec<(usize, f32)>> {
    let num_fft_bins = fft_size / 2 + 1;
    let high_freq = sample_rate as f32 / 2.0;
    let low_mel = hz_to_mel(ECAPA_LOW_FREQ);
    let high_mel = hz_to_mel(high_freq);
    let mel_step = (high_mel - low_mel) / (mel_bins + 1) as f32;
    let mel_points: Vec<f32> = (0..mel_bins + 2)
        .map(|index| mel_to_hz(low_mel + index as f32 * mel_step))
        .collect();

    let mut filters = Vec::with_capacity(mel_bins);
    for mel in 0..mel_bins {
        let left = mel_points[mel];
        let center = mel_points[mel + 1];
        let right = mel_points[mel + 2];
        let mut filter = Vec::new();

        for bin in 0..num_fft_bins {
            let freq = bin as f32 * sample_rate as f32 / fft_size as f32;
            let weight = if freq > left && freq <= center {
                (freq - left) / (center - left)
            } else if freq > center && freq < right {
                (right - freq) / (right - center)
            } else {
                0.0
            };
            if weight > 0.0 {
                filter.push((bin, weight));
            }
        }
        filters.push(filter);
    }

    filters
}

fn hz_to_mel(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}

fn hamming_window(size: usize) -> Vec<f32> {
    if size == 1 {
        return vec![1.0];
    }
    (0..size)
        .map(|index| 0.54 - 0.46 * (2.0 * PI * index as f32 / (size - 1) as f32).cos())
        .collect()
}

fn enhance_audio(
    model: &EnhancementModel<B>,
    device: &WgpuDevice,
    noisy_wav: &[f32],
    embedding: &[f32],
    metadata: &Metadata,
) -> Result<Vec<f32>> {
    if noisy_wav.len() <= metadata.chunk_size {
        return enhance_chunk_to_length(
            model,
            device,
            noisy_wav,
            metadata.chunk_size,
            noisy_wav.len(),
            embedding,
            metadata,
        );
    }

    let hop_size = metadata.chunk_size - metadata.overlap_size;
    let mut output = vec![0.0; noisy_wav.len()];
    let mut weight = vec![0.0; noisy_wav.len()];

    let mut start = 0;
    while start < noisy_wav.len() {
        let end = (start + metadata.chunk_size).min(noisy_wav.len());
        let chunk_len = end - start;
        let enhanced = enhance_chunk_to_length(
            model,
            device,
            &noisy_wav[start..end],
            metadata.chunk_size,
            chunk_len,
            embedding,
            metadata,
        )?;
        let window = overlap_window(
            chunk_len,
            metadata.overlap_size,
            start > 0,
            end < noisy_wav.len(),
        );

        for index in 0..chunk_len {
            output[start + index] += enhanced[index] * window[index];
            weight[start + index] += window[index];
        }

        if end == noisy_wav.len() {
            break;
        }
        start += hop_size;
    }

    for (sample, weight) in output.iter_mut().zip(weight) {
        *sample /= weight.max(1e-8);
    }
    Ok(output)
}

fn enhance_chunk_to_length(
    model: &EnhancementModel<B>,
    device: &WgpuDevice,
    noisy_wav: &[f32],
    model_chunk_size: usize,
    output_length: usize,
    embedding: &[f32],
    metadata: &Metadata,
) -> Result<Vec<f32>> {
    let mut chunk = noisy_wav.to_vec();
    chunk.resize(model_chunk_size, 0.0);
    let enhanced = enhance_chunk(model, device, &chunk, embedding, metadata)?;
    let mut fixed = enhanced;
    fixed.resize(output_length, 0.0);
    fixed.truncate(output_length);
    Ok(fixed)
}

fn enhance_chunk(
    model: &EnhancementModel<B>,
    device: &WgpuDevice,
    noisy_wav: &[f32],
    embedding: &[f32],
    metadata: &Metadata,
) -> Result<Vec<f32>> {
    let stft = mag_pha_stft(noisy_wav, metadata);
    let freq_bins = metadata.n_fft / 2 + 1;
    let frames = stft.frames;

    let amp = Tensor::<B, 3>::from_data(TensorData::new(stft.mag, [1, freq_bins, frames]), device);
    let pha = Tensor::<B, 3>::from_data(TensorData::new(stft.pha, [1, freq_bins, frames]), device);
    let prior = Tensor::<B, 2>::from_data(
        TensorData::new(embedding.to_vec(), [1, metadata.embed_dim]),
        device,
    );

    let (amp_g, pha_g) = model.forward(amp, pha, prior);
    let amp_vec = amp_g.into_data().into_vec::<f32>()?;
    let pha_vec = pha_g.into_data().into_vec::<f32>()?;
    Ok(mag_pha_istft(
        &amp_vec,
        &pha_vec,
        frames,
        noisy_wav.len(),
        metadata,
    ))
}

fn overlap_window(
    chunk_len: usize,
    overlap_size: usize,
    fade_in: bool,
    fade_out: bool,
) -> Vec<f32> {
    let mut window = vec![1.0; chunk_len];
    if overlap_size == 0 {
        return window;
    }
    let fade_len = overlap_size.min(chunk_len);
    if fade_in {
        for index in 0..fade_len {
            window[index] = (index + 1) as f32 / (fade_len + 1) as f32;
        }
    }
    if fade_out {
        for index in 0..fade_len {
            window[chunk_len - fade_len + index] =
                (fade_len - index) as f32 / (fade_len + 1) as f32;
        }
    }
    window
}

struct Stft {
    mag: Vec<f32>,
    pha: Vec<f32>,
    frames: usize,
}

fn mag_pha_stft(samples: &[f32], metadata: &Metadata) -> Stft {
    let padded = reflect_pad(samples, metadata.n_fft / 2);
    let frames = (padded.len() - metadata.n_fft) / metadata.hop_size + 1;
    let freq_bins = metadata.n_fft / 2 + 1;
    let window = hann_window(metadata.win_size);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(metadata.n_fft);
    let mut buffer = vec![Complex32::new(0.0, 0.0); metadata.n_fft];
    let mut mag = vec![0.0; freq_bins * frames];
    let mut pha = vec![0.0; freq_bins * frames];

    for frame in 0..frames {
        buffer.fill(Complex32::new(0.0, 0.0));
        let frame_start = frame * metadata.hop_size;
        let win_offset = (metadata.n_fft - metadata.win_size) / 2;
        for index in 0..metadata.win_size {
            buffer[win_offset + index].re =
                padded[frame_start + win_offset + index] * window[index];
        }
        fft.process(&mut buffer);

        for freq in 0..freq_bins {
            let value = buffer[freq];
            let out_index = freq * frames + frame;
            mag[out_index] = (value.re.mul_add(value.re, value.im * value.im) + 1e-9)
                .sqrt()
                .powf(metadata.compress_factor);
            pha[out_index] = (value.im + 1e-10).atan2(value.re + 1e-5);
        }
    }

    Stft { mag, pha, frames }
}

fn mag_pha_istft(
    mag: &[f32],
    pha: &[f32],
    frames: usize,
    output_len: usize,
    metadata: &Metadata,
) -> Vec<f32> {
    let freq_bins = metadata.n_fft / 2 + 1;
    let padded_len = metadata.n_fft + metadata.hop_size * (frames - 1);
    let mut output = vec![0.0; padded_len];
    let mut window_sum = vec![0.0; padded_len];
    let window = hann_window(metadata.win_size);

    let mut planner = FftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(metadata.n_fft);
    let mut buffer = vec![Complex32::new(0.0, 0.0); metadata.n_fft];

    for frame in 0..frames {
        buffer.fill(Complex32::new(0.0, 0.0));
        for freq in 0..freq_bins {
            let index = freq * frames + frame;
            let magnitude = mag[index].max(0.0).powf(1.0 / metadata.compress_factor);
            let phase = pha[index];
            buffer[freq] = Complex32::new(magnitude * phase.cos(), magnitude * phase.sin());
        }
        for freq in 1..(freq_bins - 1) {
            buffer[metadata.n_fft - freq] = buffer[freq].conj();
        }

        ifft.process(&mut buffer);

        let frame_start = frame * metadata.hop_size;
        let win_offset = (metadata.n_fft - metadata.win_size) / 2;
        for index in 0..metadata.win_size {
            let sample = buffer[win_offset + index].re / metadata.n_fft as f32;
            let position = frame_start + win_offset + index;
            output[position] += sample * window[index];
            window_sum[position] += window[index] * window[index];
        }
    }

    for (sample, weight) in output.iter_mut().zip(window_sum) {
        if weight > 1e-8 {
            *sample /= weight;
        }
    }

    let trim = metadata.n_fft / 2;
    output[trim..(trim + output_len).min(output.len())].to_vec()
}

fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    if pad == 0 {
        return samples.to_vec();
    }
    if samples.len() <= 1 {
        let mut padded = vec![samples.first().copied().unwrap_or(0.0); samples.len() + 2 * pad];
        if !samples.is_empty() {
            padded[pad] = samples[0];
        }
        return padded;
    }

    let mut padded = Vec::with_capacity(samples.len() + 2 * pad);
    for index in (1..=pad).rev() {
        padded.push(samples[index.min(samples.len() - 1)]);
    }
    padded.extend_from_slice(samples);
    for index in 0..pad {
        let source = samples.len().saturating_sub(2 + index);
        padded.push(samples[source]);
    }
    padded
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|index| 0.5 - 0.5 * (2.0 * PI * index as f32 / size as f32).cos())
        .collect()
}
