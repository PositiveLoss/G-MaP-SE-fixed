use std::{
    collections::HashMap,
    f32::consts::PI,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, ValueEnum};
use kaldi_native_fbank::{
    fbank::{FbankComputer, FbankOptions},
    online::{FeatureComputer, OnlineFeature},
};
use num_complex::Complex32;
#[cfg(feature = "cuda")]
use ort::ep;
use ort::{session::Session, value::Tensor};
use rustfft::{Fft, FftPlanner};
use serde::Deserialize;
use serde_json::Value;

#[derive(Parser, Debug)]
#[command(about = "Run G-MaP-SE inference with ONNX Runtime")]
struct Args {
    #[arg(long)]
    input_noisy_wavs_dir: PathBuf,

    #[arg(long)]
    output_dir: PathBuf,

    #[arg(long, default_value = "../onnx/g_map_se.onnx.json")]
    metadata: PathBuf,

    #[arg(long)]
    onnx_file: Option<PathBuf>,

    #[arg(long)]
    ecapa_onnx_file: Option<PathBuf>,

    #[arg(long)]
    prior_embedding_json: Option<PathBuf>,

    #[arg(long)]
    allow_zero_embedding_fallback: bool,

    #[arg(long, value_enum, default_value_t = Provider::Auto)]
    provider: Provider,

    #[arg(long, default_value = "cuda:0")]
    device: String,

    #[arg(long)]
    profile: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Provider {
    Auto,
    Cpu,
    Cuda,
}

#[derive(Debug, Deserialize)]
struct Metadata {
    onnx_file: PathBuf,
    #[serde(default)]
    slim_onnx_file: Option<PathBuf>,
    #[serde(default)]
    ecapa_model_path: Option<PathBuf>,
    chunk_size: usize,
    overlap_size: usize,
    sampling_rate: u32,
    n_fft: usize,
    hop_size: usize,
    win_size: usize,
    compress_factor: f32,
    embed_dim: usize,
    #[serde(default = "default_ort_batch_size")]
    burn_batch_size: usize,
}

fn default_ort_batch_size() -> usize {
    1
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

    let onnx_file = resolve_path(
        args.onnx_file
            .as_deref()
            .or(metadata.slim_onnx_file.as_deref())
            .unwrap_or(&metadata.onnx_file),
    )?;
    let ecapa_onnx_file = args
        .ecapa_onnx_file
        .as_deref()
        .or(metadata.ecapa_model_path.as_deref())
        .map(resolve_path)
        .transpose()?;
    let cuda_device_id = parse_device_id(&args.device)?;
    let mut model = create_session(&onnx_file, args.provider, cuda_device_id)
        .with_context(|| format!("failed to load enhancement model {}", onnx_file.display()))?;
    let mut ecapa_model = match ecapa_onnx_file.as_deref() {
        Some(path) if path.is_file() => Some(
            create_session(path, Provider::Cpu, cuda_device_id)
                .with_context(|| format!("failed to load ECAPA model {}", path.display()))?,
        ),
        _ => None,
    };
    let embeddings = match &args.prior_embedding_json {
        Some(path) => Some(load_embeddings(path, metadata.embed_dim)?),
        None => None,
    };

    let input_files = wav_files(&args.input_noisy_wavs_dir)?;
    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("failed to create {}", args.output_dir.display()))?;

    println!("ONNX Runtime provider: {}", provider_label(args.provider));
    if matches!(args.provider, Provider::Cuda) {
        println!("CUDA device: {cuda_device_id}");
    }
    println!("Enhancement model: {}", onnx_file.display());
    if let Some(path) = &ecapa_onnx_file {
        println!("ECAPA model: {}", path.display());
    }
    let warmup_start = Instant::now();
    warmup_models(&mut model, ecapa_model.as_mut(), &metadata)?;
    println!("Warmup: {:.3}s", warmup_start.elapsed().as_secs_f64());
    println!("Files: {}", input_files.len());

    let mut dsp = DspContext::new(&metadata);
    for (index, input_file) in input_files.iter().enumerate() {
        println!(
            "[{}/{}] {}",
            index + 1,
            input_files.len(),
            input_file.display()
        );
        let file_start = Instant::now();
        let read_start = Instant::now();
        let wav = read_wav_mono(input_file, metadata.sampling_rate)?;
        let read_elapsed = read_start.elapsed();

        let embedding_start = Instant::now();
        let embedding = embedding_for_file(
            input_file,
            embeddings.as_ref(),
            ecapa_model.as_mut(),
            &mut dsp,
            &wav,
            &metadata,
            metadata.embed_dim,
            args.allow_zero_embedding_fallback,
        )?;
        let embedding_elapsed = embedding_start.elapsed();

        let enhance_start = Instant::now();
        let norm_factor = rms_norm_factor(&wav);
        let normalized: Vec<f32> = wav.iter().map(|sample| sample * norm_factor).collect();
        let mut enhance_stats = EnhanceStats::default();
        let enhanced = enhance_audio(
            &mut model,
            &mut dsp,
            &normalized,
            &embedding,
            &metadata,
            args.profile.then_some(&mut enhance_stats),
        )?;
        let enhance_elapsed = enhance_start.elapsed();

        let write_start = Instant::now();
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
        let write_elapsed = write_start.elapsed();

        if args.profile {
            eprintln!(
                "profile {}: total={:.3}s read={:.3}s embedding={:.3}s enhance={:.3}s write={:.3}s",
                input_file.display(),
                seconds(file_start.elapsed()),
                seconds(read_elapsed),
                seconds(embedding_elapsed),
                seconds(enhance_elapsed),
                seconds(write_elapsed),
            );
            eprintln!(
                "profile enhance: chunks={} stft={:.3}s tensor_upload={:.3}s forward={:.3}s download={:.3}s istft={:.3}s blend={:.3}s",
                enhance_stats.chunks,
                seconds(enhance_stats.stft),
                seconds(enhance_stats.tensor_upload),
                seconds(enhance_stats.forward),
                seconds(enhance_stats.download),
                seconds(enhance_stats.istft),
                seconds(enhance_stats.blend),
            );
        }
    }

    Ok(())
}

fn seconds(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

fn create_session(path: &Path, provider: Provider, cuda_device_id: i32) -> Result<Session> {
    let mut builder = Session::builder()?;
    if matches!(provider, Provider::Cuda) {
        builder = with_cuda_provider(builder, cuda_device_id)?;
    }
    Ok(builder.commit_from_file(path)?)
}

#[cfg(feature = "cuda")]
fn with_cuda_provider(
    builder: ort::session::builder::SessionBuilder,
    cuda_device_id: i32,
) -> Result<ort::session::builder::SessionBuilder> {
    builder
        .with_execution_providers([ep::CUDA::default().with_device_id(cuda_device_id).build()])
        .map_err(|error| anyhow!("failed to register CUDAExecutionProvider: {error}"))
}

#[cfg(not(feature = "cuda"))]
fn with_cuda_provider(
    _builder: ort::session::builder::SessionBuilder,
    _cuda_device_id: i32,
) -> Result<ort::session::builder::SessionBuilder> {
    bail!("--provider cuda requires building rust-ort with: cargo build --release --features cuda")
}

fn provider_label(provider: Provider) -> &'static str {
    match provider {
        Provider::Auto | Provider::Cpu => "CPUExecutionProvider",
        Provider::Cuda => "CUDAExecutionProvider",
    }
}

fn parse_device_id(value: &str) -> Result<i32> {
    if value == "default" {
        return Ok(0);
    }
    if let Some(index) = value.strip_prefix("cuda:") {
        return Ok(index.parse()?);
    }
    if let Ok(index) = value.parse() {
        return Ok(index);
    }
    bail!("unsupported --device '{value}', expected default, cuda:N, or N")
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
    if metadata.burn_batch_size == 0 {
        bail!("metadata burn_batch_size must be positive");
    }
    Ok(())
}

fn resolve_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() || path.exists() {
        return Ok(path.to_path_buf());
    }

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest_relative = manifest_dir.join(path);
    if manifest_relative.exists() {
        return Ok(manifest_relative);
    }

    if let Some(repo_root) = manifest_dir.parent() {
        let repo_relative = repo_root.join(path);
        if repo_relative.exists() {
            return Ok(repo_relative);
        }
    }

    Ok(path.to_path_buf())
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
    ecapa_model: Option<&mut Session>,
    dsp: &mut DspContext,
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
        None => {
            let Some(ecapa_model) = ecapa_model else {
                if allow_zero {
                    eprintln!(
                        "warning: ECAPA model unavailable for {}; using zero fallback",
                        input_file.display()
                    );
                    return Ok(vec![0.0; embed_dim]);
                }
                bail!(
                    "ECAPA ONNX model not found; pass --ecapa_onnx_file or --allow_zero_embedding_fallback"
                );
            };
            extract_ecapa_embedding(ecapa_model, dsp, wav, metadata).or_else(|error| {
                if allow_zero {
                    eprintln!(
                        "warning: ECAPA embedding failed for {}: {error}; using zero fallback",
                        input_file.display()
                    );
                    Ok(vec![0.0; embed_dim])
                } else {
                    Err(error)
                }
            })
        }
    }
}

fn extract_ecapa_embedding(
    model: &mut Session,
    dsp: &mut DspContext,
    wav: &[f32],
    metadata: &Metadata,
) -> Result<Vec<f32>> {
    let features = kaldi_fbank_fixed(wav, dsp)?;
    let feats = Tensor::from_array(([1usize, ECAPA_FRAMES, ECAPA_MEL_BINS], features))?;
    let outputs = model.run(ort::inputs! {
        "feats" => feats
    })?;
    let (_, embedding) = outputs["embs"].try_extract_tensor::<f32>()?;
    if embedding.len() != metadata.embed_dim {
        bail!(
            "ECAPA embedding length mismatch: got {}, expected {}",
            embedding.len(),
            metadata.embed_dim
        );
    }
    Ok(embedding.to_vec())
}

fn warmup_models(
    model: &mut Session,
    ecapa_model: Option<&mut Session>,
    metadata: &Metadata,
) -> Result<()> {
    if let Some(ecapa_model) = ecapa_model {
        let feats = Tensor::from_array((
            [1usize, ECAPA_FRAMES, ECAPA_MEL_BINS],
            vec![0.0_f32; ECAPA_FRAMES * ECAPA_MEL_BINS],
        ))?;
        let outputs = ecapa_model.run(ort::inputs! {
            "feats" => feats
        })?;
        let _ = outputs["embs"].try_extract_tensor::<f32>()?;
    }

    let batch_size = metadata.burn_batch_size;
    let freq_bins = metadata.n_fft / 2 + 1;
    let frames = metadata.chunk_size / metadata.hop_size + 1;
    let values = batch_size * freq_bins * frames;
    let noisy_amp = Tensor::from_array(([batch_size, freq_bins, frames], vec![0.0_f32; values]))?;
    let noisy_pha = Tensor::from_array(([batch_size, freq_bins, frames], vec![0.0_f32; values]))?;
    let prior_embedding = Tensor::from_array((
        [batch_size, metadata.embed_dim],
        vec![0.0_f32; batch_size * metadata.embed_dim],
    ))?;
    let outputs = model.run(ort::inputs! {
        "noisy_amp" => noisy_amp,
        "noisy_pha" => noisy_pha,
        "prior_embedding" => prior_embedding
    })?;
    let _ = outputs["amp_g"].try_extract_tensor::<f32>()?;
    let _ = outputs["pha_g"].try_extract_tensor::<f32>()?;
    Ok(())
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

fn kaldi_fbank_fixed(wav: &[f32], _dsp: &mut DspContext) -> Result<Vec<f32>> {
    if wav.is_empty() {
        bail!("cannot extract ECAPA features from empty audio");
    }

    let mut opts = FbankOptions::default();
    opts.frame_opts.samp_freq = 16_000.0;
    opts.frame_opts.frame_length_ms = ECAPA_FRAME_LENGTH_MS;
    opts.frame_opts.frame_shift_ms = ECAPA_FRAME_SHIFT_MS;
    opts.frame_opts.dither = 0.0;
    opts.frame_opts.preemph_coeff = 0.97;
    opts.frame_opts.remove_dc_offset = true;
    opts.frame_opts.window_type = "hamming".to_string();
    opts.frame_opts.round_to_power_of_two = true;
    opts.frame_opts.snip_edges = true;
    opts.mel_opts.num_bins = ECAPA_MEL_BINS;
    opts.mel_opts.low_freq = ECAPA_LOW_FREQ;
    opts.mel_opts.high_freq = 0.0;
    opts.mel_opts.is_librosa = false;
    opts.mel_opts.use_slaney_mel_scale = false;
    opts.mel_opts.norm.clear();
    opts.use_energy = false;
    opts.raw_energy = false;
    opts.use_log_fbank = true;
    opts.use_power = true;

    let fbank = FbankComputer::new(opts)
        .map_err(|error| anyhow!("failed to create ECAPA fbank extractor: {error}"))?;
    let mut online = OnlineFeature::new(FeatureComputer::Fbank(fbank));
    online.accept_waveform(16_000.0, wav);
    online.input_finished();

    let frames = online.num_frames_ready();
    if frames == 0 {
        bail!("ECAPA fbank produced no frames");
    }

    let mut features = Vec::with_capacity(frames * ECAPA_MEL_BINS);
    for frame in 0..frames {
        let values = online
            .get_frame(frame)
            .ok_or_else(|| anyhow!("missing ECAPA fbank frame {frame}"))?;
        if values.len() != ECAPA_MEL_BINS {
            bail!(
                "ECAPA fbank dimension mismatch: got {}, expected {}",
                values.len(),
                ECAPA_MEL_BINS
            );
        }
        features.extend_from_slice(values);
    }

    mean_normalize_features(&mut features, frames, ECAPA_MEL_BINS);

    let mut fixed = vec![0.0; ECAPA_FRAMES * ECAPA_MEL_BINS];
    let copy_frames = frames.min(ECAPA_FRAMES);
    for frame in 0..copy_frames {
        let source = frame * ECAPA_MEL_BINS;
        let destination = frame * ECAPA_MEL_BINS;
        fixed[destination..destination + ECAPA_MEL_BINS]
            .copy_from_slice(&features[source..source + ECAPA_MEL_BINS]);
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

struct DspContext {
    stft_fft: Arc<dyn Fft<f32>>,
    istft_fft: Arc<dyn Fft<f32>>,
    stft_window: Vec<f32>,
    istft_window: Vec<f32>,
    stft_buffer: Vec<Complex32>,
    istft_buffer: Vec<Complex32>,
    padded: Vec<f32>,
}

impl DspContext {
    fn new(metadata: &Metadata) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let stft_fft = planner.plan_fft_forward(metadata.n_fft);
        let istft_fft = planner.plan_fft_inverse(metadata.n_fft);
        let stft_window = hann_window(metadata.win_size);
        let istft_window = stft_window.clone();
        let stft_buffer = vec![Complex32::new(0.0, 0.0); metadata.n_fft];
        let istft_buffer = vec![Complex32::new(0.0, 0.0); metadata.n_fft];

        Self {
            stft_fft,
            istft_fft,
            stft_window,
            istft_window,
            stft_buffer,
            istft_buffer,
            padded: Vec::new(),
        }
    }
}

#[derive(Default)]
struct EnhanceStats {
    chunks: usize,
    stft: Duration,
    tensor_upload: Duration,
    forward: Duration,
    download: Duration,
    istft: Duration,
    blend: Duration,
}

fn enhance_audio(
    model: &mut Session,
    dsp: &mut DspContext,
    noisy_wav: &[f32],
    embedding: &[f32],
    metadata: &Metadata,
    mut stats: Option<&mut EnhanceStats>,
) -> Result<Vec<f32>> {
    let prior_values = repeat_embedding(embedding, metadata.embed_dim, metadata.burn_batch_size);

    if noisy_wav.len() <= metadata.chunk_size {
        let chunk = ChunkPlan {
            start: 0,
            end: noisy_wav.len(),
            chunk_len: noisy_wav.len(),
        };
        let mut enhanced = enhance_chunk_batch(
            model,
            dsp,
            noisy_wav,
            &[chunk],
            &prior_values,
            metadata,
            stats.as_deref_mut(),
        )?;
        return Ok(enhanced.remove(0));
    }

    let hop_size = metadata.chunk_size - metadata.overlap_size;
    let mut output = vec![0.0; noisy_wav.len()];
    let mut weight = vec![0.0; noisy_wav.len()];

    let mut chunks = Vec::new();
    let mut start = 0;
    while start < noisy_wav.len() {
        let end = (start + metadata.chunk_size).min(noisy_wav.len());
        let chunk_len = end - start;
        chunks.push(ChunkPlan {
            start,
            end,
            chunk_len,
        });

        if end == noisy_wav.len() {
            break;
        }
        start += hop_size;
    }

    for batch in chunks.chunks(metadata.burn_batch_size) {
        let enhanced_chunks = enhance_chunk_batch(
            model,
            dsp,
            noisy_wav,
            batch,
            &prior_values,
            metadata,
            stats.as_deref_mut(),
        )?;
        for (chunk, enhanced) in batch.iter().zip(enhanced_chunks) {
            let blend_start = Instant::now();
            let window = overlap_window(
                chunk.chunk_len,
                metadata.overlap_size,
                chunk.start > 0,
                chunk.end < noisy_wav.len(),
            );

            for index in 0..chunk.chunk_len {
                output[chunk.start + index] += enhanced[index] * window[index];
                weight[chunk.start + index] += window[index];
            }
            if let Some(stats) = stats.as_deref_mut() {
                stats.blend += blend_start.elapsed();
            }
        }
    }

    for (sample, weight) in output.iter_mut().zip(weight) {
        *sample /= weight.max(1e-8);
    }
    Ok(output)
}

#[derive(Clone, Copy)]
struct ChunkPlan {
    start: usize,
    end: usize,
    chunk_len: usize,
}

fn repeat_embedding(embedding: &[f32], embed_dim: usize, batch_size: usize) -> Vec<f32> {
    let mut repeated = Vec::with_capacity(embed_dim * batch_size);
    for _ in 0..batch_size {
        repeated.extend_from_slice(embedding);
    }
    repeated
}

fn enhance_chunk_batch(
    model: &mut Session,
    dsp: &mut DspContext,
    noisy_wav: &[f32],
    chunks: &[ChunkPlan],
    prior_values: &[f32],
    metadata: &Metadata,
    mut stats: Option<&mut EnhanceStats>,
) -> Result<Vec<Vec<f32>>> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }
    if let Some(stats) = stats.as_deref_mut() {
        stats.chunks += chunks.len();
    }

    let stft_start = Instant::now();
    let batch_size = metadata.burn_batch_size;
    let freq_bins = metadata.n_fft / 2 + 1;
    let expected_frames = metadata.chunk_size / metadata.hop_size + 1;
    let mut batch_mag = Vec::with_capacity(batch_size * freq_bins * expected_frames);
    let mut batch_pha = Vec::with_capacity(batch_size * freq_bins * expected_frames);

    for batch_index in 0..batch_size {
        let mut chunk = match chunks.get(batch_index) {
            Some(plan) => noisy_wav[plan.start..plan.end].to_vec(),
            None => Vec::new(),
        };
        chunk.resize(metadata.chunk_size, 0.0);
        let stft = mag_pha_stft(&chunk, metadata, dsp);
        if stft.frames != expected_frames {
            bail!(
                "unexpected STFT frame count: got {}, expected {}",
                stft.frames,
                expected_frames
            );
        }
        batch_mag.extend(stft.mag);
        batch_pha.extend(stft.pha);
    }

    if let Some(stats) = stats.as_deref_mut() {
        stats.stft += stft_start.elapsed();
    }

    let upload_start = Instant::now();
    let noisy_amp = Tensor::from_array(([batch_size, freq_bins, expected_frames], batch_mag))?;
    let noisy_pha = Tensor::from_array(([batch_size, freq_bins, expected_frames], batch_pha))?;
    let prior_embedding =
        Tensor::from_array(([batch_size, metadata.embed_dim], prior_values.to_vec()))?;
    if let Some(stats) = stats.as_deref_mut() {
        stats.tensor_upload += upload_start.elapsed();
    }

    let forward_start = Instant::now();
    let outputs = model.run(ort::inputs! {
        "noisy_amp" => noisy_amp,
        "noisy_pha" => noisy_pha,
        "prior_embedding" => prior_embedding
    })?;
    if let Some(stats) = stats.as_deref_mut() {
        stats.forward += forward_start.elapsed();
    }

    let download_start = Instant::now();
    let (_, amp_data) = outputs["amp_g"].try_extract_tensor::<f32>()?;
    let amp_vec = amp_data.to_vec();
    let (_, pha_data) = outputs["pha_g"].try_extract_tensor::<f32>()?;
    let pha_vec = pha_data.to_vec();
    if let Some(stats) = stats.as_deref_mut() {
        stats.download += download_start.elapsed();
    }

    let istft_start = Instant::now();
    let chunk_values = freq_bins * expected_frames;
    let mut enhanced_chunks = Vec::with_capacity(chunks.len());
    for (batch_index, chunk) in chunks.iter().enumerate() {
        let offset = batch_index * chunk_values;
        let end = offset + chunk_values;
        let mut enhanced = mag_pha_istft(
            &amp_vec[offset..end],
            &pha_vec[offset..end],
            expected_frames,
            metadata.chunk_size,
            metadata,
            dsp,
        );
        enhanced.resize(chunk.chunk_len, 0.0);
        enhanced.truncate(chunk.chunk_len);
        enhanced_chunks.push(enhanced);
    }
    if let Some(stats) = stats.as_deref_mut() {
        stats.istft += istft_start.elapsed();
    }
    Ok(enhanced_chunks)
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

fn mag_pha_stft(samples: &[f32], metadata: &Metadata, dsp: &mut DspContext) -> Stft {
    reflect_pad_into(samples, metadata.n_fft / 2, &mut dsp.padded);
    let frames = (dsp.padded.len() - metadata.n_fft) / metadata.hop_size + 1;
    let freq_bins = metadata.n_fft / 2 + 1;
    let mut mag = vec![0.0; freq_bins * frames];
    let mut pha = vec![0.0; freq_bins * frames];

    for frame in 0..frames {
        dsp.stft_buffer.fill(Complex32::new(0.0, 0.0));
        let frame_start = frame * metadata.hop_size;
        let win_offset = (metadata.n_fft - metadata.win_size) / 2;
        for index in 0..metadata.win_size {
            dsp.stft_buffer[win_offset + index].re =
                dsp.padded[frame_start + win_offset + index] * dsp.stft_window[index];
        }
        dsp.stft_fft.process(&mut dsp.stft_buffer);

        for freq in 0..freq_bins {
            let value = dsp.stft_buffer[freq];
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
    dsp: &mut DspContext,
) -> Vec<f32> {
    let freq_bins = metadata.n_fft / 2 + 1;
    let padded_len = metadata.n_fft + metadata.hop_size * (frames - 1);
    let mut output = vec![0.0; padded_len];
    let mut window_sum = vec![0.0; padded_len];

    for frame in 0..frames {
        dsp.istft_buffer.fill(Complex32::new(0.0, 0.0));
        for freq in 0..freq_bins {
            let index = freq * frames + frame;
            let magnitude = mag[index].max(0.0).powf(1.0 / metadata.compress_factor);
            let phase = pha[index];
            dsp.istft_buffer[freq] =
                Complex32::new(magnitude * phase.cos(), magnitude * phase.sin());
        }
        for freq in 1..(freq_bins - 1) {
            dsp.istft_buffer[metadata.n_fft - freq] = dsp.istft_buffer[freq].conj();
        }

        dsp.istft_fft.process(&mut dsp.istft_buffer);

        let frame_start = frame * metadata.hop_size;
        let win_offset = (metadata.n_fft - metadata.win_size) / 2;
        for index in 0..metadata.win_size {
            let sample = dsp.istft_buffer[win_offset + index].re / metadata.n_fft as f32;
            let position = frame_start + win_offset + index;
            output[position] += sample * dsp.istft_window[index];
            window_sum[position] += dsp.istft_window[index] * dsp.istft_window[index];
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

fn reflect_pad_into(samples: &[f32], pad: usize, padded: &mut Vec<f32>) {
    padded.clear();
    if pad == 0 {
        padded.extend_from_slice(samples);
        return;
    }
    if samples.len() <= 1 {
        padded.resize(
            samples.len() + 2 * pad,
            samples.first().copied().unwrap_or(0.0),
        );
        if !samples.is_empty() {
            padded[pad] = samples[0];
        }
        return;
    }

    padded.reserve(samples.len() + 2 * pad);
    for index in (1..=pad).rev() {
        padded.push(samples[index.min(samples.len() - 1)]);
    }
    padded.extend_from_slice(samples);
    for index in 0..pad {
        let source = samples.len().saturating_sub(2 + index);
        padded.push(samples[source]);
    }
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|index| 0.5 - 0.5 * (2.0 * PI * index as f32 / size as f32).cos())
        .collect()
}
