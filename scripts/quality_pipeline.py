#!/usr/bin/env python3
"""Quality Voice Conversion Pipeline for AutoVoice.

High-fidelity pipeline for song conversion using Seed-VC.
Architecture: Whisper -> Seed-VC DiT (CFM) -> BigVGAN (44kHz)

Design choices for quality:
- Whisper encoder for robust semantic extraction
- Seed-VC DiT with Conditional Flow Matching
- BigVGAN vocoder for high-fidelity 44kHz synthesis
- CAMPPlus speaker style encoding
- Optional HQ-SVC enhancement
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Any

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'seed-vc'))

import torch
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
import torchaudio
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class QualityConfig:
    """Configuration for quality pipeline."""
    sample_rate: int = 44100  # High quality 44.1kHz
    hop_length: int = 512
    diffusion_steps: int = 30  # CFM inference steps
    length_adjust: float = 1.0
    inference_cfg_rate: float = 0.7
    f0_condition: bool = True  # Use F0 for singing
    auto_f0_adjust: bool = True  # Auto-adjust pitch to target
    device: str = "cuda"
    fp16: bool = True
    max_context_window: int = 30  # seconds
    enable_hq_super_resolution: bool = False
    hq_require_gpu: bool = False
    enable_nsf_harmonic_enhancement: bool = False
    nsf_harmonic_strength: float = 0.12
    nsf_max_harmonics: int = 6
    nsf_blend: float = 0.2
    nsf_noise_blend: float = 0.05
    nsf_voice_characteristic: str = "neutral"
    enable_smoothsinger_smoothing: bool = False
    smoothness_strength: float = 0.35
    pitch_smoothing_kernel: int = 9
    preserve_vibrato: bool = True
    vibrato_mix: float = 0.25
    transfer_reference_dynamics: bool = False
    dynamics_mix: float = 0.35


class QualityVoiceConverter:
    """High-fidelity voice conversion using Seed-VC.

    Pipeline:
    1. Whisper extracts semantic features (speaker-invariant)
    2. CAMPPlus extracts speaker style from reference
    3. RMVPE extracts F0 pitch contour
    4. Seed-VC DiT transforms with CFM
    5. BigVGAN synthesizes 44kHz waveform

    Optimizations:
    - Chunked processing for long audio
    - FP16 inference
    - Crossfade for seamless chunks
    """

    def __init__(self, config: Optional[QualityConfig] = None):
        self.config = config or QualityConfig()
        self.device = torch.device(self.config.device)

        # Models (lazy loaded)
        self._model = None
        self._semantic_fn = None
        self._f0_fn = None
        self._vocoder = None
        self._campplus = None
        self._mel_fn = None
        self._mel_fn_args = None
        self._nsf_enhancer = None
        self._smoothsinger_post_processor = None
        self._hq_wrapper = None
        self.last_run_metadata: dict[str, Any] = {}

        logger.info(f"QualityVoiceConverter initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Sample rate: {self.config.sample_rate}Hz")
        logger.info(f"  Diffusion steps: {self.config.diffusion_steps}")
        logger.info(f"  F0 conditioned: {self.config.f0_condition}")

    def _load_models(self):
        """Load all Seed-VC models."""
        if self._model is not None:
            return

        logger.info("Loading Seed-VC models...")
        original_dir = os.getcwd()
        os.chdir(Path(__file__).parent.parent / 'models' / 'seed-vc')

        try:
            from modules.commons import recursive_munch, build_model, load_checkpoint
            from hf_utils import load_custom_model_from_hf

            # Load DiT model (F0-conditioned for singing)
            if self.config.f0_condition:
                dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
                    "Plachta/Seed-VC",
                    "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
                    "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
                )
            else:
                dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
                    "Plachta/Seed-VC",
                    "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
                    "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
                )

            config = yaml.safe_load(open(dit_config_path, "r"))
            model_params = recursive_munch(config["model_params"])
            model_params.dit_type = 'DiT'
            self._model = build_model(model_params, stage="DiT")

            self._model, _, _, _ = load_checkpoint(
                self._model,
                None,
                dit_checkpoint_path,
                load_only_params=True,
                ignore_modules=[],
                is_distributed=False,
            )

            for key in self._model:
                self._model[key].to(self.device)
                self._model[key].eval()

            self._model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

            # Load Whisper for semantic extraction
            logger.info("Loading Whisper encoder...")
            from transformers import AutoFeatureExtractor, WhisperModel
            whisper_name = model_params.speech_tokenizer.name
            whisper_model = WhisperModel.from_pretrained(
                whisper_name, torch_dtype=torch.float16
            ).to(self.device)
            del whisper_model.decoder
            whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

            def semantic_fn(waves_16k):
                ori_inputs = whisper_feature_extractor(
                    [waves_16k.squeeze(0).cpu().numpy()],
                    return_tensors="pt",
                    return_attention_mask=True
                )
                ori_input_features = whisper_model._mask_input_features(
                    ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
                ).to(self.device)
                with torch.no_grad():
                    ori_outputs = whisper_model.encoder(
                        ori_input_features.to(whisper_model.encoder.dtype),
                        head_mask=None,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                S_ori = ori_outputs.last_hidden_state.to(torch.float32)
                S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
                return S_ori

            self._semantic_fn = semantic_fn
            self._whisper_model = whisper_model

            # Load F0 extractor
            if self.config.f0_condition:
                logger.info("Loading RMVPE pitch extractor...")
                from modules.rmvpe import RMVPE
                model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
                f0_extractor = RMVPE(model_path, is_half=False, device=self.device)
                self._f0_fn = f0_extractor.infer_from_audio

            # Load CAMPPlus speaker encoder
            logger.info("Loading CAMPPlus speaker encoder...")
            from modules.campplus.DTDNN import CAMPPlus
            campplus_ckpt_path = load_custom_model_from_hf(
                "funasr/campplus", "campplus_cn_common.bin", config_filename=None
            )
            self._campplus = CAMPPlus(feat_dim=80, embedding_size=192)
            self._campplus.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
            self._campplus.to(self.device)
            self._campplus.eval()

            # Load BigVGAN vocoder
            logger.info("Loading BigVGAN vocoder...")
            from modules.bigvgan import bigvgan
            bigvgan_name = model_params.vocoder.name
            self._vocoder = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
            self._vocoder.remove_weight_norm()
            self._vocoder = self._vocoder.to(self.device)
            self._vocoder.eval()

            # Mel spectrogram function
            self._mel_fn_args = {
                "n_fft": config['preprocess_params']['spect_params']['n_fft'],
                "win_size": config['preprocess_params']['spect_params']['win_length'],
                "hop_size": config['preprocess_params']['spect_params']['hop_length'],
                "num_mels": config['preprocess_params']['spect_params']['n_mels'],
                "sampling_rate": config['preprocess_params']['sr'],
                "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
                "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
                "center": False
            }
            from modules.audio import mel_spectrogram
            self._mel_fn = lambda x: mel_spectrogram(x, **self._mel_fn_args)

            # Update sample rate from config
            self.config.sample_rate = config['preprocess_params']['sr']
            self.config.hop_length = config['preprocess_params']['spect_params']['hop_length']

            logger.info(f"Models loaded. SR={self.config.sample_rate}, hop={self.config.hop_length}")
        finally:
            os.chdir(original_dir)

    def extract_speaker_style(self, reference_audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract speaker style embedding from reference audio using CAMPPlus."""
        self._load_models()

        # Resample to 16kHz for CAMPPlus
        if sr != 16000:
            ref_16k = torchaudio.functional.resample(reference_audio, sr, 16000)
        else:
            ref_16k = reference_audio

        if ref_16k.dim() == 1:
            ref_16k = ref_16k.unsqueeze(0)

        # Extract fbank features
        feat = torchaudio.compliance.kaldi.fbank(
            ref_16k,
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000
        )
        feat = feat - feat.mean(dim=0, keepdim=True)

        # Get style embedding
        with torch.no_grad():
            style = self._campplus(feat.unsqueeze(0).to(self.device))

        return style

    def _get_nsf_enhancer(self):
        if self._nsf_enhancer is None:
            from auto_voice.models.nsf import NSFHarmonicEnhancer

            self._nsf_enhancer = NSFHarmonicEnhancer(
                harmonic_strength=self.config.nsf_harmonic_strength,
                max_harmonics=self.config.nsf_max_harmonics,
                blend=self.config.nsf_blend,
                noise_blend=self.config.nsf_noise_blend,
                voice_characteristic=self.config.nsf_voice_characteristic,
            )
        return self._nsf_enhancer

    def _get_smoothsinger_post_processor(self):
        if self._smoothsinger_post_processor is None:
            from auto_voice.models.smoothsinger_decoder import SmoothSingerPostProcessor

            self._smoothsinger_post_processor = SmoothSingerPostProcessor(
                smoothness_strength=self.config.smoothness_strength,
                smoothing_kernel=self.config.pitch_smoothing_kernel,
                preserve_vibrato=self.config.preserve_vibrato,
                vibrato_mix=self.config.vibrato_mix,
                dynamics_mix=self.config.dynamics_mix,
            )
        return self._smoothsinger_post_processor

    def _get_hq_wrapper(self):
        if self._hq_wrapper is None:
            from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

            self._hq_wrapper = HQSVCWrapper(
                device=self.device,
                require_gpu=self.config.hq_require_gpu,
            )
        return self._hq_wrapper

    def _apply_quality_post_processing(
        self,
        converted_audio: np.ndarray,
        output_sample_rate: int,
        f0_contour: Optional[np.ndarray],
        reference_audio: np.ndarray,
        reference_sr: int,
    ) -> Tuple[np.ndarray, int]:
        metadata: dict[str, Any] = {
            "post_processing": list(self.last_run_metadata.get("post_processing", [])),
            "smooth_pitch_applied": self.last_run_metadata.get("smooth_pitch_applied", False),
            "f0_conditioned": self.last_run_metadata.get("f0_conditioned", False),
        }
        processed_audio = np.asarray(converted_audio, dtype=np.float32)
        current_sr = int(output_sample_rate)
        resampled_reference = np.asarray(reference_audio, dtype=np.float32)

        if reference_sr != current_sr:
            resampled_reference = librosa.resample(
                resampled_reference,
                orig_sr=reference_sr,
                target_sr=current_sr,
            ).astype(np.float32)

        if self.config.transfer_reference_dynamics:
            processor = self._get_smoothsinger_post_processor()
            processed_audio = processor.transfer_dynamics(processed_audio, resampled_reference)
            metadata["post_processing"].append("smoothsinger_dynamics_transfer")

        if self.config.enable_nsf_harmonic_enhancement:
            processed_audio = self._get_nsf_enhancer().enhance(
                processed_audio,
                f0_contour,
                current_sr,
            )
            metadata["post_processing"].append("nsf_harmonic_enhancement")

        if self.config.enable_hq_super_resolution:
            try:
                super_resolved = self._get_hq_wrapper().super_resolve(
                    torch.tensor(processed_audio, dtype=torch.float32),
                    sample_rate=current_sr,
                )
                processed_audio = np.asarray(super_resolved["audio"], dtype=np.float32)
                current_sr = int(super_resolved["sample_rate"])
                metadata["post_processing"].append("hq_super_resolution")
            except Exception as exc:
                logger.warning("HQ-SVC enhancement unavailable, continuing without it: %s", exc)
                metadata["hq_super_resolution_skipped"] = str(exc)

        self.last_run_metadata = metadata
        return processed_audio, current_sr

    def convert(
        self,
        source_audio: np.ndarray,
        source_sr: int,
        reference_audio: np.ndarray,
        reference_sr: int,
        pitch_shift: int = 0,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[np.ndarray, int]:
        """Convert source audio to target voice style.

        Args:
            source_audio: Source audio to convert
            source_sr: Source sample rate
            reference_audio: Reference audio with target voice
            reference_sr: Reference sample rate
            pitch_shift: Pitch shift in semitones
            progress_callback: Optional callback(progress, status)

        Returns:
            (converted_audio, output_sample_rate)
        """
        self._load_models()
        self.last_run_metadata = {
            "post_processing": [],
            "smooth_pitch_applied": False,
            "f0_conditioned": bool(self.config.f0_condition),
        }

        if progress_callback:
            progress_callback(0.1, "Loading audio...")

        sr = self.config.sample_rate
        hop_length = self.config.hop_length

        # Convert to tensors and resample
        source = torch.tensor(source_audio).float()
        if source_sr != sr:
            source = torchaudio.functional.resample(source, source_sr, sr)
        source = source.unsqueeze(0).to(self.device)

        reference = torch.tensor(reference_audio).float()
        if reference_sr != sr:
            reference = torchaudio.functional.resample(reference, reference_sr, sr)
        # Limit reference to 25 seconds
        max_ref_samples = sr * 25
        reference = reference[:max_ref_samples].unsqueeze(0).to(self.device)

        if progress_callback:
            progress_callback(0.2, "Extracting semantic features...")

        # Extract semantic features
        source_16k = torchaudio.functional.resample(source, sr, 16000)
        ref_16k = torchaudio.functional.resample(reference, sr, 16000)

        # Handle long audio with chunking
        max_context_window = sr // hop_length * self.config.max_context_window
        overlap_frame_len = 16
        overlap_wave_len = overlap_frame_len * hop_length

        if source_16k.size(-1) <= 16000 * 30:
            S_alt = self._semantic_fn(source_16k)
        else:
            # Chunk processing for long audio
            overlapping_time = 5
            S_alt_list = []
            buffer = None
            traversed_time = 0
            total_time = source_16k.size(-1)

            while traversed_time < total_time:
                if buffer is None:
                    chunk = source_16k[:, traversed_time:traversed_time + 16000 * 30]
                else:
                    chunk = torch.cat([
                        buffer,
                        source_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]
                    ], dim=-1)

                S_alt = self._semantic_fn(chunk)

                if traversed_time == 0:
                    S_alt_list.append(S_alt)
                else:
                    S_alt_list.append(S_alt[:, 50 * overlapping_time:])

                buffer = chunk[:, -16000 * overlapping_time:]
                traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time

                if progress_callback:
                    prog = 0.2 + 0.2 * (traversed_time / total_time)
                    progress_callback(prog, f"Extracting semantics... {traversed_time//16000}s")

            S_alt = torch.cat(S_alt_list, dim=1)

        S_ori = self._semantic_fn(ref_16k)

        if progress_callback:
            progress_callback(0.4, "Extracting mel spectrograms...")

        # Mel spectrograms
        mel = self._mel_fn(source.float())
        mel2 = self._mel_fn(reference.float())

        target_lengths = torch.LongTensor([int(mel.size(2) * self.config.length_adjust)]).to(self.device)
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(self.device)

        if progress_callback:
            progress_callback(0.5, "Extracting speaker style...")

        # Speaker style
        feat2 = torchaudio.compliance.kaldi.fbank(
            ref_16k, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = self._campplus(feat2.unsqueeze(0))

        # F0 extraction
        if self.config.f0_condition:
            if progress_callback:
                progress_callback(0.55, "Extracting pitch...")

            F0_ori = self._f0_fn(ref_16k[0].cpu().numpy(), thred=0.03)
            F0_alt = self._f0_fn(source_16k[0].cpu().numpy(), thred=0.03)

            F0_ori = torch.from_numpy(F0_ori).to(self.device)[None]
            F0_alt = torch.from_numpy(F0_alt).to(self.device)[None]

            # Auto F0 adjustment
            voiced_F0_ori = F0_ori[F0_ori > 1]
            voiced_F0_alt = F0_alt[F0_alt > 1]

            log_f0_alt = torch.log(F0_alt + 1e-5)
            voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
            voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
            median_log_f0_ori = torch.median(voiced_log_f0_ori)
            median_log_f0_alt = torch.median(voiced_log_f0_alt)

            shifted_log_f0_alt = log_f0_alt.clone()
            if self.config.auto_f0_adjust:
                shifted_log_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori

            shifted_f0_alt = torch.exp(shifted_log_f0_alt)

            if pitch_shift != 0:
                factor = 2 ** (pitch_shift / 12)
                shifted_f0_alt[F0_alt > 1] = shifted_f0_alt[F0_alt > 1] * factor

            if self.config.enable_smoothsinger_smoothing:
                processor = self._get_smoothsinger_post_processor()
                smoothed_f0 = processor.smooth_f0_contour(
                    shifted_f0_alt.squeeze(0).detach().cpu().numpy()
                )
                shifted_f0_alt = torch.from_numpy(smoothed_f0).to(self.device)[None]
                self.last_run_metadata["smooth_pitch_applied"] = True
                self.last_run_metadata["post_processing"].append("smoothsinger_pitch_smoothing")
        else:
            F0_ori = None
            shifted_f0_alt = None

        if progress_callback:
            progress_callback(0.6, "Running length regulator...")

        # Length regulation
        cond, _, codes, _, _ = self._model.length_regulator(
            S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
        )
        prompt_condition, _, _, _, _ = self._model.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
        )

        if progress_callback:
            progress_callback(0.65, "Running voice conversion...")

        # Chunked conversion
        max_source_window = max_context_window - mel2.size(2)
        processed_frames = 0
        generated_wave_chunks = []
        total_frames = cond.size(1)

        while processed_frames < total_frames:
            chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
            is_last_chunk = processed_frames + max_source_window >= total_frames
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)

            dtype = torch.float16 if self.config.fp16 else torch.float32
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, dtype=dtype):
                    vc_target = self._model.cfm.inference(
                        cat_condition,
                        torch.LongTensor([cat_condition.size(1)]).to(self.device),
                        mel2, style2, None,
                        self.config.diffusion_steps,
                        inference_cfg_rate=self.config.inference_cfg_rate
                    )
                    vc_target = vc_target[:, :, mel2.size(-1):]

                # Clone tensor to avoid inference mode tracking issues
                vc_target_clone = vc_target.clone().float()
                vc_wave = self._vocoder(vc_target_clone).squeeze()
            vc_wave = vc_wave[None, :]

            if processed_frames == 0:
                if is_last_chunk:
                    output_wave = vc_wave[0].cpu().numpy()
                    generated_wave_chunks.append(output_wave)
                    break
                output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -overlap_wave_len:]
                processed_frames += vc_target.size(2) - overlap_frame_len
            elif is_last_chunk:
                output_wave = self._crossfade(
                    previous_chunk.cpu().numpy(),
                    vc_wave[0].cpu().numpy(),
                    overlap_wave_len
                )
                generated_wave_chunks.append(output_wave)
                break
            else:
                output_wave = self._crossfade(
                    previous_chunk.cpu().numpy(),
                    vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                    overlap_wave_len
                )
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -overlap_wave_len:]
                processed_frames += vc_target.size(2) - overlap_frame_len

            if progress_callback:
                prog = 0.65 + 0.3 * (processed_frames / total_frames)
                progress_callback(prog, f"Converting... {processed_frames}/{total_frames} frames")

        converted = np.concatenate(generated_wave_chunks)
        converted, sr = self._apply_quality_post_processing(
            converted_audio=converted,
            output_sample_rate=sr,
            f0_contour=(
                shifted_f0_alt.squeeze(0).detach().cpu().numpy()
                if shifted_f0_alt is not None
                else None
            ),
            reference_audio=reference_audio,
            reference_sr=reference_sr,
        )

        if progress_callback:
            progress_callback(1.0, "Complete!")

        return converted, sr

    def _crossfade(self, chunk1: np.ndarray, chunk2: np.ndarray, overlap: int) -> np.ndarray:
        """Crossfade between two audio chunks."""
        fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
        fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
        if len(chunk2) < overlap:
            chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
        else:
            chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
        return chunk2

    def unload(self):
        """Unload models to free GPU memory."""
        self._model = None
        self._semantic_fn = None
        self._f0_fn = None
        self._vocoder = None
        self._campplus = None
        self._whisper_model = None
        self._nsf_enhancer = None
        self._smoothsinger_post_processor = None
        self._hq_wrapper = None
        torch.cuda.empty_cache()
        logger.info("Models unloaded")


def load_audio_file(path: Path) -> Tuple[np.ndarray, int]:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    audio, sample_rate = librosa.load(path, sr=None, mono=True)
    return audio, sample_rate


def resolve_reference_audio(
    reference_audio: Optional[str],
    target_profile_id: Optional[str],
) -> Tuple[np.ndarray, int, Path]:
    if reference_audio:
        reference_path = Path(reference_audio).resolve()
        audio, sample_rate = load_audio_file(reference_path)
        return audio, sample_rate, reference_path

    if target_profile_id:
        from auto_voice.storage.voice_profiles import VoiceProfileStore

        store = VoiceProfileStore()
        candidate_paths = [Path(path) for path in store.get_all_vocals_paths(target_profile_id)]
        if not candidate_paths:
            raise FileNotFoundError(
                f"No training vocals found for target profile: {target_profile_id}"
            )
        reference_path = candidate_paths[0].resolve()
        audio, sample_rate = load_audio_file(reference_path)
        return audio, sample_rate, reference_path

    raise ValueError("Either --reference-audio or --target-profile-id must be provided")


def build_progress_callback():
    def progress(progress_value: float, status: str):
        bar_length = 40
        filled = int(bar_length * progress_value)
        bar = "=" * filled + "-" * (bar_length - filled)
        print(f"\r[{bar}] {progress_value * 100:3.0f}% - {status}", end="", flush=True)

    return progress


def write_quality_report(
    report_dir: Path,
    source_path: Path,
    output_path: Path,
    reference_path: Path,
    source_audio: np.ndarray,
    converted_audio: np.ndarray,
    reference_audio: np.ndarray,
    output_sample_rate: int,
    elapsed_seconds: float,
    report_max_seconds: Optional[float] = 30.0,
    run_metadata: Optional[dict[str, Any]] = None,
):
    from auto_voice.evaluation import BenchmarkRunner, QualityMetrics

    if output_sample_rate <= 0:
        raise ValueError("output_sample_rate must be positive")

    if report_max_seconds is not None and report_max_seconds > 0:
        max_frames = int(report_max_seconds * output_sample_rate)
    else:
        max_frames = min(
            len(source_audio),
            len(converted_audio),
            len(reference_audio),
        )

    common_frames = min(
        len(source_audio),
        len(converted_audio),
        len(reference_audio),
        max_frames,
    )
    source_window = source_audio[:common_frames]
    converted_window = converted_audio[:common_frames]
    reference_window = reference_audio[:common_frames]

    metrics = QualityMetrics().compute_all(
        reference_audio=torch.tensor(reference_window),
        converted_audio=torch.tensor(converted_window),
        sample_rate=output_sample_rate,
    )
    runner = BenchmarkRunner(metrics=QualityMetrics())
    runner.write_report_artifacts(
        results=[
            {
                "sample_id": source_path.stem,
                "source_audio": torch.tensor(source_window),
                "reference_audio": torch.tensor(reference_window),
                "converted_audio": torch.tensor(converted_window),
                "metrics": metrics,
                "latency_ms": elapsed_seconds * 1000,
                "sample_rate": output_sample_rate,
                "metadata": {
                    "source_path": str(source_path),
                    "reference_path": str(reference_path),
                    "output_path": str(output_path),
                    "evaluation_window_seconds": common_frames / output_sample_rate,
                    "pipeline": run_metadata or {},
                },
            }
        ],
        output_dir=str(report_dir),
        title=f"Quality Pipeline Report: {source_path.stem}",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audio", required=True, help="Source vocals/audio to convert.")
    parser.add_argument(
        "--reference-audio",
        help="Reference audio carrying the target voice. Optional when --target-profile-id is supplied.",
    )
    parser.add_argument(
        "--target-profile-id",
        help="Voice profile ID used to resolve a reference vocal sample when --reference-audio is omitted.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the converted waveform.",
    )
    parser.add_argument(
        "--report-dir",
        help="Optional benchmark report directory. Emits summary.json, metrics.csv, report.md, figures/, tensorboard/.",
    )
    parser.add_argument(
        "--report-max-seconds",
        type=float,
        default=30.0,
        help="Maximum audio duration to score when generating the benchmark report bundle.",
    )
    parser.add_argument("--pitch-shift", type=int, default=0, help="Pitch shift in semitones.")
    parser.add_argument("--diffusion-steps", type=int, default=30, help="Seed-VC diffusion steps.")
    parser.add_argument("--device", default="cuda", help="Inference device.")
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Disable FP16 inference and run the converter in full precision.",
    )
    parser.add_argument(
        "--disable-f0-condition",
        action="store_true",
        help="Disable F0 conditioning for non-singing use cases.",
    )
    parser.add_argument(
        "--disable-auto-f0-adjust",
        action="store_true",
        help="Disable automatic median F0 matching against the reference.",
    )
    parser.add_argument(
        "--hq-super-resolution",
        action="store_true",
        help="Run HQ-SVC enhancement after Seed-VC conversion.",
    )
    parser.add_argument(
        "--hq-require-gpu",
        action="store_true",
        help="Require CUDA for HQ-SVC enhancement instead of skipping when unavailable.",
    )
    parser.add_argument(
        "--enable-nsf-harmonic-enhancement",
        action="store_true",
        help="Apply lightweight NSF harmonic enhancement after conversion.",
    )
    parser.add_argument("--nsf-harmonic-strength", type=float, default=0.12)
    parser.add_argument("--nsf-max-harmonics", type=int, default=6)
    parser.add_argument("--nsf-blend", type=float, default=0.2)
    parser.add_argument("--nsf-noise-blend", type=float, default=0.05)
    parser.add_argument(
        "--nsf-voice-characteristic",
        default="neutral",
        choices=["neutral", "male", "female", "baritone", "tenor", "alto", "soprano"],
    )
    parser.add_argument(
        "--enable-smoothsinger-smoothing",
        action="store_true",
        help="Apply SmoothSinger-inspired F0 smoothing before conversion.",
    )
    parser.add_argument("--smoothness-strength", type=float, default=0.35)
    parser.add_argument("--pitch-smoothing-kernel", type=int, default=9)
    parser.add_argument(
        "--disable-vibrato-preservation",
        action="store_true",
        help="Disable vibrato-preserving residual blending during pitch smoothing.",
    )
    parser.add_argument("--vibrato-mix", type=float, default=0.25)
    parser.add_argument(
        "--transfer-reference-dynamics",
        action="store_true",
        help="Match the converted vocal envelope to the reference performance.",
    )
    parser.add_argument("--dynamics-mix", type=float, default=0.35)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    os.chdir(Path(__file__).parent.parent)

    source_path = Path(args.source_audio).resolve()
    output_path = Path(args.output).resolve()

    source_audio, source_sr = load_audio_file(source_path)
    reference_audio, reference_sr, reference_path = resolve_reference_audio(
        args.reference_audio,
        args.target_profile_id,
    )

    config = QualityConfig(
        diffusion_steps=args.diffusion_steps,
        f0_condition=not args.disable_f0_condition,
        auto_f0_adjust=not args.disable_auto_f0_adjust,
        fp16=not args.fp32,
        device=args.device,
        enable_hq_super_resolution=args.hq_super_resolution,
        hq_require_gpu=args.hq_require_gpu,
        enable_nsf_harmonic_enhancement=args.enable_nsf_harmonic_enhancement,
        nsf_harmonic_strength=args.nsf_harmonic_strength,
        nsf_max_harmonics=args.nsf_max_harmonics,
        nsf_blend=args.nsf_blend,
        nsf_noise_blend=args.nsf_noise_blend,
        nsf_voice_characteristic=args.nsf_voice_characteristic,
        enable_smoothsinger_smoothing=args.enable_smoothsinger_smoothing,
        smoothness_strength=args.smoothness_strength,
        pitch_smoothing_kernel=args.pitch_smoothing_kernel,
        preserve_vibrato=not args.disable_vibrato_preservation,
        vibrato_mix=args.vibrato_mix,
        transfer_reference_dynamics=args.transfer_reference_dynamics,
        dynamics_mix=args.dynamics_mix,
    )
    converter = QualityVoiceConverter(config)
    try:
        progress_callback = build_progress_callback()
        print("\nConverting with quality pipeline...")
        start_time = time.time()
        converted, output_sample_rate = converter.convert(
            source_audio=source_audio,
            source_sr=source_sr,
            reference_audio=reference_audio,
            reference_sr=reference_sr,
            pitch_shift=args.pitch_shift,
            progress_callback=progress_callback,
        )
        elapsed_seconds = time.time() - start_time
        print()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), converted, output_sample_rate)
        print(f"Saved converted audio: {output_path}")
        print(f"Elapsed: {elapsed_seconds:.2f}s")

        if args.report_dir:
            report_dir = Path(args.report_dir).resolve()
            report_dir.mkdir(parents=True, exist_ok=True)
            reference_report_audio = reference_audio
            if reference_sr != output_sample_rate:
                reference_report_audio = librosa.resample(
                    reference_audio,
                    orig_sr=reference_sr,
                    target_sr=output_sample_rate,
                )
            source_report_audio = source_audio
            if source_sr != output_sample_rate:
                source_report_audio = librosa.resample(
                    source_audio,
                    orig_sr=source_sr,
                    target_sr=output_sample_rate,
                )
            write_quality_report(
                report_dir=report_dir,
                source_path=source_path,
                output_path=output_path,
                reference_path=reference_path,
                source_audio=source_report_audio,
                converted_audio=converted,
                reference_audio=reference_report_audio,
                output_sample_rate=output_sample_rate,
                elapsed_seconds=elapsed_seconds,
                report_max_seconds=args.report_max_seconds,
                run_metadata=converter.last_run_metadata,
            )
            print(f"Saved benchmark report bundle: {report_dir}")

        return 0
    finally:
        converter.unload()


if __name__ == "__main__":
    raise SystemExit(main())
