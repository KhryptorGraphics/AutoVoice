"""Runtime helpers for the opt-in So-VITS-SVC-Fork SPA cycle experiment.

This is a research transfer from DDSP-SVC-Diff, not a production quality fix.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

AUTOVOICE_SPA_CYCLE_RUNTIME = "v1"
CONTENTVEC_SAMPLE_RATE = 16_000


def sample_shifted_f0(
    f0: torch.Tensor,
    uv: torch.Tensor,
    *,
    f0_ceiling_hz: float = 1_100.0,
    min_semitones: int = 6,
    max_semitones: int = 18,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shift the whole F0 contour up, bounded by each sample's headroom.

    The whole INTERPOLATED contour moves, not only its voiced frames. The fork
    feeds the interpolated contour to ``f0_to_coarse`` (the pitch embedding
    into the content encoder) and the masked contour to the decoder; shifting
    only the voiced frames would leave the unvoiced ones at the original
    pitch and put a step of up to 2.8x at every voiced/unvoiced boundary.
    A sawtooth contour is not what a singer sang, and the embedding would
    read it as such. SPA-SVC shifts the whole conditioning sequence too.

    Samples with no positive headroom are marked ineligible and receive an
    unchanged contour. When less than six semitones remain, the available
    positive interval is used rather than exceeding the ceiling.
    """
    if f0.ndim != 2:
        raise ValueError(
            f"f0 must have shape [batch, frames], got {tuple(f0.shape)}"
        )
    if uv.ndim == 3 and uv.shape[1] == 1:
        uv = uv.squeeze(1)
    if uv.shape != f0.shape:
        raise ValueError(
            f"uv shape {tuple(uv.shape)} does not match f0 {tuple(f0.shape)}"
        )
    if f0_ceiling_hz <= 0:
        raise ValueError("f0_ceiling_hz must be positive")
    if min_semitones < 1 or max_semitones < min_semitones:
        raise ValueError("semitone range must satisfy 1 <= min <= max")

    f0_float = f0.float()
    finite = torch.isfinite(f0_float) & (f0_float > 0)
    voiced = (uv > 0) & finite
    # Headroom is set by the HIGHEST frame of the contour, voiced or not:
    # the contour is scaled as a whole, so any frame above the ceiling after
    # scaling would break the uniform scale.
    contour_max = torch.where(finite, f0_float, torch.zeros_like(f0_float))
    sample_max = contour_max.amax(dim=1)
    voiced_max = torch.where(
        voiced, f0_float, torch.zeros_like(f0_float)
    ).amax(dim=1)
    headroom = 12.0 * torch.log2(
        torch.as_tensor(f0_ceiling_hz, device=f0.device, dtype=torch.float32)
        / sample_max.clamp_min(1e-6)
    )
    max_available = torch.floor(headroom).to(torch.long).clamp_(
        0, max_semitones
    )
    eligible = (voiced_max > 0) & (max_available > 0)

    lower = torch.minimum(
        max_available, torch.full_like(max_available, min_semitones)
    )
    choices = (max_available - lower + 1).clamp_min(1)
    draw = torch.floor(
        torch.rand(max_available.shape, device=f0.device, dtype=torch.float32)
        * choices.to(torch.float32)
    ).to(torch.long)
    semitones = torch.where(eligible, lower + draw, torch.zeros_like(draw))
    scale = torch.pow(2.0, semitones.to(torch.float32) / 12.0).unsqueeze(1)
    # One uniform factor per sample: the contour keeps its shape.
    shifted = torch.where(
        finite, f0_float * scale, f0_float
    ).to(f0.dtype)
    return shifted, eligible, semitones


def contentvec_features(
    content_model: nn.Module,
    audio: torch.Tensor,
    *,
    sample_rate: int,
    legacy_final_proj: bool,
) -> torch.Tensor:
    """Extract fork-compatible ContentVec features without detaching audio.

    `utils.get_content` wraps its model call in `torch.no_grad`, blocking the
    SPA cycle gradient. This direct path keeps autograd on generated audio
    while encoder parameters remain frozen.
    """
    if audio.ndim == 3:
        if audio.shape[1] != 1:
            raise ValueError(
                f"audio must be mono, got shape {tuple(audio.shape)}"
            )
        audio = audio.squeeze(1)
    if audio.ndim != 2:
        raise ValueError(
            f"audio must have shape [batch, samples], got {tuple(audio.shape)}"
        )
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    original_device = audio.device
    model_parameter = next(content_model.parameters(), None)
    model_device = (
        model_parameter.device if model_parameter is not None else original_device
    )
    audio = audio.to(device=model_device, dtype=torch.float32)
    if sample_rate != CONTENTVEC_SAMPLE_RATE:
        import torchaudio.functional

        audio = torchaudio.functional.resample(
            audio, orig_freq=sample_rate, new_freq=CONTENTVEC_SAMPLE_RATE
        )

    content_model.eval()
    if legacy_final_proj:
        output = content_model(audio, output_hidden_states=True)
        hidden_states = (
            output.hidden_states if hasattr(output, "hidden_states")
            else output["hidden_states"]
        )
        features = content_model.final_proj(hidden_states[9])
    else:
        output = content_model(audio)
        features = (
            output.last_hidden_state if hasattr(output, "last_hidden_state")
            else output["last_hidden_state"]
        )
    return features.transpose(1, 2).to(original_device)


class SPACycleObjective(nn.Module):
    """Differentiable F0-only shift → ContentVec → original-F0 cycle loss."""

    def __init__(
        self,
        content_model: nn.Module,
        *,
        sample_rate: int,
        legacy_final_proj: bool,
        f0_ceiling_hz: float = 1_100.0,
        max_cycle_samples: int = 4,
        encoder_device: str = "cpu",
    ) -> None:
        super().__init__()
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if f0_ceiling_hz <= 0:
            raise ValueError("f0_ceiling_hz must be positive")
        if max_cycle_samples < 1:
            raise ValueError("max_cycle_samples must be positive")
        self.sample_rate = sample_rate
        self.legacy_final_proj = legacy_final_proj
        self.f0_ceiling_hz = f0_ceiling_hz
        self.max_cycle_samples = max_cycle_samples
        self.encoder_device = encoder_device
        # The encoder is deliberately NOT registered as a child module. A
        # registered submodule would be moved to the accelerator by the
        # trainer and would appear in .parameters(). It stays outside the
        # optimiser, frozen; the audio is moved to it and back, and .to() is
        # differentiable in both directions.
        #
        # Its DEVICE is a choice, not a detail: on CPU the encoder's backward
        # pass dominates the step (a measured canary epoch took 157 s against
        # 29 s without the cycle). On the GPU it is much faster, but this box's
        # GPU memory IS system RAM, so the encoder competes with the two extra
        # generator forwards for it. Measured, not assumed.
        object.__setattr__(self, "_content_model", content_model)
        self._content_model.eval()
        if encoder_device:
            self._content_model.to(encoder_device)
        for parameter in self._content_model.parameters():
            parameter.requires_grad_(False)

    @property
    def content_model(self) -> nn.Module:
        return self._content_model

    def forward(
        self,
        *,
        net_g: nn.Module,
        content: torch.Tensor,
        f0: torch.Tensor,
        uv: torch.Tensor,
        g: torch.Tensor,
        target_mel: torch.Tensor,
        mel_spectrogram_fn,
        hparams,
    ) -> torch.Tensor:
        shifted_f0, eligible, _ = sample_shifted_f0(
            f0,
            uv,
            f0_ceiling_hz=self.f0_ceiling_hz,
        )
        indices = torch.nonzero(eligible, as_tuple=False).flatten()
        if indices.numel() > self.max_cycle_samples:
            order = torch.randperm(indices.numel(), device=indices.device)
            indices = indices.index_select(
                0, order[: self.max_cycle_samples]
            )
        if indices.numel() == 0:
            return target_mel.new_zeros(())

        content = content.index_select(0, indices).float()
        original_f0 = f0.index_select(0, indices).float()
        shifted_f0 = shifted_f0.index_select(0, indices).float()
        uv = uv.index_select(0, indices)
        g = g.index_select(0, indices)
        target_mel = target_mel.index_select(0, indices).float()

        was_training = net_g.training
        net_g.eval()
        self.content_model.eval()
        try:
            shifted_audio = net_g.infer(
                content, shifted_f0, uv, g=g, noice_scale=0.0
            )
            cycle_content = contentvec_features(
                self.content_model,
                shifted_audio,
                sample_rate=self.sample_rate,
                legacy_final_proj=self.legacy_final_proj,
            )
            cycle_content = F.interpolate(
                cycle_content,
                size=original_f0.shape[-1],
                mode="nearest",
            )
            cycle_audio = net_g.infer(
                cycle_content, original_f0, uv, g=g, noice_scale=0.0
            )
        finally:
            net_g.train(was_training)

        cycle_mel = mel_spectrogram_fn(
            cycle_audio.squeeze(1).float(), hparams
        ).float()
        if cycle_mel.ndim != 3 or target_mel.ndim != 3:
            raise ValueError(
                "cycle and target mels must have shape [batch, mel, frames]"
            )
        if cycle_mel.shape[:2] != target_mel.shape[:2]:
            raise ValueError(
                f"cycle mel shape {tuple(cycle_mel.shape)} does not match target "
                f"{tuple(target_mel.shape)}"
            )
        frame_delta = abs(cycle_mel.shape[-1] - target_mel.shape[-1])
        if frame_delta > 1:
            raise ValueError(
                f"cycle mel frame count differs from target by {frame_delta}; "
                "refusing to compare misaligned segments"
            )
        frame_count = min(cycle_mel.shape[-1], target_mel.shape[-1])
        return F.mse_loss(
            cycle_mel[..., :frame_count], target_mel[..., :frame_count]
        )
