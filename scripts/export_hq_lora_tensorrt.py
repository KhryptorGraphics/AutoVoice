#!/usr/bin/env python3
"""Export artist-specific HQ LoRA adapters to ONNX and TensorRT.

Builds a TensorRT engine for the trained HQ LoRA adapter used in the
Pillowtalk releases. The engine is artist-specific because the adapter
weights differ per profile.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from auto_voice.export.tensorrt_engine import ShapeProfile, TRTEngineBuilder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class ArtistSpec:
    artist_key: str
    display_name: str
    canonical_profile_id: str
    checkpoint_profile_id: str


ARTISTS = (
    ArtistSpec(
        artist_key="william_singe",
        display_name="William Singe",
        canonical_profile_id="7da05140-1303-40c6-95d9-5b6e2c3624df",
        checkpoint_profile_id="7da05140-1303-40c6-95d9-5b6e2c3624df",
    ),
    ArtistSpec(
        artist_key="conor_maynard",
        display_name="Conor Maynard",
        canonical_profile_id="9679a6ec-e6e2-43c4-b64e-1f004fed34f9",
        checkpoint_profile_id="c572d02c-c687-4bed-8676-6ad253cf1c91",
    ),
)


class LoRALayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        alpha: float = 256.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        lora_out = x @ self.lora_A.T @ self.lora_B.T
        lora_out = self.dropout(lora_out)
        return base_output + self.scaling * lora_out


class HQVoiceLoRAAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 1024,
        output_dim: int = 768,
        lora_rank: int = 128,
        lora_alpha: float = 256.0,
        dropout: float = 0.05,
        num_layers: int = 6,
    ):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.base_layers = nn.ModuleList()
        self.lora_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        for i in range(num_layers):
            self.base_layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.lora_layers.append(
                LoRALayer(dims[i], dims[i + 1], lora_rank, lora_alpha, dropout)
            )
            self.layer_norms.append(nn.LayerNorm(dims[i + 1]))
            if dims[i] != dims[i + 1]:
                self.residual_projs.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            else:
                self.residual_projs.append(nn.Identity())

    def forward(self, content: torch.Tensor) -> torch.Tensor:
        x = content
        for i, (base, lora, ln, res) in enumerate(
            zip(self.base_layers, self.lora_layers, self.layer_norms, self.residual_projs)
        ):
            residual = res(x)
            x = lora(x, base(x))
            x = ln(x)
            if i < len(self.base_layers) - 1:
                x = torch.nn.functional.gelu(x) + 0.1 * residual
        return x

    def load_lora_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for i, lora in enumerate(self.lora_layers):
            key_a = f"lora_{i}_A"
            key_b = f"lora_{i}_B"
            if key_a in state and key_b in state:
                lora.lora_A.data.copy_(state[key_a])
                lora.lora_B.data.copy_(state[key_b])


def _find_artist(artist_key: str) -> ArtistSpec:
    for spec in ARTISTS:
        if spec.artist_key == artist_key:
            return spec
    raise KeyError(f"Unknown artist key: {artist_key}")


def _checkpoint_path(spec: ArtistSpec) -> Path:
    return DATA_DIR / "checkpoints" / "hq" / f"{spec.checkpoint_profile_id}_hq_lora.pt"


def _load_adapter(spec: ArtistSpec) -> tuple[HQVoiceLoRAAdapter, dict]:
    checkpoint_path = _checkpoint_path(spec)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    model = HQVoiceLoRAAdapter(**config)
    model.load_lora_state_dict(payload["lora_state"])
    model.eval()
    return model, payload


def _export_onnx(model: HQVoiceLoRAAdapter, output_path: Path) -> None:
    dummy_content = torch.randn(1, 100, 768)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_content,
        str(output_path),
        opset_version=17,
        dynamo=False,
        input_names=["content"],
        output_names=["adapted_content"],
        dynamic_axes={
            "content": {0: "batch", 1: "frames"},
            "adapted_content": {0: "batch", 1: "frames"},
        },
        do_constant_folding=True,
    )


def export_artist(spec: ArtistSpec, workspace_size_gb: float = 2.0) -> dict:
    release_dir = MODELS_DIR / spec.artist_key
    tensorrt_dir = release_dir / "artifacts" / "tensorrt"
    tensorrt_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = tensorrt_dir / "hq_voice_lora.onnx"
    engine_path = tensorrt_dir / "hq_voice_lora.engine"
    metadata_path = tensorrt_dir / "engine_metadata.json"

    model, payload = _load_adapter(spec)
    _export_onnx(model, onnx_path)

    builder = TRTEngineBuilder(workspace_size_gb=workspace_size_gb)
    shape_profiles = {
        "content": ShapeProfile(
            min=(1, 10, 768),
            opt=(1, 100, 768),
            max=(1, 1000, 768),
        )
    }
    builder.load_cached_engine(str(onnx_path), str(engine_path), fp16=True, shape_profiles=shape_profiles)

    metadata = {
        "artist_key": spec.artist_key,
        "display_name": spec.display_name,
        "canonical_profile_id": spec.canonical_profile_id,
        "checkpoint_profile_id": spec.checkpoint_profile_id,
        "checkpoint_path": str(_checkpoint_path(spec)),
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "precision": "fp16",
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_global_step": payload.get("global_step"),
        "checkpoint_loss": payload.get("loss"),
        "config": payload.get("config", {}),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artist",
        choices=["william_singe", "conor_maynard", "all"],
        default="all",
        help="Which artist package to export.",
    )
    parser.add_argument(
        "--workspace-size-gb",
        type=float,
        default=2.0,
        help="TensorRT workspace size in GB.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    targets = ARTISTS if args.artist == "all" else (_find_artist(args.artist),)
    exported = [export_artist(spec, workspace_size_gb=args.workspace_size_gb) for spec in targets]
    print(json.dumps({"exported": exported}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
