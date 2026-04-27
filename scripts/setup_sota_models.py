#!/usr/bin/env python3
"""Download and setup SOTA voice conversion models.

Downloads:
- BigVGAN v2 24kHz 100band (NVIDIA)
- ContentVec (lengyue233)
- RMVPE pitch extractor
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def resolve_models_dir(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    override = os.environ.get("AUTOVOICE_PRETRAINED_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (PROJECT_ROOT / "models" / "pretrained").resolve()


def print_banner(text: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width + "\n")


def download_bigvgan(models_dir: Path):
    """Download BigVGAN v2 24kHz 100band from HuggingFace."""
    print_banner("Downloading BigVGAN v2")

    try:
        from huggingface_hub import hf_hub_download

        # NVIDIA BigVGAN v2 - best quality vocoder
        model_id = "nvidia/bigvgan_v2_24khz_100band_256x"

        print(f"  Downloading from: {model_id}")

        # Download generator weights
        generator_path = hf_hub_download(
            repo_id=model_id,
            filename="bigvgan_generator.pt",
            cache_dir=str(models_dir / 'cache'),
            local_dir=str(models_dir),
        )

        print(f"  ✅ BigVGAN downloaded: {generator_path}")
        return generator_path

    except Exception as e:
        logger.warning(f"Could not download BigVGAN from HuggingFace: {e}")

        # Try alternative: BigVGAN from GitHub releases
        print("  Trying alternative download...")
        try:
            import urllib.request

            # BigVGAN universal vocoder (fallback)
            url = "https://github.com/NVIDIA/BigVGAN/releases/download/v2.0/bigvgan_v2_24khz_100band_256x.zip"
            dest = models_dir / "bigvgan_v2.zip"

            print(f"  Downloading: {url}")
            urllib.request.urlretrieve(url, dest)

            # Extract
            import zipfile
            with zipfile.ZipFile(dest, 'r') as z:
                z.extractall(models_dir)

            print(f"  ✅ BigVGAN extracted to {models_dir}")
            return str(models_dir / "bigvgan_v2_24khz_100band_256x")

        except Exception as e2:
            logger.error(f"Alternative download also failed: {e2}")
            return None


def download_contentvec(models_dir: Path):
    """Download ContentVec encoder from HuggingFace."""
    print_banner("Downloading ContentVec")

    try:
        from transformers import HubertModel

        # ContentVec - best content encoder for SVC
        model_id = "lengyue233/content-vec-best"

        print(f"  Downloading from: {model_id}")

        # This will cache the model
        model = HubertModel.from_pretrained(model_id)

        # Save locally
        save_path = models_dir / "content-vec-best"
        model.save_pretrained(save_path)

        print(f"  ✅ ContentVec downloaded: {save_path}")
        return str(save_path)

    except Exception as e:
        logger.warning(f"Could not download ContentVec: {e}")

        # Check if we have hubert-soft as fallback
        hubert_path = models_dir / "hubert-soft-35d9f29f.pt"
        if hubert_path.exists():
            print(f"  ⚠️  Using existing HuBERT-Soft as fallback: {hubert_path}")
            return str(hubert_path)

        return None


def download_rmvpe(models_dir: Path):
    """Download RMVPE pitch extractor."""
    print_banner("Downloading RMVPE")

    try:
        from huggingface_hub import hf_hub_download

        # RMVPE from RVC project
        model_id = "lj1995/VoiceConversionWebUI"

        print(f"  Downloading from: {model_id}")

        rmvpe_path = hf_hub_download(
            repo_id=model_id,
            filename="rmvpe.pt",
            cache_dir=str(models_dir / 'cache'),
            local_dir=str(models_dir),
        )

        print(f"  ✅ RMVPE downloaded: {rmvpe_path}")
        return rmvpe_path

    except Exception as e:
        logger.warning(f"Could not download RMVPE: {e}")
        return None


def verify_models(models_dir: Path):
    """Verify all models are loadable."""
    print_banner("Verifying Models")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    results = {}

    # Test ContentVec
    print("\n  Testing ContentVec...")
    try:
        from auto_voice.models.encoder import ContentVecEncoder

        encoder = ContentVecEncoder(
            output_dim=768,
            pretrained="lengyue233/content-vec-best",
            device=device
        )

        # Test inference
        test_audio = torch.randn(1, 16000).to(device)  # 1 second
        with torch.no_grad():
            features = encoder.encode(test_audio)

        print(f"    ✅ ContentVec OK - Output shape: {features.shape}")
        results['contentvec'] = True

    except Exception as e:
        print(f"    ❌ ContentVec failed: {e}")
        results['contentvec'] = False

    # Test RMVPE
    print("\n  Testing RMVPE...")
    try:
        from auto_voice.models.pitch import RMVPEPitchExtractor

        rmvpe_path = models_dir / "rmvpe.pt"
        extractor = RMVPEPitchExtractor(
            pretrained=str(rmvpe_path) if rmvpe_path.exists() else None,
            device=device
        )

        test_audio = torch.randn(1, 16000).to(device)
        with torch.no_grad():
            f0 = extractor.extract(test_audio)

        print(f"    ✅ RMVPE OK - Output shape: {f0.shape}")
        results['rmvpe'] = True

    except Exception as e:
        print(f"    ❌ RMVPE failed: {e}")
        results['rmvpe'] = False

    # Test BigVGAN
    print("\n  Testing BigVGAN...")
    try:
        from auto_voice.models.vocoder import BigVGANVocoder

        vocoder = BigVGANVocoder(device=device)

        # Test with random mel
        test_mel = torch.randn(1, 100, 100).to(device)  # [B, mels, time]
        with torch.no_grad():
            audio = vocoder.synthesize(test_mel)

        print(f"    ✅ BigVGAN OK - Output shape: {audio.shape}")
        results['bigvgan'] = True

    except Exception as e:
        print(f"    ❌ BigVGAN failed: {e}")
        results['bigvgan'] = False

    return results


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Bootstrap destination for pretrained model assets. Defaults to AUTOVOICE_PRETRAINED_DIR or repo-root models/pretrained.",
    )
    args = parser.parse_args(argv)
    models_dir = resolve_models_dir(args.models_dir)

    print_banner("SOTA Model Setup")

    os.makedirs(models_dir, exist_ok=True)

    print(f"  Models directory: {models_dir}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Download models
    contentvec_path = download_contentvec(models_dir)
    rmvpe_path = download_rmvpe(models_dir)
    bigvgan_path = download_bigvgan(models_dir)

    # Verify
    results = verify_models(models_dir)

    # Summary
    print_banner("Setup Summary")

    all_ok = all(results.values())

    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {name}")

    if all_ok:
        print("\n  🎉 All SOTA models ready!")
    else:
        print("\n  ⚠️  Some models need attention")

    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
