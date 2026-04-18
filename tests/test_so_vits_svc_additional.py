"""Focused checkpoint and helper coverage for So-VITS-SVC."""
import torch

from auto_voice.models.so_vits_svc import (
    SoVitsSvc,
    _get_state_tensor,
    _infer_config_from_state_dict,
    _ssim_loss,
)


def test_get_state_tensor_supports_module_prefix():
    state_dict = {"module.content_proj.weight": torch.randn(192, 256)}

    tensor = _get_state_tensor(state_dict, "content_proj.weight")

    assert tensor is state_dict["module.content_proj.weight"]


def test_infer_config_from_state_dict_handles_non_mapping():
    inferred = _infer_config_from_state_dict(state_dict=None, base_config={"content_dim": 123})
    assert inferred["content_dim"] == 123


def test_ssim_loss_returns_per_sample_when_not_size_average():
    pred = torch.randn(2, 1, 8, 8)
    target = torch.randn(2, 1, 8, 8)

    loss = _ssim_loss(pred, target, window_size=3, size_average=False)

    assert loss.shape == (2,)


def test_load_pretrained_infers_legacy_dimensions_from_state_dict(tmp_path):
    legacy_model = SoVitsSvc(config={"content_dim": 256, "pitch_dim": 256})
    checkpoint = tmp_path / "legacy-sovits.pt"
    torch.save({"state_dict": legacy_model.state_dict()}, checkpoint)

    loaded = SoVitsSvc.load_pretrained(str(checkpoint), device=torch.device("cpu"))

    assert loaded.content_dim == 256
    assert loaded.pitch_dim == 256
    mel = loaded.infer(
        torch.randn(1, 6, 256),
        torch.randn(1, 6, 256),
        torch.randn(1, 256),
    )
    assert mel.shape == (1, loaded.n_mels, 6)


def test_load_pretrained_falls_back_to_base_config_on_invalid_checkpoint(tmp_path):
    checkpoint = tmp_path / "invalid-sovits.pt"
    checkpoint.write_text("not a checkpoint")

    loaded = SoVitsSvc.load_pretrained(
        str(checkpoint),
        device=torch.device("cpu"),
        config={"content_dim": 320, "pitch_dim": 320},
    )

    assert loaded.content_dim == 320
    assert loaded.pitch_dim == 320
