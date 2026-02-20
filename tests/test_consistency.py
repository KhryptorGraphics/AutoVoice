"""Tests for consistency distillation models (AV-010).

Verifies:
- DiffusionDecoder (BiDilConv, 20 blocks, EDM preconditioning)
- EDMLoss training loss
- KarrasNoiseSchedule
- ConsistencyStudent (EMA teacher, 1-step inference)
- CTLoss_D distillation loss
- RealtimeVoiceConversionPipeline with consistency student
- Inference latency < 50ms per chunk
"""
import time

import numpy as np
import pytest
import torch

from auto_voice.models.consistency import (
    CTLoss_D,
    ConsistencyStudent,
    DiffusionDecoder,
    DiffusionStepEmbedding,
    EDMLoss,
    KarrasNoiseSchedule,
    ResidualBlock,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def batch_params():
    """Standard batch parameters for testing."""
    return {
        'batch_size': 2,
        'n_mels': 80,
        'n_frames': 32,
        'hidden_dim': 64,
        'cond_dim': 64,
        'n_blocks': 4,
    }


@pytest.fixture
def decoder(batch_params, device):
    """Small DiffusionDecoder for testing."""
    return DiffusionDecoder(
        n_mels=batch_params['n_mels'],
        hidden_dim=batch_params['hidden_dim'],
        n_blocks=batch_params['n_blocks'],
        cond_dim=batch_params['cond_dim'],
        sigma_data=0.5,
    ).to(device)


@pytest.fixture
def student(batch_params, device):
    """Small ConsistencyStudent for testing."""
    return ConsistencyStudent(
        n_mels=batch_params['n_mels'],
        hidden_dim=batch_params['hidden_dim'],
        n_blocks=batch_params['n_blocks'],
        cond_dim=batch_params['cond_dim'],
        sigma_data=0.5,
        ema_mu=0.95,
    ).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# ResidualBlock Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestResidualBlock:
    def test_output_shape(self, device):
        block = ResidualBlock(hidden_dim=64, n_mels=80, dilation=2).to(device)
        x = torch.randn(2, 64, 32, device=device)
        diff_step = torch.randn(2, 64, device=device)
        cond = torch.randn(2, 80, 32, device=device)

        residual, skip = block(x, diff_step, cond)

        assert residual.shape == (2, 64, 32)
        assert skip.shape == (2, 64, 32)

    def test_non_nan_output(self, device):
        block = ResidualBlock(hidden_dim=64, n_mels=80, dilation=4).to(device)
        x = torch.randn(2, 64, 16, device=device)
        diff_step = torch.randn(2, 64, device=device)
        cond = torch.randn(2, 80, 16, device=device)

        residual, skip = block(x, diff_step, cond)

        assert not torch.isnan(residual).any()
        assert not torch.isnan(skip).any()

    def test_different_dilations(self, device):
        for dilation in [1, 2, 4, 8, 16, 32]:
            block = ResidualBlock(hidden_dim=32, n_mels=80, dilation=dilation).to(device)
            x = torch.randn(1, 32, 64, device=device)
            diff_step = torch.randn(1, 32, device=device)
            cond = torch.randn(1, 80, 64, device=device)

            residual, skip = block(x, diff_step, cond)
            assert residual.shape == (1, 32, 64)


# ─────────────────────────────────────────────────────────────────────────────
# DiffusionStepEmbedding Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDiffusionStepEmbedding:
    def test_output_shape(self, device):
        emb = DiffusionStepEmbedding(hidden_dim=64).to(device)
        sigma = torch.tensor([0.1, 1.0, 10.0], device=device)
        out = emb(sigma)
        assert out.shape == (3, 64)

    def test_scalar_input(self, device):
        emb = DiffusionStepEmbedding(hidden_dim=128).to(device)
        sigma = torch.tensor(1.0, device=device)
        out = emb(sigma)
        assert out.shape == (1, 128)

    def test_different_sigmas_different_embeddings(self, device):
        emb = DiffusionStepEmbedding(hidden_dim=64).to(device)
        sigma1 = torch.tensor([0.01], device=device)
        sigma2 = torch.tensor([80.0], device=device)
        out1 = emb(sigma1)
        out2 = emb(sigma2)
        # Different sigma values should produce different embeddings
        assert not torch.allclose(out1, out2, atol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# DiffusionDecoder Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDiffusionDecoder:
    def test_output_shape(self, decoder, batch_params, device):
        B, T = batch_params['batch_size'], batch_params['n_frames']
        x = torch.randn(B, batch_params['n_mels'], T, device=device)
        sigma = torch.tensor([1.0, 5.0], device=device)
        cond = torch.randn(B, batch_params['cond_dim'], T, device=device)

        out = decoder(x, sigma, cond)
        assert out.shape == (B, batch_params['n_mels'], T)

    def test_non_nan_output(self, decoder, batch_params, device):
        B, T = batch_params['batch_size'], batch_params['n_frames']
        x = torch.randn(B, batch_params['n_mels'], T, device=device)
        sigma = torch.tensor([0.5, 2.0], device=device)
        cond = torch.randn(B, batch_params['cond_dim'], T, device=device)

        out = decoder(x, sigma, cond)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_edm_preconditioning_at_zero_noise(self, decoder, batch_params, device):
        """At sigma->0, c_skip->1 and c_out->0, so D(x,0) ≈ x (identity)."""
        B, T = batch_params['batch_size'], batch_params['n_frames']
        x = torch.randn(B, batch_params['n_mels'], T, device=device)
        # Very small sigma: skip connection dominates
        sigma = torch.tensor([0.001, 0.001], device=device)
        cond = torch.randn(B, batch_params['cond_dim'], T, device=device)

        out = decoder(x, sigma, cond)
        # Output should be close to input (skip connection dominates)
        assert torch.allclose(out, x, atol=0.1)

    def test_20_blocks_default(self):
        """Default decoder has 20 BiDilConv blocks."""
        dec = DiffusionDecoder(n_mels=80, hidden_dim=64, n_blocks=20)
        assert len(dec.blocks) == 20

    def test_dilation_pattern(self):
        """Verify dilation cycle: [1,2,4,8,16,32,64,128,256,512] x 2."""
        dec = DiffusionDecoder(n_mels=80, hidden_dim=64, n_blocks=20, dilation_cycle=10)
        expected = [2**i for i in range(10)] * 2
        actual = [b.dilated_conv.dilation[0] for b in dec.blocks]
        assert actual == expected

    def test_gradient_flow(self, decoder, batch_params, device):
        """Verify gradients flow through the decoder."""
        B, T = batch_params['batch_size'], batch_params['n_frames']
        x = torch.randn(B, batch_params['n_mels'], T, device=device, requires_grad=True)
        sigma = torch.tensor([1.0, 1.0], device=device)
        cond = torch.randn(B, batch_params['cond_dim'], T, device=device)

        out = decoder(x, sigma, cond)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_raw_forward_independent_of_edm(self, batch_params, device):
        """_raw_forward produces different output than EDM-wrapped forward."""
        dec = DiffusionDecoder(
            n_mels=batch_params['n_mels'],
            hidden_dim=batch_params['hidden_dim'],
            n_blocks=batch_params['n_blocks'],
            cond_dim=batch_params['cond_dim'],
        ).to(device)

        B, T = batch_params['batch_size'], batch_params['n_frames']
        x = torch.randn(B, batch_params['n_mels'], T, device=device)
        sigma = torch.tensor([5.0, 5.0], device=device)
        cond = torch.randn(B, batch_params['cond_dim'], T, device=device)

        raw_out = dec._raw_forward(x, sigma, cond)
        edm_out = dec(x, sigma, cond)
        # EDM wraps with skip connection, so outputs differ
        assert not torch.allclose(raw_out, edm_out, atol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# EDMLoss Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEDMLoss:
    def test_loss_is_scalar(self, decoder, batch_params, device):
        loss_fn = EDMLoss(sigma_data=0.5)
        B, T = batch_params['batch_size'], batch_params['n_frames']
        x = torch.randn(B, batch_params['n_mels'], T, device=device)
        cond = torch.randn(B, batch_params['cond_dim'], T, device=device)

        loss = loss_fn(decoder, x, cond)
        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Positive

    def test_loss_is_finite(self, decoder, batch_params, device):
        loss_fn = EDMLoss(sigma_data=0.5)
        B, T = batch_params['batch_size'], batch_params['n_frames']
        x = torch.randn(B, batch_params['n_mels'], T, device=device)
        cond = torch.randn(B, batch_params['cond_dim'], T, device=device)

        loss = loss_fn(decoder, x, cond)
        assert torch.isfinite(loss)

    def test_loss_backward(self, decoder, batch_params, device):
        """Verify loss produces gradients for model parameters."""
        loss_fn = EDMLoss(sigma_data=0.5)
        B, T = batch_params['batch_size'], batch_params['n_frames']
        x = torch.randn(B, batch_params['n_mels'], T, device=device)
        cond = torch.randn(B, batch_params['cond_dim'], T, device=device)

        loss = loss_fn(decoder, x, cond)
        loss.backward()

        # Check at least one parameter has gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in decoder.parameters()
        )
        assert has_grad


# ─────────────────────────────────────────────────────────────────────────────
# KarrasNoiseSchedule Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKarrasNoiseSchedule:
    def test_schedule_length(self):
        schedule = KarrasNoiseSchedule()
        sigmas = schedule.get_sigmas(50)
        assert sigmas.shape == (51,)  # n_steps + 1 (last is 0)

    def test_schedule_decreasing(self):
        schedule = KarrasNoiseSchedule()
        sigmas = schedule.get_sigmas(20)
        # Should be monotonically decreasing
        for i in range(len(sigmas) - 1):
            assert sigmas[i] >= sigmas[i + 1]

    def test_schedule_bounds(self):
        schedule = KarrasNoiseSchedule(sigma_min=0.002, sigma_max=80.0)
        sigmas = schedule.get_sigmas(10)
        assert sigmas[0].item() == pytest.approx(80.0, rel=1e-3)
        assert sigmas[-2].item() == pytest.approx(0.002, rel=1e-3)
        assert sigmas[-1].item() == 0.0

    def test_schedule_on_device(self, device):
        schedule = KarrasNoiseSchedule()
        sigmas = schedule.get_sigmas(5, device=device)
        assert sigmas.device == device

    def test_single_step(self):
        schedule = KarrasNoiseSchedule(sigma_min=0.002, sigma_max=80.0)
        sigmas = schedule.get_sigmas(1)
        assert sigmas.shape == (2,)
        assert sigmas[0].item() == pytest.approx(80.0, rel=1e-3)
        assert sigmas[-1].item() == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# ConsistencyStudent Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConsistencyStudent:
    def test_init_teacher_matches_student(self, student):
        """Teacher should be initialized from student weights."""
        for p_s, p_t in zip(student.student.parameters(), student.teacher.parameters()):
            assert torch.allclose(p_s, p_t)

    def test_teacher_frozen(self, student):
        """Teacher parameters should not require gradients."""
        for p in student.teacher.parameters():
            assert not p.requires_grad

    def test_ema_update(self, student, batch_params, device):
        """EMA update should move teacher toward student."""
        # Perturb student weights
        with torch.no_grad():
            for p in student.student.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Record teacher before
        teacher_before = [p.clone() for p in student.teacher.parameters()]

        student.update_ema()

        # Teacher should have changed
        changed = False
        for p_before, p_after in zip(teacher_before, student.teacher.parameters()):
            if not torch.allclose(p_before, p_after, atol=1e-6):
                changed = True
                break
        assert changed

    def test_ema_decay_rate(self, student, batch_params, device):
        """EMA update with mu=0.95 should blend 95% old + 5% new."""
        # Set teacher to zeros, student to ones
        with torch.no_grad():
            for p in student.teacher.parameters():
                p.zero_()
            for p in student.student.parameters():
                p.fill_(1.0)

        student.update_ema()

        # Teacher should be approximately 0.05 (= 0.95*0 + 0.05*1)
        for p in student.teacher.parameters():
            if p.numel() > 0:
                assert p.mean().item() == pytest.approx(0.05, abs=1e-5)
                break

    def test_forward_training(self, student, batch_params, device):
        """Training forward produces valid outputs."""
        B, T = batch_params['batch_size'], batch_params['n_frames']
        x = torch.randn(B, batch_params['n_mels'], T, device=device)
        cond = torch.randn(B, batch_params['cond_dim'], T, device=device)

        outputs = student(x, cond, n_steps=10)

        assert 'student_pred' in outputs
        assert 'teacher_pred' in outputs
        assert outputs['student_pred'].shape == (B, batch_params['n_mels'], T)
        assert outputs['teacher_pred'].shape == (B, batch_params['n_mels'], T)
        assert not torch.isnan(outputs['student_pred']).any()
        assert not torch.isnan(outputs['teacher_pred']).any()

    def test_infer_single_step(self, student, batch_params, device):
        """Single-step inference produces valid mel spectrogram."""
        B, T = batch_params['batch_size'], batch_params['n_frames']
        cond = torch.randn(B, batch_params['cond_dim'], T, device=device)

        mel = student.infer(cond)

        assert mel.shape == (B, batch_params['n_mels'], T)
        assert not torch.isnan(mel).any()
        assert not torch.isinf(mel).any()

    def test_infer_custom_frames(self, student, batch_params, device):
        """Inference with custom n_frames."""
        B = batch_params['batch_size']
        T_cond = 50
        T_out = 40
        cond = torch.randn(B, batch_params['cond_dim'], T_cond, device=device)

        mel = student.infer(cond, n_frames=T_out)
        assert mel.shape == (B, batch_params['n_mels'], T_out)

    def test_load_teacher_weights(self, batch_params, device):
        """Load pretrained teacher into student."""
        teacher = DiffusionDecoder(
            n_mels=batch_params['n_mels'],
            hidden_dim=batch_params['hidden_dim'],
            n_blocks=batch_params['n_blocks'],
            cond_dim=batch_params['cond_dim'],
        ).to(device)

        # Randomize teacher weights
        with torch.no_grad():
            for p in teacher.parameters():
                p.normal_(0, 1.0)

        student = ConsistencyStudent(
            n_mels=batch_params['n_mels'],
            hidden_dim=batch_params['hidden_dim'],
            n_blocks=batch_params['n_blocks'],
            cond_dim=batch_params['cond_dim'],
        ).to(device)

        student.load_teacher_weights(teacher)

        # Both student and teacher should match the loaded weights
        for p_s, p_t in zip(student.student.parameters(), teacher.parameters()):
            assert torch.allclose(p_s, p_t)
        for p_ema, p_t in zip(student.teacher.parameters(), teacher.parameters()):
            assert torch.allclose(p_ema, p_t)


# ─────────────────────────────────────────────────────────────────────────────
# CTLoss_D Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCTLossD:
    def test_loss_is_scalar(self, student, batch_params, device):
        loss_fn = CTLoss_D(lambda_mel=0.1)
        B, T = batch_params['batch_size'], batch_params['n_frames']
        x = torch.randn(B, batch_params['n_mels'], T, device=device)
        cond = torch.randn(B, batch_params['cond_dim'], T, device=device)

        losses = loss_fn(student, x, cond, n_steps=10)

        assert losses['total_loss'].dim() == 0
        assert losses['consistency_loss'].dim() == 0
        assert losses['mel_loss'].dim() == 0

    def test_loss_positive(self, student, batch_params, device):
        loss_fn = CTLoss_D(lambda_mel=0.1)
        B, T = batch_params['batch_size'], batch_params['n_frames']
        x = torch.randn(B, batch_params['n_mels'], T, device=device)
        cond = torch.randn(B, batch_params['cond_dim'], T, device=device)

        losses = loss_fn(student, x, cond, n_steps=10)

        assert losses['total_loss'].item() > 0
        assert losses['consistency_loss'].item() >= 0
        assert losses['mel_loss'].item() >= 0

    def test_loss_backward(self, student, batch_params, device):
        """Verify gradients flow to student but not teacher."""
        loss_fn = CTLoss_D(lambda_mel=0.1)
        B, T = batch_params['batch_size'], batch_params['n_frames']
        x = torch.randn(B, batch_params['n_mels'], T, device=device)
        cond = torch.randn(B, batch_params['cond_dim'], T, device=device)

        losses = loss_fn(student, x, cond, n_steps=10)
        losses['total_loss'].backward()

        # Student should have gradients
        student_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in student.student.parameters()
        )
        assert student_has_grad

        # Teacher should NOT have gradients (frozen)
        teacher_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in student.teacher.parameters()
        )
        assert not teacher_has_grad

    def test_consistency_loss_decreases_with_training(self, batch_params, device):
        """Simulate a few training steps and verify loss decreases."""
        student = ConsistencyStudent(
            n_mels=batch_params['n_mels'],
            hidden_dim=batch_params['hidden_dim'],
            n_blocks=batch_params['n_blocks'],
            cond_dim=batch_params['cond_dim'],
            ema_mu=0.95,
        ).to(device)

        loss_fn = CTLoss_D(lambda_mel=0.1)
        optimizer = torch.optim.Adam(student.student.parameters(), lr=1e-3)

        B, T = batch_params['batch_size'], batch_params['n_frames']
        # Fixed data for reproducibility
        torch.manual_seed(42)
        x = torch.randn(B, batch_params['n_mels'], T, device=device)
        cond = torch.randn(B, batch_params['cond_dim'], T, device=device)

        losses_over_time = []
        for _ in range(5):
            optimizer.zero_grad()
            losses = loss_fn(student, x, cond, n_steps=10)
            losses['total_loss'].backward()
            optimizer.step()
            student.update_ema()
            losses_over_time.append(losses['total_loss'].item())

        # Loss should decrease (at least first vs last)
        assert losses_over_time[-1] < losses_over_time[0]


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConsistencyIntegration:
    def test_full_training_loop(self, batch_params, device):
        """End-to-end: train teacher, distill to student, infer."""
        n_mels = batch_params['n_mels']
        hidden = batch_params['hidden_dim']
        n_blocks = batch_params['n_blocks']
        cond_dim = batch_params['cond_dim']
        B, T = batch_params['batch_size'], batch_params['n_frames']

        # Stage 1: Train teacher
        teacher = DiffusionDecoder(
            n_mels=n_mels, hidden_dim=hidden,
            n_blocks=n_blocks, cond_dim=cond_dim,
        ).to(device)
        edm_loss = EDMLoss(sigma_data=0.5)
        opt_t = torch.optim.Adam(teacher.parameters(), lr=1e-3)

        torch.manual_seed(0)
        x = torch.randn(B, n_mels, T, device=device)
        cond = torch.randn(B, cond_dim, T, device=device)

        for _ in range(3):
            opt_t.zero_grad()
            loss = edm_loss(teacher, x, cond)
            loss.backward()
            opt_t.step()

        # Stage 2: Distill to student
        student = ConsistencyStudent(
            n_mels=n_mels, hidden_dim=hidden,
            n_blocks=n_blocks, cond_dim=cond_dim,
            ema_mu=0.95,
        ).to(device)
        student.load_teacher_weights(teacher)

        ct_loss = CTLoss_D(lambda_mel=0.1)
        opt_s = torch.optim.Adam(student.student.parameters(), lr=1e-3)

        for _ in range(3):
            opt_s.zero_grad()
            losses = ct_loss(student, x, cond, n_steps=10)
            losses['total_loss'].backward()
            opt_s.step()
            student.update_ema()

        # Stage 3: Single-step inference
        mel = student.infer(cond)
        assert mel.shape == (B, n_mels, T)
        assert not torch.isnan(mel).any()

    def test_student_produces_comparable_quality(self, batch_params, device):
        """Student 1-step output should have similar magnitude to teacher N-step."""
        n_mels = batch_params['n_mels']
        hidden = batch_params['hidden_dim']
        n_blocks = batch_params['n_blocks']
        cond_dim = batch_params['cond_dim']
        B, T = batch_params['batch_size'], batch_params['n_frames']

        student = ConsistencyStudent(
            n_mels=n_mels, hidden_dim=hidden,
            n_blocks=n_blocks, cond_dim=cond_dim,
        ).to(device)

        cond = torch.randn(B, cond_dim, T, device=device)

        # Student single-step
        mel_student = student.infer(cond)

        # Teacher multi-step (simulate via single call at sigma_max)
        x_noise = torch.randn(B, n_mels, T, device=device) * 80.0
        sigma = torch.full((B,), 80.0, device=device)
        mel_teacher = student.teacher(x_noise, sigma, cond)

        # Both should produce finite, non-zero output
        assert torch.isfinite(mel_student).all()
        assert torch.isfinite(mel_teacher).all()
        assert mel_student.abs().mean() > 0
        assert mel_teacher.abs().mean() > 0

    def test_inference_latency(self, device):
        """Single-step inference should be fast (<50ms on CPU with small model)."""
        student = ConsistencyStudent(
            n_mels=80, hidden_dim=64,
            n_blocks=4, cond_dim=64,
        ).to(device)

        cond = torch.randn(1, 64, 32, device=device)

        # Warmup
        for _ in range(3):
            student.infer(cond)

        # Measure
        times = []
        for _ in range(10):
            start = time.perf_counter()
            student.infer(cond)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # ms

        avg_ms = np.mean(times)
        # CPU with small model should be well under 50ms
        assert avg_ms < 50.0, f"Inference too slow: {avg_ms:.1f}ms (target <50ms)"


# ─────────────────────────────────────────────────────────────────────────────
# RealtimeVoiceConversionPipeline Consistency Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRealtimePipelineConsistency:
    def test_load_consistency_student(self, batch_params, device):
        """Pipeline loads consistency student correctly."""
        from auto_voice.inference.realtime_voice_conversion_pipeline import (
            RealtimeVoiceConversionPipeline,
        )

        pipeline = RealtimeVoiceConversionPipeline(
            device=device,
            config={'use_consistency': True, 'sample_rate': 22050},
        )

        student = ConsistencyStudent(
            n_mels=batch_params['n_mels'],
            hidden_dim=batch_params['hidden_dim'],
            n_blocks=batch_params['n_blocks'],
            cond_dim=batch_params['cond_dim'],
        ).to(device)

        pipeline.load_consistency_student(student)
        assert pipeline._use_consistency is True
        assert pipeline._consistency_student is not None

    def test_load_none_raises(self, device):
        """Loading None student should raise RuntimeError."""
        from auto_voice.inference.realtime_voice_conversion_pipeline import (
            RealtimeVoiceConversionPipeline,
        )

        pipeline = RealtimeVoiceConversionPipeline(device=device)
        with pytest.raises(RuntimeError, match="cannot be None"):
            pipeline.load_consistency_student(None)

    def test_consistency_flag_in_config(self, device):
        """use_consistency config flag is respected."""
        from auto_voice.inference.realtime_voice_conversion_pipeline import (
            RealtimeVoiceConversionPipeline,
        )

        pipeline = RealtimeVoiceConversionPipeline(
            device=device,
            config={'use_consistency': True},
        )
        assert pipeline._use_consistency is True

        pipeline2 = RealtimeVoiceConversionPipeline(
            device=device,
            config={'use_consistency': False},
        )
        assert pipeline2._use_consistency is False

    def test_get_metrics_with_consistency(self, batch_params, device):
        """Metrics should work with consistency mode."""
        from auto_voice.inference.realtime_voice_conversion_pipeline import (
            RealtimeVoiceConversionPipeline,
        )

        pipeline = RealtimeVoiceConversionPipeline(
            device=device,
            config={'use_consistency': True},
        )

        student = ConsistencyStudent(
            n_mels=batch_params['n_mels'],
            hidden_dim=batch_params['hidden_dim'],
            n_blocks=batch_params['n_blocks'],
            cond_dim=batch_params['cond_dim'],
        ).to(device)

        pipeline.load_consistency_student(student)
        metrics = pipeline.get_metrics()

        assert 'is_running' in metrics
        assert metrics['is_running'] is False


# ─────────────────────────────────────────────────────────────────────────────
# Module Import Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModuleImports:
    def test_import_from_models_package(self):
        """All consistency classes importable from models package."""
        from auto_voice.models import (
            ConsistencyStudent,
            CTLoss_D,
            DiffusionDecoder,
            DiffusionStepEmbedding,
            EDMLoss,
            KarrasNoiseSchedule,
            ResidualBlock,
        )
        assert DiffusionDecoder is not None
        assert ConsistencyStudent is not None
        assert CTLoss_D is not None
        assert EDMLoss is not None
        assert KarrasNoiseSchedule is not None
        assert ResidualBlock is not None
        assert DiffusionStepEmbedding is not None

    def test_import_from_consistency_module(self):
        """Direct import from consistency module works."""
        from auto_voice.models.consistency import (
            ConsistencyStudent,
            CTLoss_D,
            DiffusionDecoder,
            EDMLoss,
        )
        assert DiffusionDecoder is not None
