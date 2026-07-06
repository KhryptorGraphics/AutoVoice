"""Tests for shifts/overlap options on VocalSeparator.

Uses the module-level get_model/apply_model injection mechanism in
separation.py (mocks with an ``assert_called`` attribute take precedence
over the real demucs backend).
"""
import os

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

import auto_voice.audio.separation as separation
from auto_voice.audio.separation import VocalSeparator, recommended_model_name


@pytest.fixture
def mock_backend():
    """Inject module-level mock get_model/apply_model; restore after test."""
    saved_get, saved_apply = separation.get_model, separation.apply_model

    model = MagicMock()
    model.samplerate = 44100
    model.sources = ['drums', 'bass', 'other', 'vocals']

    separation.get_model = MagicMock(return_value=model)
    separation.apply_model = MagicMock()
    try:
        yield separation.get_model, separation.apply_model
    finally:
        separation.get_model = saved_get
        separation.apply_model = saved_apply


def _make_separator(mock_apply, **kwargs):
    separator = VocalSeparator(device=torch.device('cpu'), **kwargs)
    audio = np.random.randn(44100).astype(np.float32) * 0.3
    mock_apply.return_value = torch.randn(1, 4, 2, len(audio))
    separator.separate(audio, 44100)
    return mock_apply.call_args.kwargs


class TestSeparationOptions:
    def test_defaults_omit_shifts_and_overlap(self, mock_backend):
        """Backwards compat: default shifts=1/overlap=None pass no extra kwargs."""
        _, mock_apply = mock_backend
        kwargs = _make_separator(mock_apply)
        assert 'shifts' not in kwargs
        assert 'overlap' not in kwargs

    def test_shifts_reaches_apply_model(self, mock_backend):
        _, mock_apply = mock_backend
        kwargs = _make_separator(mock_apply, shifts=4)
        assert kwargs['shifts'] == 4

    def test_overlap_reaches_apply_model(self, mock_backend):
        _, mock_apply = mock_backend
        kwargs = _make_separator(mock_apply, overlap=0.5)
        assert kwargs['overlap'] == 0.5

    def test_all_options_combined(self, mock_backend):
        _, mock_apply = mock_backend
        kwargs = _make_separator(mock_apply, shifts=2, overlap=0.1, segment=7.8)
        assert kwargs['shifts'] == 2
        assert kwargs['overlap'] == 0.1
        assert kwargs['segment'] == 7.8

    def test_attributes_stored(self, mock_backend):
        separator = VocalSeparator(device=torch.device('cpu'),
                                   shifts=3, overlap=0.4)
        assert separator.shifts == 3
        assert separator.overlap == 0.4

    def test_default_model_name_unchanged(self, mock_backend):
        assert VocalSeparator(device=torch.device('cpu')).model_name == 'htdemucs'


class TestRecommendedModelName:
    def test_default(self, monkeypatch):
        monkeypatch.delenv('AUTOVOICE_SEPARATION_MODEL', raising=False)
        assert recommended_model_name() == 'htdemucs_ft'

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv('AUTOVOICE_SEPARATION_MODEL', 'mdx_extra')
        assert recommended_model_name() == 'mdx_extra'
