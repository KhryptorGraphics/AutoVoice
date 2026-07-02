"""Ingest speaker suggestions must match against ALL profile roles.

A diarized voice belonging to a known target_user profile (e.g. the user's
own trained voice) must be recognized so its segments can be assigned to
that profile — previously suggestions were filtered to source_artist only.
"""
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from auto_voice.storage.voice_profiles import VoiceProfileStore
from auto_voice.web.api_youtube import _build_ingest_suggestions


@pytest.fixture
def store(tmp_path):
    return VoiceProfileStore(
        profiles_dir=str(tmp_path / 'profiles'),
        samples_dir=str(tmp_path / 'samples'),
    )


def _make_profile(store, name, role, embedding):
    profile_id = str(uuid.uuid4())
    store.save({
        'profile_id': profile_id,
        'user_id': 'op',
        'name': name,
        'profile_role': role,
    })
    store.save_speaker_embedding(profile_id, embedding)
    return profile_id


def test_target_user_profiles_are_match_candidates(store):
    rng = np.random.default_rng(3)
    voice = rng.standard_normal(512).astype(np.float32)
    voice /= np.linalg.norm(voice)
    other = rng.standard_normal(512).astype(np.float32)
    other /= np.linalg.norm(other)

    target_id = _make_profile(store, 'Target User', 'target_user', voice)
    _make_profile(store, 'Some Source Artist', 'source_artist', other)

    diarizer = MagicMock()
    diarizer.extract_speaker_embedding.return_value = voice

    suggestions = _build_ingest_suggestions(
        diarizer=diarizer,
        profile_store=store,
        vocals_path='/tmp/vocals.wav',
        segments=[{'speaker_id': 'SPEAKER_00', 'start': 0.0, 'end': 10.0, 'duration': 10.0}],
        metadata={'main_artist': 'Somebody', 'featured_artists': []},
    )

    assert len(suggestions) == 1
    matches = suggestions[0]['matches']
    match_ids = [m['profile_id'] for m in matches]
    assert target_id in match_ids, 'target_user profile must be a match candidate'
    best = matches[0]
    assert best['profile_id'] == target_id
    assert best['profile_role'] == 'target_user'
    assert suggestions[0]['recommended_action'] == 'assign_existing'
    assert suggestions[0]['recommended_profile_id'] == target_id
