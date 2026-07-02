"""Sample ids must never collide after deletions (count-based ids overwrote samples)."""
import os
import uuid

import numpy as np
import pytest
import soundfile as sf

from auto_voice.storage.voice_profiles import VoiceProfileStore


@pytest.fixture
def store(tmp_path):
    return VoiceProfileStore(
        profiles_dir=str(tmp_path / 'profiles'),
        samples_dir=str(tmp_path / 'samples'),
    )


def _wav(tmp_path):
    path = os.path.join(str(tmp_path), 'a.wav')
    sf.write(path, (np.random.randn(22050) * 0.1).astype('float32'), 22050)
    return path


def test_sample_ids_do_not_collide_after_deletion(store, tmp_path):
    pid = store.save({'profile_id': str(uuid.uuid4()), 'user_id': 't', 'name': 'T'})
    wav = _wav(tmp_path)

    ids = [
        store.add_training_sample(profile_id=pid, vocals_path=wav, duration=1.0, source_file='x').sample_id
        for _ in range(3)
    ]
    assert ids == ['sample_001', 'sample_002', 'sample_003']

    store.delete_training_sample(pid, ids[1])
    new_id = store.add_training_sample(
        profile_id=pid, vocals_path=wav, duration=1.0, source_file='x'
    ).sample_id

    # count-based generation would reissue sample_003 and overwrite it
    assert new_id == 'sample_004'
    surviving = {s.sample_id for s in store.list_training_samples(pid)}
    assert surviving == {'sample_001', 'sample_003', 'sample_004'}
