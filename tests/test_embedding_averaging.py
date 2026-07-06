"""Pure-numpy unit tests for multi-clip embedding averaging.

Covers VoiceProfileStore.recompute_profile_embedding (and the underlying
aggregate_embeddings helper) using a tmp_path-backed store:

(a) two orthogonal unit vectors, equal weights -> unit-norm, equidistant
(b) weights respected (3:1 pulls toward the first vector)
(c) single embedding -> same-direction unit vector
(d) persisted .npy readable and metadata updated
(e) zero-vector input handled without NaN
"""
import json
import os

import numpy as np
import pytest

from auto_voice.storage.voice_profiles import (
    ProfileNotFoundError,
    VoiceProfileStore,
    aggregate_embeddings,
)


@pytest.fixture
def store(tmp_path):
    """VoiceProfileStore rooted in a temporary directory."""
    return VoiceProfileStore(
        profiles_dir=str(tmp_path / "profiles"),
        samples_dir=str(tmp_path / "samples"),
        trained_models_dir=str(tmp_path / "trained_models"),
    )


@pytest.fixture
def profile_id(store):
    """A minimal saved profile to recompute embeddings for."""
    return store.save({
        'name': 'averaging-test',
        'user_id': 'user-avg',
        'embedding': np.ones(8, dtype=np.float32),
    })


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


class TestOrthogonalEqualWeights:
    """(a) Two orthogonal unit vectors with equal weights."""

    def test_result_is_unit_norm(self, store, profile_id):
        e1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        e2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        result = store.recompute_profile_embedding(profile_id, [e1, e2])
        assert result.dtype == np.float32
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-6)

    def test_result_is_equidistant(self, store, profile_id):
        e1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        e2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        result = store.recompute_profile_embedding(profile_id, [e1, e2])
        # Equal cosine similarity to both inputs (1/sqrt(2) each)
        cos1 = float(np.dot(result, unit(e1)))
        cos2 = float(np.dot(result, unit(e2)))
        assert np.isclose(cos1, cos2, atol=1e-6)
        assert np.isclose(cos1, 1.0 / np.sqrt(2.0), atol=1e-6)

    def test_explicit_equal_weights_match_default(self, store, profile_id):
        e1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        e2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        r_default = store.recompute_profile_embedding(profile_id, [e1, e2])
        r_equal = store.recompute_profile_embedding(profile_id, [e1, e2], weights=[5.0, 5.0])
        np.testing.assert_allclose(r_default, r_equal, atol=1e-6)


class TestWeightsRespected:
    """(b) Weight 3:1 pulls the aggregate toward the first vector."""

    def test_three_to_one_pulls_toward_first(self, store, profile_id):
        e1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        e2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        result = store.recompute_profile_embedding(profile_id, [e1, e2], weights=[3.0, 1.0])
        cos1 = float(np.dot(result, unit(e1)))
        cos2 = float(np.dot(result, unit(e2)))
        assert cos1 > cos2
        # Expected direction: (3*e1 + 1*e2)/4, renormalized
        expected = unit(3.0 * unit(e1) + 1.0 * unit(e2))
        np.testing.assert_allclose(result, expected.astype(np.float32), atol=1e-6)

    def test_normalization_before_weighting(self, store, profile_id):
        # A clip with a huge raw magnitude must NOT dominate beyond its
        # weight: embeddings are unit-normalized before averaging.
        e1 = np.array([1000.0, 0.0, 0.0, 0.0], dtype=np.float32)
        e2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        result = store.recompute_profile_embedding(profile_id, [e1, e2], weights=[1.0, 1.0])
        cos1 = float(np.dot(result, unit(e1)))
        cos2 = float(np.dot(result, unit(e2)))
        assert np.isclose(cos1, cos2, atol=1e-6)


class TestSingleEmbedding:
    """(c) Single embedding -> same-direction unit vector."""

    def test_direction_preserved(self, store, profile_id):
        e = np.array([3.0, 4.0, 0.0], dtype=np.float32)  # norm 5
        result = store.recompute_profile_embedding(profile_id, [e])
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-6)
        np.testing.assert_allclose(result, unit(e).astype(np.float32), atol=1e-6)

    def test_unit_input_roundtrips(self, store, profile_id):
        e = unit(np.array([1.0, 2.0, 3.0, 4.0])).astype(np.float32)
        result = store.recompute_profile_embedding(profile_id, [e])
        np.testing.assert_allclose(result, e, atol=1e-6)


class TestPersistenceAndMetadata:
    """(d) Persisted .npy readable and metadata updated."""

    def test_npy_persisted_and_readable(self, store, profile_id):
        e1 = np.array([1.0, 0.0], dtype=np.float32)
        e2 = np.array([0.0, 1.0], dtype=np.float32)
        result = store.recompute_profile_embedding(profile_id, [e1, e2], weights=[2.0, 1.0])

        emb_path = store._embedding_path(profile_id)
        assert os.path.exists(emb_path)
        loaded = np.load(emb_path)
        np.testing.assert_allclose(loaded, result, atol=1e-7)

        # load() surfaces the persisted embedding too
        profile = store.load(profile_id)
        np.testing.assert_allclose(profile['embedding'], result, atol=1e-7)

    def test_metadata_updated(self, store, profile_id):
        e1 = np.array([1.0, 0.0], dtype=np.float32)
        e2 = np.array([0.0, 1.0], dtype=np.float32)
        e3 = np.array([1.0, 1.0], dtype=np.float32)
        store.recompute_profile_embedding(profile_id, [e1, e2, e3])

        with open(store._profile_path(profile_id)) as f:
            raw = json.load(f)
        assert raw['embedding_clip_count'] == 3
        assert raw['embedding_aggregation'] == 'l2_weighted_mean'
        # Pre-existing metadata survives the read-modify-write
        assert raw['name'] == 'averaging-test'

    def test_missing_profile_raises(self, store):
        with pytest.raises(ProfileNotFoundError):
            store.recompute_profile_embedding(
                'no-such-profile', [np.ones(4, dtype=np.float32)]
            )


class TestZeroVectorHandling:
    """(e) Zero-vector input handled without NaN (degenerate clips skipped)."""

    def test_zero_vector_skipped_no_nan(self, store, profile_id):
        zero = np.zeros(4, dtype=np.float32)
        e = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        result = store.recompute_profile_embedding(profile_id, [zero, e], weights=[10.0, 1.0])
        assert not np.any(np.isnan(result))
        # The zero (silent) clip is skipped; result follows the valid clip
        np.testing.assert_allclose(result, unit(e).astype(np.float32), atol=1e-6)

    def test_all_zero_raises_instead_of_nan(self, store, profile_id):
        zeros = [np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)]
        with pytest.raises(ValueError):
            store.recompute_profile_embedding(profile_id, zeros)

    def test_empty_embeddings_raises(self, store, profile_id):
        with pytest.raises(ValueError):
            store.recompute_profile_embedding(profile_id, [])

    def test_mismatched_weights_raises(self, store, profile_id):
        e = np.ones(4, dtype=np.float32)
        with pytest.raises(ValueError):
            store.recompute_profile_embedding(profile_id, [e, e], weights=[1.0])


class TestAggregateHelperDirect:
    """Sanity checks on the shared aggregate_embeddings helper."""

    def test_helper_matches_store_result(self, store, profile_id):
        e1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        e2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        direct = aggregate_embeddings([e1, e2], [3.0, 1.0])
        via_store = store.recompute_profile_embedding(profile_id, [e1, e2], weights=[3.0, 1.0])
        np.testing.assert_allclose(direct, via_store, atol=1e-7)

    def test_negative_weight_rejected(self):
        e = np.ones(3, dtype=np.float32)
        with pytest.raises(ValueError):
            aggregate_embeddings([e, e], [1.0, -1.0])
