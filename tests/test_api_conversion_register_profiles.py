"""register_profiles: the per-request duet setting parsed at the conversion API.

Shape is {"female": <profile_id>|None, "male": <profile_id>|None}. A register
with no profile keeps the original singer's voice, so absent/all-null must
collapse to None (feature off, behaviour unchanged). A misspelled register or a
non-string id must raise rather than be dropped - a silently ignored register
converts a singer the operator asked to keep.
"""
import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from auto_voice.web.api_conversion import (
    _parse_register_profiles,
    _validate_register_profiles,
)


class _ProfileNotFound(Exception):
    pass


def _root(profiles):
    """Minimal stand-in for the api module: profile store + serializer."""
    def ensure(profile_id):
        if profile_id not in profiles:
            raise _ProfileNotFound(profile_id)
        return profiles[profile_id]

    return SimpleNamespace(
        _ensure_profile_in_store=ensure,
        _serialize_profile_for_response=lambda p: p,
        ProfileNotFoundError=_ProfileNotFound,
    )


def validate(profiles, parsed, fork_backed=True):
    """Run the validator with the Flask app context and fork registry stubbed."""
    with ExitStack() as stack:
        stack.enter_context(patch(
            'auto_voice.web.api_conversion.current_app',
            SimpleNamespace(config={'DATA_DIR': 'data'}),
        ))
        stack.enter_context(patch(
            'auto_voice.inference.svc_fork_bridge.is_available',
            lambda profile_id, data_dir='data': fork_backed,
        ))
        _validate_register_profiles(_root(profiles), parsed)


TRAINED = {'profile_role': 'target_user', 'has_trained_model': True}
UNTRAINED = {'profile_role': 'target_user', 'has_trained_model': False}
SOURCE_ARTIST = {'profile_role': 'source_artist', 'has_trained_model': True}


@pytest.mark.parametrize('raw', [None, '', {}, {'female': None, 'male': None},
                                 {'female': '', 'male': '  '}, '{}'])
def test_absent_or_all_null_is_feature_off(raw):
    assert _parse_register_profiles(raw) is None


def test_single_register_returns_both_keys():
    assert _parse_register_profiles({'female': 'abc'}) == {
        'female': 'abc', 'male': None}


def test_json_string_from_multipart_form_parses():
    raw = json.dumps({'female': 'abc', 'male': 'def'})
    assert _parse_register_profiles(raw) == {'female': 'abc', 'male': 'def'}


def test_ids_are_stripped():
    assert _parse_register_profiles({'male': ' def '})['male'] == 'def'


def test_unknown_register_rejected():
    with pytest.raises(ValueError, match='Unknown vocal register'):
        _parse_register_profiles({'femail': 'abc'})


@pytest.mark.parametrize('bad', [{'female': 123}, {'male': ['abc']},
                                 {'female': {'id': 'abc'}}])
def test_non_string_profile_id_rejected(bad):
    with pytest.raises(ValueError, match='must be a profile id string'):
        _parse_register_profiles(bad)


@pytest.mark.parametrize('bad', ['not json', '[1, 2]', '"abc"', [1, 2], 7])
def test_non_object_payload_rejected(bad):
    with pytest.raises(ValueError):
        _parse_register_profiles(bad)


def test_validate_accepts_trained_target_profiles():
    validate({'a': TRAINED, 'b': TRAINED}, {'female': 'a', 'male': 'b'})


def test_validate_skips_null_register():
    validate({'a': TRAINED}, {'female': 'a', 'male': None})


def test_validate_allows_same_profile_in_both_registers():
    validate({'a': TRAINED}, {'female': 'a', 'male': 'a'})


def test_validate_rejects_unknown_profile():
    with pytest.raises(ValueError, match='was not found'):
        validate({'a': TRAINED}, {'female': 'a', 'male': 'nope'})


def test_validate_rejects_untrained_profile():
    with pytest.raises(ValueError, match='no trained model'):
        validate({'a': UNTRAINED}, {'female': 'a', 'male': None})


def test_validate_rejects_source_artist_profile():
    with pytest.raises(ValueError, match='source artist'):
        validate({'a': SOURCE_ARTIST}, {'female': 'a', 'male': None})


def test_validate_rejects_non_fork_backed_profile():
    """The pipeline raises ConversionError on these mid-render; fail at submit."""
    with pytest.raises(ValueError, match='so-vits-svc-fork'):
        validate({'a': TRAINED}, {'female': 'a', 'male': None}, fork_backed=False)


def test_validate_noop_when_feature_off():
    validate({}, None, fork_backed=False)
