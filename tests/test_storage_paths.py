from pathlib import Path

from auto_voice.storage.paths import (
    resolve_diarized_audio_dir,
    resolve_separated_audio_dir,
    resolve_training_vocals_dir,
    resolve_youtube_audio_dir,
)


def test_runtime_path_resolvers_use_data_dir_root():
    data_dir = Path("/tmp/autovoice-data")

    assert resolve_training_vocals_dir(data_dir=str(data_dir)) == data_dir / "training_vocals"
    assert resolve_youtube_audio_dir(data_dir=str(data_dir)) == data_dir / "youtube_audio"
    assert resolve_separated_audio_dir(data_dir=str(data_dir)) == data_dir / "separated_youtube"
    assert resolve_diarized_audio_dir(data_dir=str(data_dir)) == data_dir / "diarized_youtube"


def test_runtime_path_resolvers_append_artist_name_when_requested():
    data_dir = Path("/tmp/autovoice-data")

    assert resolve_youtube_audio_dir(data_dir=str(data_dir), artist_name="artist") == (
        data_dir / "youtube_audio" / "artist"
    )
    assert resolve_separated_audio_dir(data_dir=str(data_dir), artist_name="artist") == (
        data_dir / "separated_youtube" / "artist"
    )
    assert resolve_diarized_audio_dir(data_dir=str(data_dir), artist_name="artist") == (
        data_dir / "diarized_youtube" / "artist"
    )
