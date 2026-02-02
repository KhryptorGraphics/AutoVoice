"""Browser automation tests for voice profile training UI workflow.

Uses xdotool + Chromium on VNC display for visual validation.
Tests the complete UI workflow from user perspective.

VNC Display: :99 (accessible at http://192.168.1.64:16080/vnc.html)
"""

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


# ============================================================================
# Configuration
# ============================================================================

VNC_DISPLAY = ':99'
APP_URL = 'http://localhost:5000'
TEST_TIMEOUT = 300  # 5 minutes


def run_xdo(command: str) -> str:
    """Run xdotool command on VNC display."""
    full_cmd = f'DISPLAY={VNC_DISPLAY} xdotool {command}'
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()


def launch_browser(url: str) -> None:
    """Launch Chromium on VNC display."""
    cmd = f'DISPLAY={VNC_DISPLAY} chromium-browser --no-sandbox --disable-gpu --start-maximized "{url}" &'
    subprocess.Popen(cmd, shell=True)
    time.sleep(5)  # Wait for browser to start


def find_browser_window() -> str:
    """Find Chromium window ID."""
    window_id = run_xdo('search --name "Chromium" | head -1')
    if not window_id:
        window_id = run_xdo('search --name "AutoVoice" | head -1')
    return window_id


def click_at(x: int, y: int) -> None:
    """Click at coordinates on VNC display."""
    run_xdo(f'mousemove {x} {y} click 1')
    time.sleep(0.5)


def type_text(text: str) -> None:
    """Type text on VNC display."""
    run_xdo(f'type "{text}"')
    time.sleep(0.3)


def press_key(key: str) -> None:
    """Press a key on VNC display."""
    run_xdo(f'key {key}')
    time.sleep(0.3)


def screenshot(filename: str) -> None:
    """Take screenshot of VNC display."""
    cmd = f'DISPLAY={VNC_DISPLAY} import -window root {filename}'
    subprocess.run(cmd, shell=True)


def close_browser() -> None:
    """Close Chromium browser."""
    run_xdo('search --name "Chromium" windowkill')
    time.sleep(1)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_rate():
    """Audio sample rate."""
    return 24000


@pytest.fixture
def test_audio_dir(tmp_path):
    """Create test audio files."""
    audio_dir = tmp_path / 'test_audio'
    audio_dir.mkdir()

    # Generate 3 voice samples
    for i in range(3):
        audio = generate_test_audio(10.0, 24000, base_freq=440.0 + i * 20)
        sf.write(audio_dir / f'sample_{i}.wav', audio, 24000)

    # Generate a song for conversion
    song = generate_test_audio(8.0, 24000, base_freq=330.0)
    sf.write(audio_dir / 'test_song.wav', song, 24000)

    return audio_dir


def generate_test_audio(duration: float, sr: int, base_freq: float = 440.0) -> np.ndarray:
    """Generate synthetic voice audio."""
    t = np.linspace(0, duration, int(duration * sr), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * base_freq * t)
    audio += 0.25 * np.sin(2 * np.pi * base_freq * 2 * t)
    return audio.astype(np.float32)


@pytest.fixture
def browser_session():
    """Setup and teardown browser session."""
    # Ensure VNC is running
    vnc_check = subprocess.run(
        f'DISPLAY={VNC_DISPLAY} xdpyinfo &> /dev/null',
        shell=True
    )
    if vnc_check.returncode != 0:
        pytest.skip("VNC display :99 not available")

    # Launch browser
    launch_browser(APP_URL)

    yield

    # Cleanup
    close_browser()


# ============================================================================
# Test Cases
# ============================================================================

@pytest.mark.browser
@pytest.mark.slow
class TestVoiceProfilePageUI:
    """Test VoiceProfilePage UI workflow (Phase 1)."""

    def test_navigate_to_profiles_page(self, browser_session):
        """Test Task 1.1: Navigate to profiles page."""
        # Wait for page load
        time.sleep(3)

        # Click on "Voice Profiles" link/button
        # Coordinates depend on UI layout - adjust as needed
        click_at(200, 100)  # Example: navigation menu
        time.sleep(2)

        screenshot('/tmp/profiles_page.png')

    def test_create_new_profile(self, browser_session, test_audio_dir):
        """Test Task 1.1: Create profile via UI."""
        time.sleep(3)

        # Navigate to profiles
        click_at(200, 100)
        time.sleep(2)

        # Click "Create Profile" button
        click_at(960, 200)  # Center-top area
        time.sleep(1)

        # Enter profile name
        type_text('Browser Test Artist')
        press_key('Tab')
        time.sleep(1)

        # Upload sample file
        # Click file input
        click_at(960, 400)
        time.sleep(1)

        # In file dialog, type path
        press_key('ctrl+l')  # Focus location bar
        type_text(str(test_audio_dir / 'sample_0.wav'))
        press_key('Return')
        time.sleep(2)

        # Submit form
        press_key('Return')
        time.sleep(3)

        screenshot('/tmp/profile_created.png')

    def test_upload_additional_samples(self, browser_session, test_audio_dir):
        """Test Task 1.2: Upload samples to existing profile."""
        time.sleep(3)

        # Navigate to profiles
        click_at(200, 100)
        time.sleep(2)

        # Click on first profile in list
        click_at(960, 350)
        time.sleep(2)

        # Click "Upload Sample" button
        click_at(1200, 300)
        time.sleep(1)

        # Upload file
        click_at(960, 400)
        time.sleep(1)
        press_key('ctrl+l')
        type_text(str(test_audio_dir / 'sample_1.wav'))
        press_key('Return')
        time.sleep(2)

        screenshot('/tmp/samples_uploaded.png')


@pytest.mark.browser
@pytest.mark.slow
class TestTrainingConfigUI:
    """Test TrainingConfigPanel UI (Phase 2)."""

    def test_configure_training_settings(self, browser_session):
        """Test Task 2.1: Configure training parameters."""
        time.sleep(3)

        # Navigate to profile with samples
        click_at(200, 100)
        time.sleep(2)
        click_at(960, 350)  # Select profile
        time.sleep(2)

        # Switch to Config tab
        click_at(600, 250)  # Config tab
        time.sleep(1)

        # Adjust epoch slider
        click_at(800, 400)  # Epochs slider
        time.sleep(0.5)

        # Adjust learning rate
        click_at(800, 500)  # Learning rate control
        time.sleep(0.5)

        screenshot('/tmp/training_config.png')

    def test_start_training_job(self, browser_session):
        """Test Task 2.2: Start training job."""
        time.sleep(3)

        # Navigate to profile
        click_at(200, 100)
        time.sleep(2)
        click_at(960, 350)
        time.sleep(2)

        # Click "Start Training" button
        click_at(960, 600)
        time.sleep(2)

        # Verify training started
        # Should see progress indicator
        screenshot('/tmp/training_started.png')

    def test_monitor_training_progress(self, browser_session):
        """Test Task 2.3: Monitor training progress."""
        time.sleep(3)

        # Navigate to profile
        click_at(200, 100)
        time.sleep(2)
        click_at(960, 350)
        time.sleep(2)

        # Switch to Jobs tab
        click_at(800, 250)
        time.sleep(1)

        # Take screenshots over time to show progress
        for i in range(5):
            time.sleep(10)
            screenshot(f'/tmp/training_progress_{i}.png')

            # Check if training completed
            # Look for completion indicator


@pytest.mark.browser
@pytest.mark.slow
class TestDiarizationUI:
    """Test diarization UI workflow (Phase 1)."""

    def test_run_diarization(self, browser_session, test_audio_dir):
        """Test Task 1.3: Run diarization on audio."""
        time.sleep(3)

        # Navigate to diarization page/section
        click_at(300, 100)  # Diarization link
        time.sleep(2)

        # Upload multi-speaker audio
        click_at(960, 300)
        time.sleep(1)
        press_key('ctrl+l')
        type_text(str(test_audio_dir / 'test_song.wav'))
        press_key('Return')
        time.sleep(2)

        # Click "Run Diarization"
        click_at(960, 400)
        time.sleep(5)  # Wait for processing

        screenshot('/tmp/diarization_result.png')

    def test_assign_segment_to_profile(self, browser_session):
        """Test Task 1.4: Assign diarized segment to profile."""
        time.sleep(3)

        # After diarization completes
        # Click on a speaker segment
        click_at(500, 600)
        time.sleep(1)

        # Click "Assign to Profile"
        click_at(960, 700)
        time.sleep(1)

        # Select profile from dropdown
        click_at(960, 400)
        time.sleep(1)
        press_key('Down')
        press_key('Return')
        time.sleep(2)

        screenshot('/tmp/segment_assigned.png')


@pytest.mark.browser
@pytest.mark.slow
class TestYouTubeMultiArtistUI:
    """Test YouTube multi-artist workflow (Phase 3)."""

    def test_youtube_url_input(self, browser_session):
        """Test Task 3.1: Enter YouTube URL with featured artists."""
        time.sleep(3)

        # Navigate to YouTube download page
        click_at(400, 100)
        time.sleep(2)

        # Enter YouTube URL
        click_at(960, 300)
        type_text('https://www.youtube.com/watch?v=dQw4w9WgXcQ')
        time.sleep(2)

        # Click "Detect Artists" or similar
        click_at(960, 400)
        time.sleep(3)

        screenshot('/tmp/youtube_artists_detected.png')

    def test_create_profiles_for_artists(self, browser_session):
        """Test Task 3.4: Auto-create profiles from detected artists."""
        time.sleep(3)

        # After artist detection
        # Click "Create Profiles for All Artists"
        click_at(960, 600)
        time.sleep(3)

        # Verify profiles created
        screenshot('/tmp/artist_profiles_created.png')


@pytest.mark.browser
@pytest.mark.slow
class TestKaraokeWithAdapter:
    """Test karaoke page with trained adapter (Phase 4)."""

    def test_select_trained_profile_in_karaoke(self, browser_session):
        """Test Task 4.1: Use trained profile in realtime pipeline."""
        time.sleep(3)

        # Navigate to karaoke page
        click_at(500, 100)
        time.sleep(2)

        # Select profile with trained adapter
        click_at(300, 300)  # Profile dropdown
        time.sleep(1)
        press_key('Down')
        press_key('Return')
        time.sleep(2)

        # Verify adapter info is shown
        screenshot('/tmp/karaoke_with_adapter.png')

    def test_start_karaoke_session(self, browser_session):
        """Test starting karaoke session with adapter."""
        time.sleep(3)

        # Navigate to karaoke
        click_at(500, 100)
        time.sleep(2)

        # Select profile
        click_at(300, 300)
        time.sleep(1)
        press_key('Down')
        press_key('Return')
        time.sleep(2)

        # Upload/select song
        click_at(960, 400)
        time.sleep(2)

        # Start session
        click_at(960, 600)
        time.sleep(3)

        screenshot('/tmp/karaoke_session_active.png')


# ============================================================================
# Utility Tests
# ============================================================================

@pytest.mark.browser
class TestBrowserSetup:
    """Verify browser automation is working."""

    def test_vnc_display_available(self):
        """Check VNC display is accessible."""
        result = subprocess.run(
            f'DISPLAY={VNC_DISPLAY} xdpyinfo',
            shell=True,
            capture_output=True
        )
        assert result.returncode == 0, "VNC display :99 not running"

    def test_xdotool_available(self):
        """Check xdotool is installed."""
        result = subprocess.run('which xdotool', shell=True, capture_output=True)
        assert result.returncode == 0, "xdotool not installed"

    def test_chromium_available(self):
        """Check Chromium is installed."""
        result = subprocess.run('which chromium-browser', shell=True, capture_output=True)
        assert result.returncode == 0, "chromium-browser not installed"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'browser'])
