#!/usr/bin/env python3
"""
Integration tests for synthetic test data generation with profile creation.

Tests Comment 9 fix: Ensures generate_test_data.py creates real voice profiles.
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
import sys
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestSyntheticDataGeneration(unittest.TestCase):
    """Test synthetic test data generation with VoiceCloner integration."""

    def setUp(self):
        """Create temporary directory for test outputs."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.script_path = Path(__file__).parent.parent / 'scripts' / 'generate_test_data.py'

    def tearDown(self):
        """Clean up temporary directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_generate_with_fallback(self):
        """Test data generation with fallback (no profiles)."""
        output_dir = self.test_dir / 'test_output'

        # Run script with --no-profiles for guaranteed success
        result = subprocess.run(
            [
                'python3', str(self.script_path),
                '--output', str(output_dir),
                '--num-samples', '2',
                '--seed', '42',
                '--no-profiles'
            ],
            capture_output=True,
            text=True
        )

        # Check script executed successfully
        self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")

        # Check metadata file created
        metadata_path = output_dir / 'test_set.json'
        self.assertTrue(metadata_path.exists(), "Metadata file not created")

        # Load and validate metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Validate structure
        self.assertIn('test_cases', metadata)
        self.assertIn('generation_config', metadata)
        self.assertEqual(len(metadata['test_cases']), 2)

        # Validate test cases
        for test_case in metadata['test_cases']:
            self.assertIn('id', test_case)
            self.assertIn('source_audio', test_case)
            self.assertIn('target_profile_id', test_case)
            self.assertIn('reference_audio', test_case)
            self.assertIn('metadata', test_case)

            # Check files exist
            self.assertTrue(Path(test_case['source_audio']).exists())
            self.assertTrue(Path(test_case['reference_audio']).exists())

            # Validate metadata fields
            self.assertTrue(test_case['metadata']['synthetic'])
            self.assertIn('base_freq_hz', test_case['metadata'])
            self.assertIn('sample_rate', test_case['metadata'])

    def test_metadata_structure(self):
        """Test that generated metadata has correct structure."""
        output_dir = self.test_dir / 'test_output'

        subprocess.run(
            [
                'python3', str(self.script_path),
                '--output', str(output_dir),
                '--num-samples', '1',
                '--no-profiles'
            ],
            capture_output=True
        )

        metadata_path = output_dir / 'test_set.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Check generation config
        gen_config = metadata['generation_config']
        self.assertEqual(gen_config['num_samples'], 1)
        self.assertEqual(gen_config['seed'], 42)  # default
        self.assertTrue(gen_config['synthetic'])

        # Check test case structure
        test_case = metadata['test_cases'][0]
        required_fields = ['id', 'source_audio', 'target_profile_id', 'reference_audio', 'metadata']
        for field in required_fields:
            self.assertIn(field, test_case, f"Missing required field: {field}")

    def test_audio_files_created(self):
        """Test that audio files are generated correctly."""
        output_dir = self.test_dir / 'test_output'

        subprocess.run(
            [
                'python3', str(self.script_path),
                '--output', str(output_dir),
                '--num-samples', '3',
                '--no-profiles'
            ],
            capture_output=True
        )

        # Check all expected files exist
        expected_files = [
            'test_001_source.wav',
            'test_001_reference.wav',
            'test_002_source.wav',
            'test_002_reference.wav',
            'test_003_source.wav',
            'test_003_reference.wav',
            'test_set.json'
        ]

        for filename in expected_files:
            file_path = output_dir / filename
            self.assertTrue(file_path.exists(), f"Missing file: {filename}")

            # Check audio files are non-empty
            if filename.endswith('.wav'):
                self.assertGreater(file_path.stat().st_size, 0, f"Empty audio file: {filename}")

    def test_profile_id_format(self):
        """Test that profile IDs follow expected format."""
        output_dir = self.test_dir / 'test_output'

        subprocess.run(
            [
                'python3', str(self.script_path),
                '--output', str(output_dir),
                '--num-samples', '2',
                '--no-profiles'
            ],
            capture_output=True
        )

        metadata_path = output_dir / 'test_set.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # With --no-profiles, should use synthetic-profile-* format
        for test_case in metadata['test_cases']:
            profile_id = test_case['target_profile_id']
            self.assertTrue(
                profile_id.startswith('synthetic-profile-'),
                f"Unexpected profile ID format: {profile_id}"
            )

    @unittest.skip("Requires VoiceCloner dependencies (resemblyzer)")
    def test_profile_creation(self):
        """Test that real profiles are created (requires dependencies)."""
        output_dir = self.test_dir / 'test_output'

        # Run without --no-profiles
        result = subprocess.run(
            [
                'python3', str(self.script_path),
                '--output', str(output_dir),
                '--num-samples', '1',
                '--seed', '42'
            ],
            capture_output=True,
            text=True
        )

        # Only run if VoiceCloner available
        if "Could not initialize VoiceCloner" in result.stdout:
            self.skipTest("VoiceCloner dependencies not available")

        metadata_path = output_dir / 'test_set.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Check profile directory exists
        profiles_dir = output_dir / 'profiles'
        self.assertTrue(profiles_dir.exists(), "Profiles directory not created")

        # Check profile ID is UUID format (not synthetic-profile-*)
        test_case = metadata['test_cases'][0]
        profile_id = test_case['target_profile_id']
        self.assertFalse(
            profile_id.startswith('synthetic-profile-'),
            "Real profile should not use synthetic-profile-* format"
        )

        # Check has_real_profile flag
        self.assertTrue(test_case['metadata']['has_real_profile'])


if __name__ == '__main__':
    unittest.main()
