"""Tests for pipeline status API endpoint.

Coverage target: 80%

Covers:
- Pipeline status endpoint success scenarios
- Response structure validation
- Factory unavailable scenarios
- Exception handling
- Timestamp format validation
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone


class TestPipelineStatusEndpoint:
    """Test /api/v1/pipelines/status endpoint."""

    @pytest.mark.smoke
    def test_pipeline_status_success(self, client):
        """Test successful pipeline status retrieval."""
        # Mock PipelineFactory to avoid GPU dependencies
        mock_factory = MagicMock()
        mock_factory.get_status.return_value = {
            'realtime': {
                'loaded': True,
                'memory_gb': 1.2,
                'latency_target_ms': 100,
                'sample_rate': 22050,
                'description': 'Low-latency pipeline for live karaoke'
            },
            'quality': {
                'loaded': False,
                'memory_gb': 0.0,
                'latency_target_ms': 3000,
                'sample_rate': 24000,
                'description': 'High-quality CoMoSVC with 30-step diffusion'
            }
        }

        with patch('auto_voice.web.api.PIPELINE_FACTORY_AVAILABLE', True):
            with patch('auto_voice.web.api.PipelineFactory') as MockFactory:
                MockFactory.get_instance.return_value = mock_factory

                response = client.get('/api/v1/pipelines/status')

                assert response.status_code == 200
                data = response.get_json()

                # Verify response structure
                assert 'status' in data
                assert data['status'] == 'ok'
                assert 'timestamp' in data
                assert 'pipelines' in data

                # Verify pipelines data
                pipelines = data['pipelines']
                assert 'realtime' in pipelines
                assert 'quality' in pipelines
                assert pipelines['realtime']['loaded'] is True
                assert pipelines['quality']['loaded'] is False

    def test_pipeline_status_response_structure(self, client):
        """Test that response contains all required fields."""
        mock_factory = MagicMock()
        mock_factory.get_status.return_value = {
            'realtime': {
                'loaded': True,
                'memory_gb': 1.2,
                'latency_target_ms': 100,
                'sample_rate': 22050,
                'description': 'Low-latency pipeline'
            }
        }

        with patch('auto_voice.web.api.PIPELINE_FACTORY_AVAILABLE', True):
            with patch('auto_voice.web.api.PipelineFactory') as MockFactory:
                MockFactory.get_instance.return_value = mock_factory

                response = client.get('/api/v1/pipelines/status')

                assert response.status_code == 200
                data = response.get_json()

                # Verify required top-level fields
                required_fields = ['status', 'timestamp', 'pipelines']
                for field in required_fields:
                    assert field in data, f"Missing required field: {field}"

                # Verify pipeline fields
                pipeline = data['pipelines']['realtime']
                pipeline_fields = ['loaded', 'memory_gb', 'latency_target_ms', 'sample_rate', 'description']
                for field in pipeline_fields:
                    assert field in pipeline, f"Missing pipeline field: {field}"

    def test_pipeline_status_timestamp_format(self, client):
        """Test that timestamp is in correct ISO format with Z suffix."""
        mock_factory = MagicMock()
        mock_factory.get_status.return_value = {'realtime': {'loaded': False}}

        with patch('auto_voice.web.api.PIPELINE_FACTORY_AVAILABLE', True):
            with patch('auto_voice.web.api.PipelineFactory') as MockFactory:
                MockFactory.get_instance.return_value = mock_factory

                response = client.get('/api/v1/pipelines/status')

                assert response.status_code == 200
                data = response.get_json()

                timestamp = data.get('timestamp', '')
                # Should be ISO 8601 format with Z suffix
                assert 'T' in timestamp, "Timestamp should contain 'T' separator"
                assert timestamp.endswith('Z'), "Timestamp should end with 'Z' for UTC"

                # Verify it's a valid datetime
                try:
                    parsed = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    assert parsed.tzinfo is not None, "Timestamp should be timezone-aware"
                except ValueError:
                    pytest.fail(f"Invalid ISO 8601 timestamp format: {timestamp}")

    def test_pipeline_status_factory_unavailable(self, client):
        """Test response when PipelineFactory module is not available."""
        with patch('auto_voice.web.api.PIPELINE_FACTORY_AVAILABLE', False):
            response = client.get('/api/v1/pipelines/status')

            assert response.status_code == 503
            data = response.get_json()

            assert 'error' in data
            assert 'PipelineFactory unavailable' in data['error']
            assert 'message' in data

    def test_pipeline_status_factory_exception(self, client):
        """Test exception handling when factory raises an error."""
        mock_factory = MagicMock()
        mock_factory.get_status.side_effect = RuntimeError("Factory initialization failed")

        with patch('auto_voice.web.api.PIPELINE_FACTORY_AVAILABLE', True):
            with patch('auto_voice.web.api.PipelineFactory') as MockFactory:
                MockFactory.get_instance.return_value = mock_factory

                response = client.get('/api/v1/pipelines/status')

                assert response.status_code == 503
                data = response.get_json()

                assert 'error' in data
                assert 'Failed to get pipeline status' in data['error']
                assert 'message' in data
                assert 'Factory initialization failed' in data['message']

    def test_pipeline_status_multiple_pipelines(self, client):
        """Test response includes all pipeline types."""
        mock_factory = MagicMock()
        mock_factory.get_status.return_value = {
            'realtime': {
                'loaded': True,
                'memory_gb': 1.2,
                'latency_target_ms': 100,
                'sample_rate': 22050,
                'description': 'Realtime pipeline'
            },
            'quality': {
                'loaded': False,
                'memory_gb': 0.0,
                'latency_target_ms': 3000,
                'sample_rate': 24000,
                'description': 'Quality pipeline'
            },
            'quality_seedvc': {
                'loaded': True,
                'memory_gb': 2.5,
                'latency_target_ms': 2000,
                'sample_rate': 44100,
                'description': 'Quality SeedVC pipeline'
            }
        }

        with patch('auto_voice.web.api.PIPELINE_FACTORY_AVAILABLE', True):
            with patch('auto_voice.web.api.PipelineFactory') as MockFactory:
                MockFactory.get_instance.return_value = mock_factory

                response = client.get('/api/v1/pipelines/status')

                assert response.status_code == 200
                data = response.get_json()

                pipelines = data['pipelines']
                # Verify all three pipeline types are present
                assert 'realtime' in pipelines
                assert 'quality' in pipelines
                assert 'quality_seedvc' in pipelines

    def test_pipeline_status_loaded_states(self, client):
        """Test different loaded states for pipelines."""
        mock_factory = MagicMock()
        mock_factory.get_status.return_value = {
            'realtime': {'loaded': True, 'memory_gb': 1.2},
            'quality': {'loaded': False, 'memory_gb': 0.0}
        }

        with patch('auto_voice.web.api.PIPELINE_FACTORY_AVAILABLE', True):
            with patch('auto_voice.web.api.PipelineFactory') as MockFactory:
                MockFactory.get_instance.return_value = mock_factory

                response = client.get('/api/v1/pipelines/status')

                assert response.status_code == 200
                data = response.get_json()

                pipelines = data['pipelines']
                assert pipelines['realtime']['loaded'] is True
                assert pipelines['quality']['loaded'] is False

    def test_pipeline_status_memory_values(self, client):
        """Test that memory values are included and formatted correctly."""
        mock_factory = MagicMock()
        mock_factory.get_status.return_value = {
            'realtime': {
                'loaded': True,
                'memory_gb': 1.2,
                'latency_target_ms': 100,
                'sample_rate': 22050,
                'description': 'Test pipeline'
            }
        }

        with patch('auto_voice.web.api.PIPELINE_FACTORY_AVAILABLE', True):
            with patch('auto_voice.web.api.PipelineFactory') as MockFactory:
                MockFactory.get_instance.return_value = mock_factory

                response = client.get('/api/v1/pipelines/status')

                assert response.status_code == 200
                data = response.get_json()

                pipeline = data['pipelines']['realtime']
                assert 'memory_gb' in pipeline
                assert isinstance(pipeline['memory_gb'], (int, float))
                assert pipeline['memory_gb'] >= 0

    def test_pipeline_status_sample_rates(self, client):
        """Test that sample rates are included for all pipelines."""
        mock_factory = MagicMock()
        mock_factory.get_status.return_value = {
            'realtime': {
                'loaded': True,
                'sample_rate': 22050,
                'latency_target_ms': 100,
                'memory_gb': 1.0,
                'description': 'Realtime'
            },
            'quality': {
                'loaded': False,
                'sample_rate': 24000,
                'latency_target_ms': 3000,
                'memory_gb': 0.0,
                'description': 'Quality'
            }
        }

        with patch('auto_voice.web.api.PIPELINE_FACTORY_AVAILABLE', True):
            with patch('auto_voice.web.api.PipelineFactory') as MockFactory:
                MockFactory.get_instance.return_value = mock_factory

                response = client.get('/api/v1/pipelines/status')

                assert response.status_code == 200
                data = response.get_json()

                pipelines = data['pipelines']
                # Verify sample rates are present and valid
                assert pipelines['realtime']['sample_rate'] == 22050
                assert pipelines['quality']['sample_rate'] == 24000

    def test_pipeline_status_latency_targets(self, client):
        """Test that latency targets are included for all pipelines."""
        mock_factory = MagicMock()
        mock_factory.get_status.return_value = {
            'realtime': {
                'loaded': True,
                'latency_target_ms': 100,
                'sample_rate': 22050,
                'memory_gb': 1.0,
                'description': 'Realtime'
            },
            'quality': {
                'loaded': False,
                'latency_target_ms': 3000,
                'sample_rate': 24000,
                'memory_gb': 0.0,
                'description': 'Quality'
            }
        }

        with patch('auto_voice.web.api.PIPELINE_FACTORY_AVAILABLE', True):
            with patch('auto_voice.web.api.PipelineFactory') as MockFactory:
                MockFactory.get_instance.return_value = mock_factory

                response = client.get('/api/v1/pipelines/status')

                assert response.status_code == 200
                data = response.get_json()

                pipelines = data['pipelines']
                # Verify latency targets are present and valid
                assert pipelines['realtime']['latency_target_ms'] == 100
                assert pipelines['quality']['latency_target_ms'] == 3000
                assert isinstance(pipelines['realtime']['latency_target_ms'], int)

    def test_pipeline_status_empty_pipelines(self, client):
        """Test response when no pipelines are available."""
        mock_factory = MagicMock()
        mock_factory.get_status.return_value = {}

        with patch('auto_voice.web.api.PIPELINE_FACTORY_AVAILABLE', True):
            with patch('auto_voice.web.api.PipelineFactory') as MockFactory:
                MockFactory.get_instance.return_value = mock_factory

                response = client.get('/api/v1/pipelines/status')

                assert response.status_code == 200
                data = response.get_json()

                assert 'pipelines' in data
                assert isinstance(data['pipelines'], dict)
                assert len(data['pipelines']) == 0

    def test_pipeline_status_get_instance_called(self, client):
        """Test that PipelineFactory.get_instance() is called correctly."""
        mock_factory = MagicMock()
        mock_factory.get_status.return_value = {'realtime': {'loaded': False}}

        with patch('auto_voice.web.api.PIPELINE_FACTORY_AVAILABLE', True):
            with patch('auto_voice.web.api.PipelineFactory') as MockFactory:
                MockFactory.get_instance.return_value = mock_factory

                response = client.get('/api/v1/pipelines/status')

                assert response.status_code == 200
                # Verify get_instance was called
                MockFactory.get_instance.assert_called_once()
                # Verify get_status was called on the instance
                mock_factory.get_status.assert_called_once()

    def test_pipeline_status_descriptions_present(self, client):
        """Test that pipeline descriptions are included."""
        mock_factory = MagicMock()
        mock_factory.get_status.return_value = {
            'realtime': {
                'loaded': True,
                'memory_gb': 1.2,
                'latency_target_ms': 100,
                'sample_rate': 22050,
                'description': 'Low-latency pipeline for live karaoke'
            }
        }

        with patch('auto_voice.web.api.PIPELINE_FACTORY_AVAILABLE', True):
            with patch('auto_voice.web.api.PipelineFactory') as MockFactory:
                MockFactory.get_instance.return_value = mock_factory

                response = client.get('/api/v1/pipelines/status')

                assert response.status_code == 200
                data = response.get_json()

                pipeline = data['pipelines']['realtime']
                assert 'description' in pipeline
                assert isinstance(pipeline['description'], str)
                assert len(pipeline['description']) > 0
