"""Comprehensive tests for speaker API endpoints.

Tests cover:
- POST /api/v1/speakers/extraction/run - Trigger speaker extraction
- GET /api/v1/speakers/extraction/status/<job_id> - Get extraction status
- GET /api/v1/speakers/tracks - List tracks with filtering
- GET /api/v1/speakers/tracks/<track_id> - Get track details
- POST /api/v1/speakers/tracks/fetch-metadata - Fetch YouTube metadata
- GET /api/v1/speakers/clusters - List speaker clusters
- GET /api/v1/speakers/clusters/<cluster_id> - Get cluster details
- PUT /api/v1/speakers/clusters/<cluster_id>/name - Update cluster name
- POST /api/v1/speakers/clusters/merge - Merge clusters
- POST /api/v1/speakers/clusters/split - Split cluster
- POST /api/v1/speakers/clusters/<cluster_id>/members - Add cluster members
- DELETE /api/v1/speakers/clusters/<cluster_id>/members/<embedding_id> - Remove cluster member
- GET /api/v1/speakers/clusters/<cluster_id>/sample - Get cluster audio sample
- POST /api/v1/speakers/identify - Run speaker identification
- GET /api/v1/speakers/featured-artists - List featured artists
- POST /api/v1/speakers/organize - Organize files by artist

Target: 18% → 90% coverage (speaker_api.py)
"""
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def app_with_db():
    """Create Flask app with speaker API blueprint."""
    from flask import Flask
    from auto_voice.web.speaker_api import speaker_bp

    app = Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(speaker_bp)

    return app


@pytest.fixture
def client(app_with_db):
    """Flask test client."""
    return app_with_db.test_client()


@pytest.fixture
def mock_db_ops():
    """Mock database operations."""
    mock_ops = {
        'get_all_tracks': MagicMock(return_value=[
            {
                'id': 'track1',
                'title': 'Test Song 1',
                'artist_name': 'conor_maynard',
                'duration_sec': 180.0,
                'channel': 'Test Channel',
            },
            {
                'id': 'track2',
                'title': 'Test Song 2 ft. Artist',
                'artist_name': 'conor_maynard',
                'duration_sec': 200.0,
                'channel': 'Test Channel',
            },
        ]),
        'get_track': MagicMock(return_value={
            'id': 'track1',
            'title': 'Test Song',
            'artist_name': 'conor_maynard',
            'duration_sec': 180.0,
        }),
        'get_tracks_by_artist': MagicMock(return_value=[
            {
                'id': 'track1',
                'title': 'Test Song',
                'artist_name': 'conor_maynard',
                'duration_sec': 180.0,
            }
        ]),
        'get_all_clusters': MagicMock(return_value=[
            {
                'cluster_id': 'cluster1',
                'name': 'Artist 1',
                'member_count': 5,
                'is_verified': True,
            },
            {
                'cluster_id': 'cluster2',
                'name': 'Unknown',
                'member_count': 3,
                'is_verified': False,
            },
        ]),
        'get_cluster': MagicMock(return_value={
            'cluster_id': 'cluster1',
            'name': 'Artist 1',
            'member_count': 5,
            'is_verified': True,
        }),
        'get_cluster_members': MagicMock(return_value=[
            {
                'speaker_id': 'spk1',
                'track_id': 'track1',
                'track_title': 'Test Song',
                'artist_name': 'conor_maynard',
                'duration_sec': 60.0,
                'is_primary': True,
                'confidence': 0.95,
            }
        ]),
        'update_cluster_name': MagicMock(),
        'merge_clusters': MagicMock(),
        'create_cluster': MagicMock(return_value='new_cluster_id'),
        'get_embeddings_by_cluster': MagicMock(return_value=[]),
        'get_featured_artists_for_track': MagicMock(return_value=[
            {'name': 'Featured Artist 1'},
            {'name': 'Featured Artist 2'},
        ]),
        'get_all_featured_artists': MagicMock(return_value=[
            {'name': 'Artist 1', 'track_count': 5},
            {'name': 'Artist 2', 'track_count': 3},
        ]),
        'remove_from_cluster': MagicMock(),
    }
    return mock_ops


# =============================================================================
# Extraction Endpoints
# =============================================================================

class TestRunExtraction:
    """Test POST /api/v1/speakers/extraction/run endpoint."""

    def test_run_extraction_missing_artist_name_returns_400(self, client):
        """Returns 400 when artist_name not provided."""
        response = client.post(
            '/api/v1/speakers/extraction/run',
            json={},
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'artist_name' in data['error']

    @patch('auto_voice.audio.speaker_matcher.SpeakerMatcher')
    def test_run_extraction_success_returns_job_id(self, mock_matcher_class, client):
        """Returns job_id when extraction starts successfully."""
        mock_matcher = MagicMock()
        mock_matcher.extract_embeddings_for_artist.return_value = {
            'tracks_processed': 10,
            'embeddings_created': 25,
        }
        mock_matcher.cluster_speakers.return_value = [
            {'cluster_id': 'c1', 'member_count': 5},
        ]
        mock_matcher.auto_match_clusters_to_artists.return_value = {
            'matches_found': 3,
        }
        mock_matcher_class.return_value = mock_matcher

        response = client.post(
            '/api/v1/speakers/extraction/run',
            json={'artist_name': 'conor_maynard'},
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'job_id' in data
        assert 'status' in data
        assert data['status'] == 'complete'

    @patch('auto_voice.audio.speaker_matcher.SpeakerMatcher')
    def test_run_extraction_without_clustering(self, mock_matcher_class, client):
        """Runs extraction without clustering when run_clustering=False."""
        mock_matcher = MagicMock()
        mock_matcher.extract_embeddings_for_artist.return_value = {
            'tracks_processed': 5,
        }
        mock_matcher_class.return_value = mock_matcher

        response = client.post(
            '/api/v1/speakers/extraction/run',
            json={
                'artist_name': 'conor_maynard',
                'run_clustering': False,
            },
        )

        assert response.status_code == 200
        mock_matcher.cluster_speakers.assert_not_called()

    @patch('auto_voice.audio.speaker_matcher.SpeakerMatcher')
    def test_run_extraction_handles_extraction_error(self, mock_matcher_class, client):
        """Returns error status when extraction fails."""
        mock_matcher = MagicMock()
        mock_matcher.extract_embeddings_for_artist.side_effect = ValueError("Invalid artist")
        mock_matcher_class.return_value = mock_matcher

        response = client.post(
            '/api/v1/speakers/extraction/run',
            json={'artist_name': 'invalid_artist'},
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'failed'


class TestGetExtractionStatus:
    """Test GET /api/v1/speakers/extraction/status/<job_id> endpoint."""

    def test_get_status_job_not_found_returns_404(self, client):
        """Returns 404 when job_id doesn't exist."""
        response = client.get('/api/v1/speakers/extraction/status/nonexistent')

        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
        assert 'not found' in data['error'].lower()

    @patch('auto_voice.audio.speaker_matcher.SpeakerMatcher')
    def test_get_status_returns_job_progress(self, mock_matcher_class, client):
        """Returns job status and progress."""
        mock_matcher = MagicMock()
        mock_matcher.extract_embeddings_for_artist.return_value = {'tracks': 5}
        mock_matcher.cluster_speakers.return_value = []
        mock_matcher.auto_match_clusters_to_artists.return_value = {}
        mock_matcher_class.return_value = mock_matcher

        # Create a job first
        create_response = client.post(
            '/api/v1/speakers/extraction/run',
            json={'artist_name': 'test_artist'},
        )
        job_id = json.loads(create_response.data)['job_id']

        # Get status
        response = client.get(f'/api/v1/speakers/extraction/status/{job_id}')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'job_id' in data
        assert 'status' in data
        assert 'progress' in data
        assert 'message' in data


# =============================================================================
# Track Endpoints
# =============================================================================

class TestListTracks:
    """Test GET /api/v1/speakers/tracks endpoint."""

    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_list_all_tracks(self, mock_get_db, client, mock_db_ops):
        """Lists all tracks with featured artists."""
        mock_get_db.return_value = mock_db_ops

        response = client.get('/api/v1/speakers/tracks')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'tracks' in data
        assert 'count' in data
        assert data['count'] == 2
        assert all('featured_artists' in t for t in data['tracks'])

    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_list_tracks_filter_by_artist(self, mock_get_db, client, mock_db_ops):
        """Filters tracks by artist name."""
        mock_get_db.return_value = mock_db_ops

        response = client.get('/api/v1/speakers/tracks?artist=conor_maynard')

        assert response.status_code == 200
        mock_db_ops['get_tracks_by_artist'].assert_called_once_with('conor_maynard')

    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_list_tracks_filter_has_featured(self, mock_get_db, client, mock_db_ops):
        """Filters tracks with featured artists."""
        mock_get_db.return_value = mock_db_ops

        response = client.get('/api/v1/speakers/tracks?has_featured=true')

        assert response.status_code == 200
        data = json.loads(response.data)
        # All returned tracks should have featured artists
        assert all(t['featured_artists'] for t in data['tracks'])


class TestGetTrackDetails:
    """Test GET /api/v1/speakers/tracks/<track_id> endpoint."""

    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_get_track_details_success(self, mock_get_db, client, mock_db_ops):
        """Returns track details with featured artists."""
        mock_get_db.return_value = mock_db_ops

        response = client.get('/api/v1/speakers/tracks/track1')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['id'] == 'track1'
        assert 'featured_artists' in data
        assert len(data['featured_artists']) == 2

    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_get_track_details_not_found_returns_404(self, mock_get_db, client, mock_db_ops):
        """Returns 404 when track doesn't exist."""
        mock_db_ops['get_track'].return_value = None
        mock_get_db.return_value = mock_db_ops

        response = client.get('/api/v1/speakers/tracks/nonexistent')

        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data


class TestFetchMetadata:
    """Test POST /api/v1/speakers/tracks/fetch-metadata endpoint."""

    @patch('auto_voice.audio.youtube_metadata.populate_database_from_files')
    def test_fetch_metadata_for_all_artists(self, mock_populate, client):
        """Fetches metadata for all artists."""
        mock_populate.return_value = {
            'tracks_processed': 10,
            'featured_found': 5,
        }

        with patch('pathlib.Path.exists', return_value=True):
            response = client.post('/api/v1/speakers/tracks/fetch-metadata', json={})

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'stats' in data

    @patch('auto_voice.audio.youtube_metadata.populate_database_from_files')
    def test_fetch_metadata_for_specific_artist(self, mock_populate, client):
        """Fetches metadata for specific artist."""
        mock_populate.return_value = {'tracks_processed': 5}

        response = client.post(
            '/api/v1/speakers/tracks/fetch-metadata',
            json={'artist_name': 'conor_maynard'},
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True

    @patch('auto_voice.audio.youtube_metadata.populate_database_from_files')
    def test_fetch_metadata_handles_errors(self, mock_populate, client):
        """Returns 500 on fetch errors."""
        mock_populate.side_effect = Exception("Network error")

        response = client.post(
            '/api/v1/speakers/tracks/fetch-metadata',
            json={'artist_name': 'conor_maynard'},
        )

        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data


# =============================================================================
# Cluster Endpoints
# =============================================================================

class TestListClusters:
    """Test GET /api/v1/speakers/clusters endpoint."""

    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_list_all_clusters(self, mock_get_db, client, mock_db_ops):
        """Lists all speaker clusters."""
        mock_get_db.return_value = mock_db_ops

        response = client.get('/api/v1/speakers/clusters')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'clusters' in data
        assert 'count' in data
        assert data['count'] == 2


class TestGetClusterDetails:
    """Test GET /api/v1/speakers/clusters/<cluster_id> endpoint."""

    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_get_cluster_details_success(self, mock_get_db, client, mock_db_ops):
        """Returns cluster details with members grouped by track."""
        mock_get_db.return_value = mock_db_ops

        response = client.get('/api/v1/speakers/clusters/cluster1')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'cluster' in data
        assert 'members' in data
        assert 'tracks' in data
        assert 'track_count' in data

    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_get_cluster_details_not_found_returns_404(self, mock_get_db, client, mock_db_ops):
        """Returns 404 when cluster doesn't exist."""
        mock_db_ops['get_cluster'].return_value = None
        mock_get_db.return_value = mock_db_ops

        response = client.get('/api/v1/speakers/clusters/nonexistent')

        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data


class TestUpdateClusterName:
    """Test PUT /api/v1/speakers/clusters/<cluster_id>/name endpoint."""

    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_update_cluster_name_success(self, mock_get_db, client, mock_db_ops):
        """Updates cluster name successfully."""
        mock_get_db.return_value = mock_db_ops

        response = client.put(
            '/api/v1/speakers/clusters/cluster1/name',
            json={'name': 'Updated Name'},
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'cluster' in data
        mock_db_ops['update_cluster_name'].assert_called_once_with(
            'cluster1', 'Updated Name', True
        )

    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_update_cluster_name_with_verification_flag(self, mock_get_db, client, mock_db_ops):
        """Accepts is_verified flag."""
        mock_get_db.return_value = mock_db_ops

        response = client.put(
            '/api/v1/speakers/clusters/cluster1/name',
            json={'name': 'New Name', 'is_verified': False},
        )

        assert response.status_code == 200
        mock_db_ops['update_cluster_name'].assert_called_once_with(
            'cluster1', 'New Name', False
        )

    def test_update_cluster_name_missing_name_returns_400(self, client):
        """Returns 400 when name not provided."""
        response = client.put(
            '/api/v1/speakers/clusters/cluster1/name',
            json={},
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'name' in data['error']

    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_update_cluster_name_handles_errors(self, mock_get_db, client, mock_db_ops):
        """Returns 500 on database errors."""
        mock_db_ops['update_cluster_name'].side_effect = Exception("DB error")
        mock_get_db.return_value = mock_db_ops

        response = client.put(
            '/api/v1/speakers/clusters/cluster1/name',
            json={'name': 'Test'},
        )

        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data


class TestMergeClusters:
    """Test POST /api/v1/speakers/clusters/merge endpoint."""

    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_merge_clusters_success(self, mock_get_db, client, mock_db_ops):
        """Merges two clusters successfully."""
        mock_get_db.return_value = mock_db_ops

        response = client.post(
            '/api/v1/speakers/clusters/merge',
            json={'target_id': 'cluster1', 'source_id': 'cluster2'},
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'cluster' in data
        assert 'member_count' in data
        mock_db_ops['merge_clusters'].assert_called_once_with('cluster1', 'cluster2')

    def test_merge_clusters_missing_params_returns_400(self, client):
        """Returns 400 when required params missing."""
        response = client.post(
            '/api/v1/speakers/clusters/merge',
            json={'target_id': 'cluster1'},
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_merge_clusters_same_cluster_returns_400(self, client):
        """Returns 400 when trying to merge cluster with itself."""
        response = client.post(
            '/api/v1/speakers/clusters/merge',
            json={'target_id': 'cluster1', 'source_id': 'cluster1'},
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'itself' in data['error']

    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_merge_clusters_handles_errors(self, mock_get_db, client, mock_db_ops):
        """Returns 500 on merge errors."""
        mock_db_ops['merge_clusters'].side_effect = Exception("Merge failed")
        mock_get_db.return_value = mock_db_ops

        response = client.post(
            '/api/v1/speakers/clusters/merge',
            json={'target_id': 'c1', 'source_id': 'c2'},
        )

        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data


class TestSplitCluster:
    """Test POST /api/v1/speakers/clusters/split endpoint."""

    @patch('auto_voice.db.operations.remove_from_cluster')
    @patch('auto_voice.db.operations.add_to_cluster')
    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_split_cluster_success(self, mock_get_db, mock_add, mock_remove, client, mock_db_ops):
        """Splits cluster by moving embeddings to new cluster."""
        mock_get_db.return_value = mock_db_ops

        response = client.post(
            '/api/v1/speakers/clusters/split',
            json={
                'cluster_id': 'cluster1',
                'embedding_ids': ['emb1', 'emb2'],
                'new_name': 'Split Cluster',
            },
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'original_cluster' in data
        assert 'new_cluster' in data
        mock_db_ops['create_cluster'].assert_called_once_with('Split Cluster', is_verified=False)

    def test_split_cluster_missing_cluster_id_returns_400(self, client):
        """Returns 400 when cluster_id missing."""
        response = client.post(
            '/api/v1/speakers/clusters/split',
            json={'embedding_ids': ['emb1']},
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_split_cluster_missing_embeddings_returns_400(self, client):
        """Returns 400 when embedding_ids missing."""
        response = client.post(
            '/api/v1/speakers/clusters/split',
            json={'cluster_id': 'cluster1'},
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_split_cluster_handles_errors(self, mock_get_db, client, mock_db_ops):
        """Returns 500 on split errors."""
        mock_db_ops['create_cluster'].side_effect = Exception("Split failed")
        mock_get_db.return_value = mock_db_ops

        response = client.post(
            '/api/v1/speakers/clusters/split',
            json={
                'cluster_id': 'c1',
                'embedding_ids': ['e1'],
            },
        )

        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data


class TestAddClusterMembers:
    """Test POST /api/v1/speakers/clusters/<cluster_id>/members endpoint."""

    @patch('auto_voice.db.operations.add_to_cluster')
    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_add_single_member_success(self, mock_get_db, mock_add, client, mock_db_ops):
        """Adds single member to cluster successfully."""
        mock_get_db.return_value = mock_db_ops

        response = client.post(
            '/api/v1/speakers/clusters/cluster1/members',
            json={'embedding_id': 123},
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'cluster' in data
        assert 'member_count' in data
        assert data['added_count'] == 1
        mock_add.assert_called_once_with('cluster1', 123, confidence=None)

    @patch('auto_voice.db.operations.add_to_cluster')
    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_add_multiple_members_success(self, mock_get_db, mock_add, client, mock_db_ops):
        """Adds multiple members to cluster successfully."""
        mock_get_db.return_value = mock_db_ops

        response = client.post(
            '/api/v1/speakers/clusters/cluster1/members',
            json={'embedding_ids': [123, 456, 789]},
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['added_count'] == 3
        assert mock_add.call_count == 3

    def test_add_members_missing_embedding_ids_returns_400(self, client):
        """Returns 400 when no embedding IDs provided."""
        response = client.post(
            '/api/v1/speakers/clusters/cluster1/members',
            json={},
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data


class TestRemoveClusterMember:
    """Test DELETE /api/v1/speakers/clusters/<cluster_id>/members/<embedding_id> endpoint."""

    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_remove_member_success(self, mock_get_db, client, mock_db_ops):
        """Removes member from cluster successfully."""
        mock_get_db.return_value = mock_db_ops

        response = client.delete('/api/v1/speakers/clusters/cluster1/members/123')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'cluster' in data
        assert 'member_count' in data
        mock_db_ops['remove_from_cluster'].assert_called_once_with('cluster1', 123)

    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_remove_member_cluster_not_found_returns_404(self, mock_get_db, client, mock_db_ops):
        """Returns 404 when cluster doesn't exist."""
        mock_db_ops['get_cluster'].return_value = None
        mock_get_db.return_value = mock_db_ops

        response = client.delete('/api/v1/speakers/clusters/nonexistent/members/123')

        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Cluster not found' in data['error']

    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_remove_member_invalid_embedding_id_returns_400(self, mock_get_db, client, mock_db_ops):
        """Returns 400 when embedding_id is not a valid integer."""
        mock_get_db.return_value = mock_db_ops

        response = client.delete('/api/v1/speakers/clusters/cluster1/members/invalid')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Invalid embedding_id format' in data['error']

    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_remove_member_handles_errors(self, mock_get_db, client, mock_db_ops):
        """Returns 500 on removal errors."""
        mock_db_ops['remove_from_cluster'].side_effect = Exception("Removal failed")
        mock_get_db.return_value = mock_db_ops

        response = client.delete('/api/v1/speakers/clusters/cluster1/members/123')

        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data


class TestGetClusterSample:
    """Test GET /api/v1/speakers/clusters/<cluster_id>/sample endpoint."""

    @patch('auto_voice.web.speaker_api.send_file')
    @patch('auto_voice.audio.speaker_matcher.SpeakerMatcher')
    def test_get_cluster_sample_success(self, mock_matcher_class, mock_send_file, client):
        """Returns audio sample for cluster."""
        mock_matcher = MagicMock()
        mock_matcher.get_cluster_sample_audio.return_value = (
            np.zeros(22050, dtype=np.float32),
            22050,
        )
        mock_matcher_class.return_value = mock_matcher
        mock_send_file.return_value = 'audio_response'

        response = client.get('/api/v1/speakers/clusters/cluster1/sample')

        # send_file was called
        assert mock_send_file.called
        mock_matcher.get_cluster_sample_audio.assert_called_once_with('cluster1', 10.0)

    @patch('auto_voice.audio.speaker_matcher.SpeakerMatcher')
    def test_get_cluster_sample_with_max_duration(self, mock_matcher_class, client):
        """Accepts max_duration parameter."""
        mock_matcher = MagicMock()
        mock_matcher.get_cluster_sample_audio.return_value = (
            np.zeros(22050, dtype=np.float32),
            22050,
        )
        mock_matcher_class.return_value = mock_matcher

        with patch('auto_voice.web.speaker_api.send_file'):
            response = client.get('/api/v1/speakers/clusters/cluster1/sample?max_duration=5.0')

        mock_matcher.get_cluster_sample_audio.assert_called_once_with('cluster1', 5.0)

    @patch('auto_voice.audio.speaker_matcher.SpeakerMatcher')
    def test_get_cluster_sample_not_found_returns_404(self, mock_matcher_class, client):
        """Returns 404 when cluster not found."""
        mock_matcher = MagicMock()
        mock_matcher.get_cluster_sample_audio.side_effect = ValueError("Cluster not found")
        mock_matcher_class.return_value = mock_matcher

        response = client.get('/api/v1/speakers/clusters/nonexistent/sample')

        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data

    @patch('auto_voice.audio.speaker_matcher.SpeakerMatcher')
    def test_get_cluster_sample_handles_errors(self, mock_matcher_class, client):
        """Returns 500 on sample generation errors."""
        mock_matcher = MagicMock()
        mock_matcher.get_cluster_sample_audio.side_effect = Exception("Audio error")
        mock_matcher_class.return_value = mock_matcher

        response = client.get('/api/v1/speakers/clusters/cluster1/sample')

        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data


# =============================================================================
# Speaker Identification Endpoints
# =============================================================================

class TestRunSpeakerIdentification:
    """Test POST /api/v1/speakers/identify endpoint."""

    @patch('auto_voice.audio.speaker_matcher.SpeakerMatcher')
    def test_identify_default_artists(self, mock_matcher_class, client):
        """Runs identification for default artists."""
        mock_matcher = MagicMock()
        mock_matcher.extract_embeddings_for_artist.return_value = {'tracks': 5}
        mock_matcher.cluster_speakers.return_value = [
            {'cluster_id': 'c1', 'member_count': 10, 'total_duration_sec': 300.0}
        ]
        mock_matcher.auto_match_clusters_to_artists.return_value = {'matches': 3}
        mock_matcher_class.return_value = mock_matcher

        response = client.post('/api/v1/speakers/identify', json={})

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'stats' in data
        assert 'artists' in data['stats']
        assert 'clustering' in data['stats']
        assert 'matching' in data['stats']

    @patch('auto_voice.audio.speaker_matcher.SpeakerMatcher')
    def test_identify_specific_artists(self, mock_matcher_class, client):
        """Runs identification for specific artists."""
        mock_matcher = MagicMock()
        mock_matcher.extract_embeddings_for_artist.return_value = {'tracks': 5}
        mock_matcher.cluster_speakers.return_value = []
        mock_matcher.auto_match_clusters_to_artists.return_value = {}
        mock_matcher_class.return_value = mock_matcher

        response = client.post(
            '/api/v1/speakers/identify',
            json={'artists': ['conor_maynard']},
        )

        assert response.status_code == 200
        mock_matcher.extract_embeddings_for_artist.assert_called_once_with('conor_maynard')

    @patch('auto_voice.audio.speaker_matcher.SpeakerMatcher')
    def test_identify_with_custom_thresholds(self, mock_matcher_class, client):
        """Accepts custom threshold and min_duration."""
        mock_matcher = MagicMock()
        mock_matcher.extract_embeddings_for_artist.return_value = {}
        mock_matcher.cluster_speakers.return_value = []
        mock_matcher.auto_match_clusters_to_artists.return_value = {}
        mock_matcher_class.return_value = mock_matcher

        response = client.post(
            '/api/v1/speakers/identify',
            json={
                'threshold': 0.90,
                'min_duration': 60.0,
            },
        )

        assert response.status_code == 200
        mock_matcher_class.assert_called_once_with(
            similarity_threshold=0.90,
            min_cluster_duration=60.0,
        )

    @patch('auto_voice.audio.speaker_matcher.SpeakerMatcher')
    def test_identify_handles_errors(self, mock_matcher_class, client):
        """Returns 500 on identification errors."""
        mock_matcher = MagicMock()
        mock_matcher.extract_embeddings_for_artist.side_effect = Exception("Extraction failed")
        mock_matcher_class.return_value = mock_matcher

        response = client.post('/api/v1/speakers/identify', json={})

        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data


class TestListFeaturedArtists:
    """Test GET /api/v1/speakers/featured-artists endpoint."""

    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_list_featured_artists(self, mock_get_db, client, mock_db_ops):
        """Lists all featured artists with track counts."""
        mock_get_db.return_value = mock_db_ops

        response = client.get('/api/v1/speakers/featured-artists')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'artists' in data
        assert 'count' in data
        assert data['count'] == 2


# =============================================================================
# File Organization Endpoints
# =============================================================================

class TestOrganizeFiles:
    """Test POST /api/v1/speakers/organize endpoint."""

    @patch('auto_voice.audio.file_organizer.organize_by_identified_artist')
    def test_organize_dry_run_default(self, mock_organize, client):
        """Dry run is default mode."""
        mock_organize.return_value = {
            'files_to_move': 10,
            'files_moved': 0,
        }

        response = client.post('/api/v1/speakers/organize', json={})

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['dry_run'] is True
        mock_organize.assert_called_once_with(dry_run=True)

    @patch('auto_voice.audio.file_organizer.organize_by_identified_artist')
    def test_organize_execute_mode(self, mock_organize, client):
        """Executes organization when dry_run=False."""
        mock_organize.return_value = {
            'files_moved': 10,
        }

        response = client.post(
            '/api/v1/speakers/organize',
            json={'dry_run': False},
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['dry_run'] is False
        mock_organize.assert_called_once_with(dry_run=False)

    @patch('auto_voice.audio.file_organizer.organize_by_identified_artist')
    def test_organize_handles_errors(self, mock_organize, client):
        """Returns 500 on organization errors."""
        mock_organize.side_effect = Exception("Organization failed")

        response = client.post('/api/v1/speakers/organize', json={})

        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data


# =============================================================================
# Edge Cases and Integration
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_json_body_handled_gracefully(self, client):
        """Handles None JSON body gracefully."""
        response = client.post(
            '/api/v1/speakers/extraction/run',
            data='',  # Empty body
            content_type='application/json',
        )

        assert response.status_code == 400

    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_tracks_with_no_featured_artists(self, mock_get_db, client, mock_db_ops):
        """Handles tracks with empty featured artists list."""
        mock_db_ops['get_featured_artists_for_track'].return_value = []
        mock_get_db.return_value = mock_db_ops

        response = client.get('/api/v1/speakers/tracks')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert all('featured_artists' in t for t in data['tracks'])

    @patch('auto_voice.web.speaker_api._get_db_operations')
    def test_cluster_details_groups_speakers_by_track(self, mock_get_db, client, mock_db_ops):
        """Groups cluster members by track correctly."""
        mock_db_ops['get_cluster_members'].return_value = [
            {
                'speaker_id': 'spk1',
                'track_id': 'track1',
                'track_title': 'Song 1',
                'artist_name': 'artist1',
                'duration_sec': 30.0,
                'is_primary': True,
                'confidence': 0.95,
            },
            {
                'speaker_id': 'spk2',
                'track_id': 'track1',
                'track_title': 'Song 1',
                'artist_name': 'artist1',
                'duration_sec': 20.0,
                'is_primary': False,
                'confidence': 0.85,
            },
        ]
        mock_get_db.return_value = mock_db_ops

        response = client.get('/api/v1/speakers/clusters/cluster1')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['track_count'] == 1
        assert len(data['tracks']) == 1
        assert len(data['tracks'][0]['speakers']) == 2
