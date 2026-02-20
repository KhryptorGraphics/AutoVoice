"""Tests for Docker container startup and GPU access.

Task 1.5: Test container startup and GPU access
- Verify Docker image builds successfully
- Verify container starts and responds to health checks
- Verify GPU is accessible inside container
"""

import subprocess
import time
import pytest
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IMAGE_NAME = "autovoice:test"
CONTAINER_NAME = "autovoice-test-container"
HEALTH_URL = "http://localhost:5000/health"
READY_URL = "http://localhost:5000/ready"
MAX_STARTUP_WAIT = 120  # seconds
HEALTH_CHECK_INTERVAL = 5  # seconds


class TestDockerBuild:
    """Test Docker image builds successfully."""

    @pytest.fixture(scope="class")
    def docker_image(self):
        """Build Docker image for testing."""
        logger.info(f"Building Docker image: {IMAGE_NAME}")
        result = subprocess.run(
            ["docker", "build", "-t", IMAGE_NAME, "."],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes for build
        )

        if result.returncode != 0:
            logger.error(f"Docker build failed: {result.stderr}")
            pytest.fail(f"Docker build failed: {result.stderr}")

        logger.info("Docker image built successfully")
        yield IMAGE_NAME

        # Cleanup: Remove image after tests
        subprocess.run(
            ["docker", "rmi", "-f", IMAGE_NAME],
            capture_output=True
        )

    def test_docker_build_succeeds(self, docker_image):
        """Verify Docker image builds without errors."""
        # Verify image exists
        result = subprocess.run(
            ["docker", "images", "-q", docker_image],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "Failed to query Docker images"
        assert result.stdout.strip(), f"Image {docker_image} not found"

    def test_docker_image_has_layers(self, docker_image):
        """Verify image has expected layers (not empty)."""
        result = subprocess.run(
            ["docker", "history", "--quiet", docker_image],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "Failed to get image history"
        layers = result.stdout.strip().split("\n")
        assert len(layers) > 1, "Image has insufficient layers"


class TestContainerStartup:
    """Test container starts and health checks pass."""

    @pytest.fixture(scope="class")
    def running_container(self):
        """Start container for testing."""
        # Remove any existing test container
        subprocess.run(
            ["docker", "rm", "-f", CONTAINER_NAME],
            capture_output=True
        )

        logger.info(f"Starting container: {CONTAINER_NAME}")

        # Start container with GPU support
        result = subprocess.run(
            [
                "docker", "run", "-d",
                "--name", CONTAINER_NAME,
                "--runtime", "nvidia",
                "-e", "NVIDIA_VISIBLE_DEVICES=all",
                "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                "-e", "CUDA_VISIBLE_DEVICES=0",
                "-p", "5000:5000",
                "--memory", "8g",
                IMAGE_NAME
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            pytest.fail(f"Failed to start container: {result.stderr}")

        container_id = result.stdout.strip()
        logger.info(f"Container started: {container_id}")

        # Wait for container to be ready
        yield container_id

        # Cleanup: Stop and remove container
        logger.info(f"Stopping container: {CONTAINER_NAME}")
        subprocess.run(
            ["docker", "stop", "-t", "30", CONTAINER_NAME],
            capture_output=True
        )
        subprocess.run(
            ["docker", "rm", "-f", CONTAINER_NAME],
            capture_output=True
        )

    def test_container_is_running(self, running_container):
        """Verify container is in running state."""
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", CONTAINER_NAME],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "Failed to inspect container"
        status = result.stdout.strip()
        assert status == "running", f"Container status is {status}, expected 'running'"

    def test_container_logs_no_errors(self, running_container):
        """Verify container logs don't show critical errors on startup."""
        time.sleep(10)  # Wait for startup

        result = subprocess.run(
            ["docker", "logs", CONTAINER_NAME],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "Failed to get container logs"

        logs = result.stdout + result.stderr

        # Check for critical errors (but allow warnings and expected errors)
        critical_patterns = [
            "Traceback (most recent call last)",
            "ModuleNotFoundError",
            "ImportError: cannot import",
            "Fatal error",
            "CUDA error: out of memory"
        ]

        for pattern in critical_patterns:
            assert pattern not in logs, f"Critical error found in logs: {pattern}"

    def test_health_endpoint_responds(self, running_container):
        """Verify health endpoint returns 200."""
        # Wait for startup
        for attempt in range(MAX_STARTUP_WAIT // HEALTH_CHECK_INTERVAL):
            try:
                response = requests.get(HEALTH_URL, timeout=5)
                if response.status_code == 200:
                    logger.info(f"Health endpoint ready after {attempt * HEALTH_CHECK_INTERVAL}s")
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(HEALTH_CHECK_INTERVAL)
        else:
            pytest.fail(f"Health endpoint not responding after {MAX_STARTUP_WAIT}s")

        response = requests.get(HEALTH_URL, timeout=10)
        assert response.status_code == 200, f"Health endpoint returned {response.status_code}"

    def test_ready_endpoint_responds(self, running_container):
        """Verify ready endpoint returns 200."""
        response = requests.get(READY_URL, timeout=10)
        assert response.status_code == 200, f"Ready endpoint returned {response.status_code}"

    def test_health_response_has_required_fields(self, running_container):
        """Verify health response contains expected fields."""
        response = requests.get(HEALTH_URL, timeout=10)
        data = response.json()

        required_fields = ["status", "version", "timestamp"]
        for field in required_fields:
            assert field in data, f"Health response missing field: {field}"

        assert data["status"] == "healthy", f"Health status is {data['status']}"


class TestGPUAccess:
    """Test GPU is accessible inside container."""

    def test_nvidia_smi_runs_in_container(self, running_container):
        """Verify nvidia-smi runs successfully inside container."""
        result = subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0, f"nvidia-smi failed: {result.stderr}"
        assert "NVIDIA-SMI" in result.stdout, "nvidia-smi output doesn't contain expected header"

    def test_pytorch_cuda_available_in_container(self, running_container):
        """Verify PyTorch can access CUDA inside container."""
        python_cmd = "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

        result = subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "python", "-c", python_cmd],
            capture_output=True,
            text=True,
            timeout=30
        )

        assert result.returncode == 0, f"PyTorch CUDA check failed: {result.stderr}"
        output = result.stdout.strip()

        assert "CUDA available: True" in output, "PyTorch CUDA not available in container"
        assert "CUDA devices: 1" in output, "Expected 1 CUDA device in container"

    def test_cuda_version_matches_host(self, running_container):
        """Verify container CUDA version matches host expectations."""
        # Get container CUDA version
        result = subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "python", "-c",
             "import torch; print(torch.version.cuda)"],
            capture_output=True,
            text=True,
            timeout=30
        )

        assert result.returncode == 0, "Failed to get CUDA version from container"
        cuda_version = result.stdout.strip()

        # For Jetson Thor with CUDA 13.0
        assert cuda_version.startswith("13."), f"Expected CUDA 13.x, got {cuda_version}"


class TestResourceLimits:
    """Test container resource limits are applied."""

    def test_memory_limit_enforced(self, running_container):
        """Verify memory limit is set on container."""
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.HostConfig.Memory}}", CONTAINER_NAME],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "Failed to inspect container memory"
        memory_limit = result.stdout.strip()
        assert memory_limit != "0", "Memory limit not set"

    def test_gpu_count_allocated(self, running_container):
        """Verify GPU is allocated to container."""
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.HostConfig.DeviceRequests}}", CONTAINER_NAME],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "Failed to inspect GPU allocation"
        device_requests = result.stdout.strip()
        assert "nvidia" in device_requests.lower() or "gpu" in device_requests.lower(), \
            "GPU not allocated to container"


class TestGracefulShutdown:
    """Test container graceful shutdown behavior."""

    def test_container_stops_gracefully(self):
        """Verify container stops within timeout on SIGTERM."""
        # Start a fresh container
        subprocess.run(
            ["docker", "rm", "-f", "autovoice-shutdown-test"],
            capture_output=True
        )

        subprocess.run(
            [
                "docker", "run", "-d",
                "--name", "autovoice-shutdown-test",
                "--runtime", "nvidia",
                "-p", "5001:5000",
                IMAGE_NAME
            ],
            capture_output=True,
            check=True
        )

        # Wait for startup
        time.sleep(15)

        # Stop with timeout
        start_time = time.time()
        result = subprocess.run(
            ["docker", "stop", "-t", "10", "autovoice-shutdown-test"],
            capture_output=True,
            text=True
        )
        stop_duration = time.time() - start_time

        # Cleanup
        subprocess.run(
            ["docker", "rm", "-f", "autovoice-shutdown-test"],
            capture_output=True
        )

        assert result.returncode == 0, f"Container stop failed: {result.stderr}"
        assert stop_duration < 15, f"Container took {stop_duration}s to stop, expected <15s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
