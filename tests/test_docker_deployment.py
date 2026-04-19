"""Integration tests for Docker deployment on Jetson-class environments.

Task 1.5: Test container startup and GPU access
- Verify the Docker image builds successfully
- Verify the container starts and responds to health checks
- Verify GPU is accessible inside the container
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import time

import pytest
import requests


pytestmark = pytest.mark.integration


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_NAME = "autovoice:test"
CONTAINER_NAME = "autovoice-test-container"
SHUTDOWN_TEST_CONTAINER = "autovoice-shutdown-test"
HEALTH_URL = "http://localhost:5000/health"
READY_URL = "http://localhost:5000/ready"
MAX_STARTUP_WAIT = 120  # seconds
HEALTH_CHECK_INTERVAL = 5  # seconds
BUILD_TIMEOUT = 1800  # seconds
SHARED_SECRET = "test-secret-key-for-docker-deployment"

CRITICAL_LOG_PATTERNS = [
    "Traceback (most recent call last)",
    "ModuleNotFoundError",
    "ImportError: cannot import",
    "Fatal error",
    "CUDA error: out of memory",
]


def _run(cmd: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _cleanup_container(name: str) -> None:
    _run(["docker", "rm", "-f", name], timeout=120)


def _skip_for_unsupported_environment() -> None:
    if platform.system() != "Linux":
        pytest.skip("Docker deployment tests require Linux")
    if platform.machine() != "aarch64":
        pytest.skip("Docker deployment tests target Jetson-class aarch64 hosts")

    info = _run(["docker", "info", "--format", "{{json .}}"], timeout=30)
    if info.returncode != 0:
        pytest.skip(f"Docker daemon unavailable: {info.stderr.strip()}")

    try:
        docker_info = json.loads(info.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive path
        pytest.skip(f"Unable to parse docker info output: {exc}")

    runtimes = docker_info.get("Runtimes", {})
    if "nvidia" not in runtimes:
        pytest.skip("Docker is available but NVIDIA runtime is not configured")


def _is_external_registry_failure(stderr: str) -> bool:
    text = stderr.lower()
    external_failure_markers = (
        "pull access denied",
        "unauthorized",
        "authentication required",
        "denied:",
        "name unknown",
        "manifest unknown",
        "temporary failure",
        "tls handshake timeout",
        "connection reset by peer",
        "i/o timeout",
    )
    return any(marker in text for marker in external_failure_markers)


def _wait_for_health(url: str, *, expected_status: int = 200) -> requests.Response:
    deadline = time.time() + MAX_STARTUP_WAIT
    last_error: Exception | None = None

    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == expected_status:
                return response
            last_error = RuntimeError(
                f"{url} returned {response.status_code}, expected {expected_status}"
            )
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(HEALTH_CHECK_INTERVAL)

    pytest.fail(f"{url} did not become ready within {MAX_STARTUP_WAIT}s: {last_error}")


@pytest.fixture(scope="session")
def docker_preflight():
    """Skip the suite when Docker/NVIDIA runtime prerequisites are absent."""
    _skip_for_unsupported_environment()
    return True


@pytest.fixture(scope="session")
def docker_image(docker_preflight):
    """Build the Docker image once for the module."""
    logger.info("Building Docker image %s", IMAGE_NAME)
    result = _run(["docker", "build", "-t", IMAGE_NAME, "."], timeout=BUILD_TIMEOUT)
    if result.returncode != 0:
        if _is_external_registry_failure(result.stderr):
            pytest.skip(f"Docker build depends on external registry access: {result.stderr}")
        pytest.fail(f"Docker build failed: {result.stderr}")

    yield IMAGE_NAME

    _cleanup_container(CONTAINER_NAME)
    _cleanup_container(SHUTDOWN_TEST_CONTAINER)
    _run(["docker", "rmi", "-f", IMAGE_NAME], timeout=180)


@pytest.fixture(scope="session")
def running_container(docker_image):
    """Start a container from the built image for health/GPU checks."""
    _cleanup_container(CONTAINER_NAME)

    logger.info("Starting container %s", CONTAINER_NAME)
    result = _run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            "--runtime",
            "nvidia",
            "-e",
            f"AUTOVOICE_SECRET_FLASK_SECRET_KEY={SHARED_SECRET}",
            "-e",
            "NVIDIA_VISIBLE_DEVICES=all",
            "-e",
            "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
            "-e",
            "CUDA_VISIBLE_DEVICES=0",
            "-p",
            "5000:5000",
            "--memory",
            "8g",
            docker_image,
        ],
        timeout=120,
    )

    if result.returncode != 0:
        pytest.fail(f"Failed to start container: {result.stderr}")

    container_id = result.stdout.strip()
    _wait_for_health(HEALTH_URL, expected_status=200)

    yield container_id

    _run(["docker", "stop", "-t", "30", CONTAINER_NAME], timeout=90)
    _cleanup_container(CONTAINER_NAME)


class TestDockerBuild:
    """Test Docker image builds successfully."""

    def test_docker_build_succeeds(self, docker_image):
        result = _run(["docker", "images", "-q", docker_image], timeout=30)
        assert result.returncode == 0, "Failed to query Docker images"
        assert result.stdout.strip(), f"Image {docker_image} not found"

    def test_docker_image_has_layers(self, docker_image):
        result = _run(["docker", "history", "--quiet", docker_image], timeout=30)
        assert result.returncode == 0, "Failed to get image history"
        layers = [layer for layer in result.stdout.splitlines() if layer.strip()]
        assert len(layers) > 1, "Image has insufficient layers"


class TestContainerStartup:
    """Test container starts and health checks pass."""

    def test_container_is_running(self, running_container):
        result = _run(
            ["docker", "inspect", "-f", "{{.State.Status}}", CONTAINER_NAME],
            timeout=30,
        )
        assert result.returncode == 0, "Failed to inspect container"
        assert result.stdout.strip() == "running"

    def test_container_logs_no_errors(self, running_container):
        time.sleep(10)
        result = _run(["docker", "logs", CONTAINER_NAME], timeout=30)
        assert result.returncode == 0, "Failed to get container logs"
        logs = result.stdout + result.stderr
        for pattern in CRITICAL_LOG_PATTERNS:
            assert pattern not in logs, f"Critical error found in logs: {pattern}"

    def test_health_endpoint_responds(self, running_container):
        response = _wait_for_health(HEALTH_URL, expected_status=200)
        assert response.status_code == 200

    def test_ready_endpoint_responds(self, running_container):
        response = requests.get(READY_URL, timeout=10)
        assert response.status_code in {200, 503}
        data = response.json()
        assert "ready" in data
        assert "components" in data

    def test_health_response_has_required_fields(self, running_container):
        response = requests.get(HEALTH_URL, timeout=10)
        assert response.status_code == 200
        data = response.json()

        required_fields = ["status", "version", "timestamp"]
        for field in required_fields:
            assert field in data, f"Health response missing field: {field}"

        assert data["status"] in {"healthy", "degraded", "unhealthy"}


class TestGPUAccess:
    """Test GPU is accessible inside container."""

    def test_nvidia_smi_runs_in_container(self, running_container):
        result = _run(["docker", "exec", CONTAINER_NAME, "nvidia-smi"], timeout=30)
        assert result.returncode == 0, f"nvidia-smi failed: {result.stderr}"
        assert "NVIDIA-SMI" in result.stdout

    def test_pytorch_cuda_available_in_container(self, running_container):
        python_cmd = (
            "import torch; "
            "print(f'CUDA available: {torch.cuda.is_available()}'); "
            "print(f'CUDA devices: {torch.cuda.device_count()}')"
        )
        result = _run(
            ["docker", "exec", CONTAINER_NAME, "python", "-c", python_cmd],
            timeout=30,
        )

        assert result.returncode == 0, f"PyTorch CUDA check failed: {result.stderr}"
        output = result.stdout.strip()
        assert "CUDA available: True" in output
        assert "CUDA devices:" in output

    def test_cuda_version_matches_host_expectations(self, running_container):
        result = _run(
            ["docker", "exec", CONTAINER_NAME, "python", "-c", "import torch; print(torch.version.cuda)"],
            timeout=30,
        )

        assert result.returncode == 0, "Failed to get CUDA version from container"
        cuda_version = result.stdout.strip()
        assert cuda_version and cuda_version != "None"

        major_version = int(cuda_version.split(".", 1)[0])
        assert major_version >= 13, f"Expected CUDA 13+, got {cuda_version}"


class TestResourceLimits:
    """Test container resource limits are applied."""

    def test_memory_limit_enforced(self, running_container):
        result = _run(
            ["docker", "inspect", "-f", "{{.HostConfig.Memory}}", CONTAINER_NAME],
            timeout=30,
        )
        assert result.returncode == 0, "Failed to inspect container memory"
        assert result.stdout.strip() != "0", "Memory limit not set"

    def test_gpu_count_allocated(self, running_container):
        device_result = _run(
            ["docker", "inspect", "-f", "{{.HostConfig.DeviceRequests}}", CONTAINER_NAME],
            timeout=30,
        )
        runtime_result = _run(
            ["docker", "inspect", "-f", "{{.HostConfig.Runtime}}", CONTAINER_NAME],
            timeout=30,
        )
        assert device_result.returncode == 0, "Failed to inspect GPU allocation"
        assert runtime_result.returncode == 0, "Failed to inspect container runtime"
        device_requests = device_result.stdout.strip().lower()
        runtime = runtime_result.stdout.strip().lower()
        assert (
            "nvidia" in device_requests
            or "gpu" in device_requests
            or runtime == "nvidia"
        ), f"Expected GPU runtime/device request, got runtime={runtime!r} device_requests={device_requests!r}"


class TestGracefulShutdown:
    """Test container graceful shutdown behavior."""

    def test_container_stops_gracefully(self, docker_image):
        _cleanup_container(SHUTDOWN_TEST_CONTAINER)

        run_result = _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                SHUTDOWN_TEST_CONTAINER,
                "--runtime",
                "nvidia",
                "-e",
                f"AUTOVOICE_SECRET_FLASK_SECRET_KEY={SHARED_SECRET}",
                "-p",
                "5001:5000",
                docker_image,
            ],
            timeout=120,
        )
        assert run_result.returncode == 0, run_result.stderr

        time.sleep(15)

        start_time = time.time()
        stop_result = _run(
            ["docker", "stop", "-t", "10", SHUTDOWN_TEST_CONTAINER],
            timeout=30,
        )
        stop_duration = time.time() - start_time

        _cleanup_container(SHUTDOWN_TEST_CONTAINER)

        assert stop_result.returncode == 0, f"Container stop failed: {stop_result.stderr}"
        assert stop_duration < 15, f"Container took {stop_duration}s to stop, expected <15s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
