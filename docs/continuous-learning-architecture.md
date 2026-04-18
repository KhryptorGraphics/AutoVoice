# Continuous Learning Architecture Design

> Historical design note: this document captures an earlier architecture track. It is useful for implementation context, but it is not the canonical current-product spec. Start with [../README.md](../README.md), [user-guide-voice-profiles.md](./user-guide-voice-profiles.md), and [frontend-persistence-boundaries.md](./frontend-persistence-boundaries.md) for current behavior.

**Track:** voice-profile-training_20260124
**Task:** 2.5
**Date:** 2026-01-24

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AutoVoice Continuous Learning System                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌───────────────┐    ┌───────────────────────────────┐│
│  │   Karaoke    │───▶│   Sample      │───▶│      Training Job Queue       ││
│  │   Sessions   │    │   Collector   │    │  (GPU-only, async execution)  ││
│  └──────────────┘    └───────────────┘    └───────────────────────────────┘│
│                             │                           │                   │
│                             ▼                           ▼                   │
│  ┌──────────────────────────────────────┐    ┌───────────────────────────┐ │
│  │         Voice Profile Database        │    │    Model Version Store    │ │
│  │  ├─ profile metadata                  │    │  ├─ base_model.pt         │ │
│  │  ├─ training_samples (audio + meta)   │    │  ├─ adapters/{profile_id}/│ │
│  │  └─ training_history                  │    │  │   ├─ v1.pt (LoRA)      │ │
│  └──────────────────────────────────────┘    │  │   ├─ v2.pt             │ │
│                             │                │  │   └─ latest -> v2.pt   │ │
│                             ▼                │  └─ ewc_fisher_matrices/  │ │
│  ┌──────────────────────────────────────┐    └───────────────────────────┘ │
│  │        Training Scheduler             │                │                │
│  │  ├─ monitors sample accumulation      │◀───────────────┘                │
│  │  ├─ triggers training jobs            │                                 │
│  │  └─ manages GPU resource allocation   │                                 │
│  └──────────────────────────────────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Specifications

### 1. Sample Collector

**Purpose:** Capture and validate training samples from karaoke sessions.

```python
class SampleCollector:
    """Collects and validates training samples."""

    quality_thresholds = {
        "min_snr_db": 20.0,          # Minimum signal-to-noise ratio
        "min_duration_sec": 2.0,      # Minimum sample length
        "max_duration_sec": 30.0,     # Maximum sample length
        "min_pitch_stability": 0.8,   # F0 detection confidence
        "min_sample_rate": 24000,     # Minimum sample rate
    }

    def validate_sample(self, audio_path: str) -> ValidationResult:
        """Validate sample meets quality thresholds."""
        pass

    def segment_phrases(self, audio_path: str) -> list[AudioSegment]:
        """Segment audio into clean phrases using silence detection."""
        pass

    def extract_metadata(self, audio_path: str) -> SampleMetadata:
        """Extract pitch range, technique markers, duration."""
        pass
```

**Integration Points:**
- WebSocket events from karaoke sessions
- Multipart file upload API
- Background processing queue

### 2. Training Job Manager

**Purpose:** Execute GPU training jobs asynchronously with proper resource management.

```python
class TrainingJobManager:
    """Manages training job queue and execution."""

    def __init__(self, config: TrainingConfig):
        self.job_queue: Queue[TrainingJob] = Queue()
        self.active_jobs: dict[str, TrainingJob] = {}
        self.max_concurrent_jobs = 1  # GPU memory constraint
        self.device = torch.device("cuda")

    def create_job(
        self,
        profile_id: str,
        job_type: Literal["initial", "incremental", "technique"],
        samples: list[TrainingSample],
    ) -> TrainingJob:
        """Create a new training job."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for training jobs")
        return TrainingJob(
            id=str(uuid4()),
            profile_id=profile_id,
            job_type=job_type,
            samples=samples,
            status="pending",
            created=datetime.now(timezone.utc),
        )

    def execute_job(self, job: TrainingJob) -> TrainingResult:
        """Execute training job on GPU."""
        # Load base model
        # Initialize/load LoRA adapter
        # Apply EWC regularization
        # Train with prior preservation
        # Save new adapter version
        pass
```

**Job Types:**
1. `initial` - First training for new profile (10+ samples required)
2. `incremental` - Add new samples to existing profile
3. `technique` - Focused training on specific techniques

### 3. Training Scheduler

**Purpose:** Automatically trigger training based on sample accumulation.

```python
@dataclass
class SchedulerConfig:
    """Configuration for training scheduler."""
    min_samples_for_initial: int = 10
    min_samples_for_incremental: int = 5
    min_duration_for_initial_sec: float = 60.0
    min_duration_for_incremental_sec: float = 30.0
    max_samples_before_forced: int = 50
    cooldown_hours: float = 24.0  # Min time between trainings

class TrainingScheduler:
    """Monitors profiles and triggers training jobs."""

    def __init__(self, config: SchedulerConfig, job_manager: TrainingJobManager):
        self.config = config
        self.job_manager = job_manager

    def check_profile(self, profile: VoiceProfile) -> ScheduleDecision:
        """Determine if profile needs training."""
        unprocessed = profile.get_unprocessed_samples()
        total_duration = sum(s.duration_seconds for s in unprocessed)

        if profile.model_version is None:
            # Initial training check
            if (len(unprocessed) >= self.config.min_samples_for_initial and
                total_duration >= self.config.min_duration_for_initial_sec):
                return ScheduleDecision(should_train=True, job_type="initial")
        else:
            # Incremental training check
            hours_since_last = self._hours_since_last_training(profile)
            if hours_since_last < self.config.cooldown_hours:
                return ScheduleDecision(should_train=False, reason="cooldown")

            if len(unprocessed) >= self.config.max_samples_before_forced:
                return ScheduleDecision(should_train=True, job_type="incremental")

            if (len(unprocessed) >= self.config.min_samples_for_incremental and
                total_duration >= self.config.min_duration_for_incremental_sec):
                return ScheduleDecision(should_train=True, job_type="incremental")

        return ScheduleDecision(should_train=False)

    def run_scheduler_loop(self):
        """Background loop to check all profiles periodically."""
        while True:
            for profile in self._get_active_profiles():
                decision = self.check_profile(profile)
                if decision.should_train:
                    self.job_manager.create_job(
                        profile_id=profile.id,
                        job_type=decision.job_type,
                        samples=profile.get_unprocessed_samples(),
                    )
            time.sleep(300)  # Check every 5 minutes
```

### 4. Model Version Manager

**Purpose:** Track adapter versions and enable rollback.

```python
@dataclass
class ModelVersion:
    """Represents a specific adapter version."""
    version: str
    profile_id: str
    created: datetime
    adapter_path: str
    metrics: dict[str, float]
    samples_trained: int
    is_active: bool = False

class ModelVersionManager:
    """Manages model versions with rollback capability."""

    def __init__(self, storage_path: Path, max_versions: int = 5):
        self.storage_path = storage_path
        self.max_versions = max_versions

    def save_version(
        self,
        profile_id: str,
        adapter_state: dict,
        metrics: dict[str, float],
        samples_trained: int,
    ) -> ModelVersion:
        """Save new adapter version."""
        profile_dir = self.storage_path / "adapters" / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)

        version_num = self._get_next_version(profile_id)
        version_str = f"v{version_num}"
        adapter_path = profile_dir / f"{version_str}.pt"

        torch.save(adapter_state, adapter_path)

        version = ModelVersion(
            version=version_str,
            profile_id=profile_id,
            created=datetime.now(timezone.utc),
            adapter_path=str(adapter_path),
            metrics=metrics,
            samples_trained=samples_trained,
            is_active=True,
        )

        self._update_symlink(profile_id, adapter_path)
        self._prune_old_versions(profile_id)
        return version

    def rollback(self, profile_id: str, target_version: str) -> ModelVersion:
        """Rollback to a previous version."""
        version = self._get_version(profile_id, target_version)
        if not version:
            raise ValueError(f"Version {target_version} not found")
        self._update_symlink(profile_id, Path(version.adapter_path))
        return version

    def compare_versions(
        self,
        profile_id: str,
        version_a: str,
        version_b: str,
        test_samples: list[str],
    ) -> ComparisonResult:
        """A/B compare two versions on test samples."""
        pass
```

### 5. LoRA Adapter Configuration

**Purpose:** Define adapter architecture for speaker-specific fine-tuning.

```python
@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    rank: int = 8                     # Low rank dimension
    alpha: int = 16                   # Scaling factor
    dropout: float = 0.1              # Dropout rate
    target_modules: list[str] = field(default_factory=lambda: [
        "content_encoder.layers",     # Content encoding layers
        "speaker_encoder.attention",  # Speaker attention
        "decoder.conv_layers",        # Decoder convolutions
    ])

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

class LoRAAdapter(nn.Module):
    """LoRA adapter for voice model fine-tuning."""

    def __init__(self, original_module: nn.Module, config: LoRAConfig):
        super().__init__()
        self.original = original_module
        self.config = config

        # Get dimensions
        if isinstance(original_module, nn.Linear):
            in_features = original_module.in_features
            out_features = original_module.out_features
        elif isinstance(original_module, nn.Conv1d):
            in_features = original_module.in_channels
            out_features = original_module.out_channels
        else:
            raise ValueError(f"Unsupported module type: {type(original_module)}")

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(config.rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, config.rank))
        self.dropout = nn.Dropout(config.dropout)

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = self.original(x)
        lora_output = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.config.scaling
        return original_output + lora_output

    def get_adapter_state(self) -> dict:
        """Get only LoRA parameters for saving."""
        return {
            "lora_A": self.lora_A.data,
            "lora_B": self.lora_B.data,
            "config": asdict(self.config),
        }
```

### 6. EWC Regularization

**Purpose:** Prevent catastrophic forgetting during incremental training.

```python
class EWCRegularizer:
    """Elastic Weight Consolidation for continual learning."""

    def __init__(self, model: nn.Module, lambda_ewc: float = 1000.0):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_matrices: dict[str, torch.Tensor] = {}
        self.optimal_params: dict[str, torch.Tensor] = {}

    def compute_fisher(
        self,
        dataloader: DataLoader,
        num_samples: int = 200,
    ):
        """Compute Fisher information matrix from training data."""
        self.fisher_matrices = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self.model.eval()
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break

            self.model.zero_grad()
            output = self.model(batch)
            loss = F.cross_entropy(output, batch["target"])
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher_matrices[n] += p.grad.data ** 2

        # Normalize
        for n in self.fisher_matrices:
            self.fisher_matrices[n] /= num_samples

        # Store optimal params
        self.optimal_params = {
            n: p.data.clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def penalty(self) -> torch.Tensor:
        """Calculate EWC penalty term."""
        loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher_matrices:
                loss += (
                    self.fisher_matrices[n] *
                    (p - self.optimal_params[n]) ** 2
                ).sum()
        return self.lambda_ewc * loss

    def save(self, path: str):
        """Save Fisher matrices and optimal params."""
        torch.save({
            "fisher": self.fisher_matrices,
            "optimal": self.optimal_params,
        }, path)

    def load(self, path: str):
        """Load Fisher matrices and optimal params."""
        data = torch.load(path)
        self.fisher_matrices = data["fisher"]
        self.optimal_params = data["optimal"]
```

## Database Schema Extensions

```sql
-- Training job tracking
CREATE TABLE training_jobs (
    id VARCHAR(36) PRIMARY KEY,
    profile_id VARCHAR(36) NOT NULL REFERENCES voice_profiles(id) ON DELETE CASCADE,
    job_type VARCHAR(20) NOT NULL,  -- 'initial', 'incremental', 'technique'
    status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- 'pending', 'running', 'completed', 'failed'
    created TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    started TIMESTAMP WITH TIME ZONE,
    completed TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    samples_count INTEGER NOT NULL,
    total_duration_sec FLOAT NOT NULL,
    result_version VARCHAR(20),
    metrics JSONB
);

-- Model version tracking
CREATE TABLE model_versions (
    id VARCHAR(36) PRIMARY KEY,
    profile_id VARCHAR(36) NOT NULL REFERENCES voice_profiles(id) ON DELETE CASCADE,
    version VARCHAR(20) NOT NULL,
    created TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    adapter_path TEXT NOT NULL,
    samples_trained INTEGER NOT NULL,
    metrics JSONB NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE(profile_id, version)
);

-- Mark samples as processed after training
ALTER TABLE training_samples
ADD COLUMN training_job_id VARCHAR(36) REFERENCES training_jobs(id);
```

## API Endpoints

### Training Management

```
POST /api/v1/training/jobs
  Create manual training job for profile

GET /api/v1/training/jobs
  List training jobs (filter by profile_id, status)

GET /api/v1/training/jobs/{job_id}
  Get job details and progress

DELETE /api/v1/training/jobs/{job_id}
  Cancel pending job

GET /api/v1/profiles/{profile_id}/versions
  List model versions for profile

POST /api/v1/profiles/{profile_id}/versions/{version}/activate
  Activate specific version (rollback)

POST /api/v1/profiles/{profile_id}/versions/compare
  A/B compare two versions
```

## GPU Resource Management

```python
class GPUResourceManager:
    """Manage GPU resources for training jobs."""

    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for training")
        self.device = torch.device("cuda")
        self.lock = threading.Lock()

    def get_memory_info(self) -> dict:
        """Get current GPU memory usage."""
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "max_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2,
        }

    def can_start_job(self, estimated_memory_mb: float) -> bool:
        """Check if enough GPU memory available for job."""
        info = self.get_memory_info()
        available = info["total_mb"] - info["allocated_mb"]
        return available >= estimated_memory_mb * 1.2  # 20% buffer

    @contextmanager
    def training_context(self):
        """Context manager for exclusive training access."""
        with self.lock:
            torch.cuda.empty_cache()
            yield self.device
            torch.cuda.empty_cache()
```

## Implementation Checklist

### Phase 4 Tasks (aligned with plan.md)
- [ ] TrainingJobManager with job queue
- [ ] LoRA adapter framework
- [ ] EWC regularization implementation
- [ ] Model version management
- [ ] Training scheduler
- [ ] GPU-only execution enforcement

### API Integration
- [ ] Training job endpoints
- [ ] Version management endpoints
- [ ] Progress WebSocket events

### Database Migrations
- [ ] training_jobs table
- [ ] model_versions table
- [ ] training_samples foreign key

---

_Architecture design for AutoVoice continuous learning system. Task 2.5 complete._
