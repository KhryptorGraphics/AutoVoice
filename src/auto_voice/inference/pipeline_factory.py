"""Pipeline factory for unified voice conversion pipeline management.

Provides lazy loading, caching, and memory management for:
- RealtimePipeline: Low-latency karaoke (22kHz, simple decoder)
- SOTAConversionPipeline: High-quality CoMoSVC (24kHz, 30-step diffusion)
- SeedVCPipeline: SOTA quality with DiT-CFM (44kHz, 5-10 step flow matching)

Usage:
    factory = PipelineFactory.get_instance()
    pipeline = factory.get_pipeline('realtime')  # or 'quality' or 'quality_seedvc'
    result = pipeline.convert(audio, sr, speaker_embedding)
"""
import logging
from typing import Dict, Literal, Optional, TYPE_CHECKING, Union

import torch
from auto_voice.runtime_contract import PIPELINE_DEFINITIONS, get_pipeline_status_template

if TYPE_CHECKING:
    from .realtime_pipeline import RealtimePipeline
    from .sota_pipeline import SOTAConversionPipeline
    from .seed_vc_pipeline import SeedVCPipeline
    from .meanvc_pipeline import MeanVCPipeline

logger = logging.getLogger(__name__)

PipelineType = Literal['realtime', 'quality', 'quality_seedvc', 'realtime_meanvc', 'quality_shortcut']


class PipelineFactory:
    """Factory for creating and managing voice conversion pipelines.

    Features:
    - Lazy loading: Pipelines only initialized when first requested
    - Caching: Re-uses existing pipeline instances
    - Memory management: Can unload pipelines to free GPU memory
    - Unified interface: Both pipelines accessible via same API

    The factory is a singleton to ensure pipeline instances are shared
    across the application and avoid duplicate GPU memory allocation.
    """

    _instance: Optional['PipelineFactory'] = None

    def __init__(self, device: Optional[torch.device] = None):
        """Initialize factory.

        Args:
            device: Target device for pipelines (default: CUDA if available)
        """
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self._pipelines: Dict[str, Union['RealtimePipeline', 'SOTAConversionPipeline', 'SeedVCPipeline', 'MeanVCPipeline']] = {}
        self._memory_usage: Dict[str, float] = {}  # GB per pipeline

    @classmethod
    def get_instance(cls, device: Optional[torch.device] = None) -> 'PipelineFactory':
        """Get singleton instance of the factory.

        Creates the factory instance on first call, then returns the same
        instance on subsequent calls to ensure pipelines are shared across
        the application.

        Args:
            device: Target device for pipelines (default: CUDA if available)

        Returns:
            The singleton PipelineFactory instance
        """
        if cls._instance is None:
            cls._instance = cls(device=device)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance and unload all pipelines.

        Unloads all cached pipelines and destroys the factory instance.
        Primarily used for testing to ensure a clean state between tests.
        """
        if cls._instance is not None:
            cls._instance.unload_all()
            cls._instance = None

    def get_pipeline(
        self,
        pipeline_type: PipelineType,
        profile_store: Optional['VoiceProfileStore'] = None,
    ) -> Union['RealtimePipeline', 'SOTAConversionPipeline', 'SeedVCPipeline', 'MeanVCPipeline']:
        """Get or create a pipeline instance.

        Args:
            pipeline_type: 'realtime', 'quality' (CoMoSVC), 'quality_seedvc' (DiT-CFM),
                          or 'realtime_meanvc' (streaming mean flow)
            profile_store: Optional voice profile store for SOTA pipeline

        Returns:
            Pipeline instance ready for conversion

        Raises:
            RuntimeError: If pipeline creation fails
            ValueError: If unknown pipeline_type
        """
        if pipeline_type not in PIPELINE_DEFINITIONS:
            raise ValueError(
                f"Unknown pipeline type: {pipeline_type}. "
                f"Use 'realtime', 'quality', 'quality_seedvc', 'realtime_meanvc', or 'quality_shortcut'"
            )

        if pipeline_type in self._pipelines:
            logger.debug(f"Returning cached {pipeline_type} pipeline")
            return self._pipelines[pipeline_type]

        logger.info(f"Creating {pipeline_type} pipeline on {self.device}")

        # Track memory before and after
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / 1e9
        else:
            mem_before = 0

        pipeline = self._create_pipeline(pipeline_type, profile_store)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated() / 1e9
            self._memory_usage[pipeline_type] = mem_after - mem_before
            logger.info(f"{pipeline_type} pipeline allocated {self._memory_usage[pipeline_type]:.2f}GB GPU memory")

        self._pipelines[pipeline_type] = pipeline
        return pipeline

    def _create_pipeline(
        self,
        pipeline_type: PipelineType,
        profile_store: Optional['VoiceProfileStore'] = None,
    ) -> Union['RealtimePipeline', 'SOTAConversionPipeline', 'SeedVCPipeline', 'MeanVCPipeline']:
        """Create a new pipeline instance.

        Args:
            pipeline_type: Type of pipeline to create
            profile_store: Optional voice profile store for SOTA pipeline

        Returns:
            Initialized pipeline instance

        Raises:
            ValueError: If unknown pipeline_type
        """
        if pipeline_type == 'realtime':
            from .realtime_pipeline import RealtimePipeline
            return RealtimePipeline(device=self.device)

        elif pipeline_type == 'quality':
            from .sota_pipeline import SOTAConversionPipeline
            return SOTAConversionPipeline(
                device=self.device,
                n_steps=1,  # Consistency model: 1 step for speed
                profile_store=profile_store,
                require_gpu=True,
            )

        elif pipeline_type == 'quality_seedvc':
            from .seed_vc_pipeline import SeedVCPipeline
            return SeedVCPipeline(
                device=self.device,
                diffusion_steps=10,  # DiT-CFM: 10 steps for quality
                f0_condition=True,   # Always True for singing voice
                require_gpu=True,
            )

        elif pipeline_type == 'realtime_meanvc':
            from .meanvc_pipeline import MeanVCPipeline
            # MeanVC is optimized for CPU streaming - lightweight 14M model
            return MeanVCPipeline(
                device=torch.device('cpu'),  # CPU for true streaming
                steps=2,  # 2-step inference for quality
                require_gpu=False,  # Can run on CPU!
            )

        elif pipeline_type == 'quality_shortcut':
            from .seed_vc_pipeline import SeedVCPipeline
            # Shortcut flow matching: 2-step inference with 2.83x speedup
            return SeedVCPipeline(
                device=self.device,
                diffusion_steps=2,  # 2-step shortcut flow matching
                f0_condition=True,  # Always True for singing voice
                require_gpu=True,
            )

        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")

    def is_loaded(self, pipeline_type: PipelineType) -> bool:
        """Check if a pipeline is currently loaded.

        Args:
            pipeline_type: Pipeline to check

        Returns:
            True if pipeline is loaded in memory, False otherwise
        """
        return pipeline_type in self._pipelines

    def get_memory_usage(self, pipeline_type: PipelineType) -> float:
        """Get GPU memory usage for a pipeline in GB.

        Args:
            pipeline_type: Pipeline to query

        Returns:
            GPU memory allocated by pipeline in gigabytes, or 0.0 if not loaded
        """
        return self._memory_usage.get(pipeline_type, 0.0)

    def get_total_memory_usage(self) -> float:
        """Get total GPU memory usage across all pipelines in GB.

        Returns:
            Sum of GPU memory allocated by all loaded pipelines in gigabytes
        """
        return sum(self._memory_usage.values())

    def unload_pipeline(self, pipeline_type: PipelineType) -> bool:
        """Unload a pipeline to free GPU memory.

        Args:
            pipeline_type: Pipeline to unload

        Returns:
            True if pipeline was unloaded, False if not loaded
        """
        if pipeline_type not in self._pipelines:
            return False

        logger.info(f"Unloading {pipeline_type} pipeline")

        # Remove reference and trigger garbage collection
        del self._pipelines[pipeline_type]
        self._memory_usage.pop(pipeline_type, None)

        # Force CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return True

    def unload_all(self) -> None:
        """Unload all pipelines and free GPU memory.

        Iterates through all loaded pipelines and unloads them,
        triggering garbage collection and CUDA cache cleanup.
        """
        pipeline_types = list(self._pipelines.keys())
        for pt in pipeline_types:
            self.unload_pipeline(pt)

    def get_status(self) -> Dict[str, Dict]:
        """Get status and capabilities of all pipeline types.

        Returns detailed information about each pipeline type including
        load status, memory usage, performance characteristics, and features.
        Used by the API to provide pipeline information to clients.

        Returns:
            Dict mapping pipeline type to status info containing:
                - loaded: Whether pipeline is currently loaded
                - memory_gb: GPU memory usage in gigabytes
                - latency_target_ms: Expected processing latency
                - sample_rate: Output sample rate
                - description: Human-readable description
                - features: List of key features (for advanced pipelines)
        """
        status = get_pipeline_status_template()
        for pipeline_type, entry in status.items():
            entry["loaded"] = self.is_loaded(pipeline_type)  # type: ignore[index]
            entry["memory_gb"] = self.get_memory_usage(pipeline_type)  # type: ignore[index]
        return status


# Type annotation for profile store (avoid circular import)
if TYPE_CHECKING:
    from ..storage.voice_profiles import VoiceProfileStore
