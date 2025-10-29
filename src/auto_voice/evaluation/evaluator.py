"""
Voice Conversion Quality Evaluator.

This module provides a high-level interface for comprehensive quality evaluation
of singing voice conversion systems, including batch processing, report generation,
and automated regression detection.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, asdict
import time
from datetime import datetime
import threading
import base64
import io

import torch
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available, visualization features disabled")

from ..audio.processor import AudioProcessor
from ..utils.quality_metrics import (
    QualityMetricsAggregator, QualityMetricsResult,
    AudioAligner, AudioNormalizer
)
from ..utils.helpers import save_json, load_json
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationSample:
    """Represents a single evaluation sample."""
    id: str
    source_audio_path: Optional[str] = None
    target_audio_path: Optional[str] = None
    source_audio: Optional[torch.Tensor] = None
    target_audio: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = None
    result: Optional[QualityMetricsResult] = None


@dataclass
class EvaluationResults:
    """Container for comprehensive evaluation results."""
    samples: List[EvaluationSample]
    summary_stats: Dict[str, Any]
    evaluation_config: Dict[str, Any]
    evaluation_timestamp: float
    total_evaluation_time: float


@dataclass
class QualityTargets:
    """Quality targets for automated validation."""
    min_pitch_accuracy_correlation: float = 0.8
    max_pitch_accuracy_rmse_hz: float = 10.0  # Hz domain RMSE (< 10 Hz target)
    max_pitch_accuracy_rmse: float = 0.1  # Log2 domain RMSE (backwards compatibility)
    min_speaker_similarity: float = 0.85  # Updated to 0.85 as per requirements
    max_spectral_distortion: float = 10.0
    min_stoi_score: float = 0.9  # Updated to 0.9 to align with quality plan
    min_pesq_score: float = 2.0
    min_mos_estimate: float = 4.0  # Minimum MOS estimation target
    min_overall_quality_score: float = 0.75


class VoiceConversionEvaluator:
    """
    Comprehensive evaluator for singing voice conversion quality.

    Supports single conversions, batch processing, report generation, and
    automated quality regression detection with configurable targets.
    """

    def __init__(self, sample_rate: int = 44100, device: str = 'auto',
                 evaluation_config_path: Optional[str] = None):
        """
        Initialize the voice conversion evaluator.

        Args:
            sample_rate: Audio sample rate for processing
            device: PyTorch device ('auto', 'cpu', 'cuda', 'cuda:N')
            evaluation_config_path: Path to evaluation configuration file
        """
        self.sample_rate = sample_rate
        self.device = self._resolve_device(device)

        # Initialize core components
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)

        # Load configuration first
        self.config = self._load_evaluation_config(evaluation_config_path)

        # Initialize metrics aggregator with mos_method from config
        mos_method = self.config.get('metrics', {}).get('naturalness', {}).get('mos_method', 'heuristic')
        self.metrics_aggregator = QualityMetricsAggregator(sample_rate=sample_rate, mos_method=mos_method)
        self.audio_aligner = AudioAligner(sample_rate=sample_rate)
        self.audio_normalizer = AudioNormalizer()

        # Set up progress tracking
        self.progress_callbacks = []
        self.evaluation_lock = threading.RLock()

        logger.info(f"VoiceConversionEvaluator initialized with device: {self.device}")

    def _resolve_device(self, device: str) -> torch.device:
        """Resolve PyTorch device from string specification."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device.startswith('cuda'):
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                return torch.device('cpu')
            if ':' in device:
                device_num = int(device.split(':')[1])
                if device_num >= torch.cuda.device_count():
                    logger.warning(f"CUDA device {device_num} not available, using CUDA:0")
                    return torch.device('cuda:0')
            return torch.device(device)
        else:
            return torch.device(device)

    def _load_evaluation_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load evaluation configuration from YAML file."""
        default_config = {
            'align_audio': True,
            'normalize_audio': True,
            'target_rms_db': -12.0,
            'batch_size': 4,
            'enable_progress_tracking': True,
            'save_intermediate_results': False,
            'output_formats': ['markdown', 'json'],
            'visualization_options': {
                'pitch_contours': True,
                'spectrograms': False,
                'quality_dashboard': True,
                'publish_quality_plots': True
            },
            'quality_targets': asdict(QualityTargets()),
            'reports': {
                'include_raw_metrics': True,
                'include_summary_stats': True,
                'include_visualizations': True
            }
        }

        # Try to auto-load config/evaluation_config.yaml if no path provided
        if config_path is None:
            default_path = Path('config/evaluation_config.yaml')
            if default_path.exists():
                config_path = str(default_path)
                logger.info(f"Auto-loading evaluation config from: {config_path}")

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    
                # Deep merge configurations
                def merge_configs(base, update):
                    for key, value in update.items():
                        if isinstance(value, dict) and key in base:
                            merge_configs(base[key], value)
                        else:
                            base[key] = value
                    return base
                
                merge_configs(default_config, user_config)
                logger.info(f"Loaded evaluation config from: {config_path}")
                
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        return default_config

    def add_progress_callback(self, callback: callable):
        """
        Add a progress tracking callback.

        Args:
            callback: Function accepting (current: int, total: int, message: str)
        """
        self.progress_callbacks.append(callback)

    def _report_progress(self, current: int, total: int, message: str):
        """Report progress to all callbacks."""
        if not self.config['enable_progress_tracking']:
            return

        for callback in self.progress_callbacks:
            try:
                callback(current, total, message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def evaluate_single_conversion(self, source_audio: torch.Tensor,
                                 target_audio: torch.Tensor,
                                 target_speaker_embedding: Optional[np.ndarray] = None) -> QualityMetricsResult:
        """
        Evaluate quality of a single voice conversion.

        Args:
            source_audio: Source (input) audio waveform
            target_audio: Converted (output) audio waveform
            target_speaker_embedding: Optional target speaker profile embedding for similarity evaluation

        Returns:
            QualityMetricsResult: Comprehensive evaluation results
        """
        with self.evaluation_lock:
            return self.metrics_aggregator.evaluate(
                source_audio, target_audio,
                align_audio=self.config['align_audio'],
                target_speaker_embedding=target_speaker_embedding
            )

    def evaluate_conversions(self, samples: List[EvaluationSample],
                            max_workers: int = None) -> EvaluationResults:
        """
        Evaluate a batch of voice conversions.

        Args:
            samples: List of evaluation samples
            max_workers: Maximum number of worker threads (None for auto)

        Returns:
            EvaluationResults: Batch evaluation results
        """
        start_time = time.time()
        total_samples = len(samples)

        logger.info(f"Starting batch evaluation of {total_samples} samples")
        self._report_progress(0, total_samples, "Starting evaluation...")

        # Process samples (could be parallelized in future)
        processed_samples = []

        for i, sample in enumerate(samples):
            try:
                self._report_progress(i, total_samples, f"Evaluating sample {sample.id}")

                # Load audio if provided as paths
                if sample.source_audio_path and sample.source_audio is None:
                    sample.source_audio = self._load_audio(sample.source_audio_path)
                if sample.target_audio_path and sample.target_audio is None:
                    sample.target_audio = self._load_audio(sample.target_audio_path)

                if sample.source_audio is None or sample.target_audio is None:
                    logger.error(f"Missing audio data for sample {sample.id}")
                    continue

                # Evaluate quality
                result = self.evaluate_single_conversion(sample.source_audio, sample.target_audio)
                sample.result = result

                # Generate visualizations if configured (for report integration)
                if self.config['visualization_options']['publish_quality_plots']:
                    sample.visualization_paths = self._generate_sample_visualizations(
                        sample, result
                    )

                processed_samples.append(sample)

            except Exception as e:
                logger.error(f"Evaluation failed for sample {sample.id}: {e}")
                continue

        self._report_progress(total_samples, total_samples, "Evaluation complete")

        # Compute summary statistics
        summary_stats = self._compute_batch_summary_statistics(processed_samples)

        # Create results object
        results = EvaluationResults(
            samples=processed_samples,
            summary_stats=summary_stats,
            evaluation_config=self.config,
            evaluation_timestamp=time.time(),
            total_evaluation_time=time.time() - start_time
        )

        logger.info(f"Batch evaluation completed in {results.total_evaluation_time:.2f} seconds")
        return results

    def _load_audio(self, audio_path: str) -> Optional[torch.Tensor]:
        """Load audio file and convert to tensor."""
        try:
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return None

            # Use audio processor to load audio
            audio, sr = self.audio_processor.load_audio(audio_path)

            # Resample if necessary
            if sr != self.sample_rate:
                audio = self.audio_processor.resample_audio(audio, sr, self.sample_rate)

            return audio

        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            return None

    def _generate_sample_visualizations(self, sample: EvaluationSample,
                                       result, output_dir: Optional[Path] = None) -> Dict[str, str]:
        """
        Generate visualization plots for a single evaluation sample.

        Args:
            sample: The evaluation sample
            result: Quality metrics result
            output_dir: Optional output directory for plots (if None, returns placeholder paths)

        Returns:
            Dictionary mapping visualization type to file path
        """
        viz_paths = {}

        # If no output_dir, just return placeholder paths for later generation
        if output_dir is None:
            if self.config['visualization_options']['pitch_contours']:
                viz_paths['pitch_contour'] = f"{sample.id}_pitch_contour.png"
            if self.config['visualization_options']['spectrograms']:
                viz_paths['spectrogram'] = f"{sample.id}_spectrogram.png"
            return viz_paths

        try:
            # Import visualization utilities
            from ..utils.visualization import (
                PitchContourVisualizer,
                SpectrogramVisualizer,
                PitchContourData
            )
            from ..audio.pitch_extractor import SingingPitchExtractor

            # Create plots directory
            plots_dir = output_dir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Generate pitch contour visualization if configured
            if self.config['visualization_options']['pitch_contours']:
                if hasattr(result, 'pitch_accuracy') and result.pitch_accuracy is not None:
                    try:
                        # Extract or reuse F0 contours
                        pitch_extractor = SingingPitchExtractor()

                        # Extract F0 from source audio
                        source_f0_dict = pitch_extractor.extract_f0_contour(
                            sample.source_audio,
                            sample_rate=self.sample_rate,
                            return_times=True
                        )

                        # Extract F0 from converted audio
                        target_f0_dict = pitch_extractor.extract_f0_contour(
                            sample.target_audio,
                            sample_rate=self.sample_rate,
                            return_times=True
                        )

                        # Create pitch contour data objects
                        source_pitch_data = PitchContourData(
                            f0=source_f0_dict['f0'],
                            times=source_f0_dict.get('times', None),
                            confidence=source_f0_dict.get('confidence', None)
                        )

                        target_pitch_data = PitchContourData(
                            f0=target_f0_dict['f0'],
                            times=target_f0_dict.get('times', None),
                            confidence=target_f0_dict.get('confidence', None)
                        )

                        # Generate pitch contour comparison plot
                        pitch_plot_path = plots_dir / f"{sample.id}_pitch.png"
                        visualizer = PitchContourVisualizer()
                        visualizer.plot_pitch_contour_comparison(
                            source_pitch=source_pitch_data,
                            target_pitch=target_pitch_data,
                            output_path=str(pitch_plot_path),
                            title=f"Pitch Contour Comparison - {sample.id}"
                        )

                        viz_paths['pitch_contour'] = str(pitch_plot_path.relative_to(output_dir))
                        logger.debug(f"Generated pitch contour plot: {pitch_plot_path}")

                    except Exception as e:
                        logger.warning(f"Failed to generate pitch contour plot for {sample.id}: {e}")

            # Generate spectrogram visualization if configured
            if self.config['visualization_options']['spectrograms']:
                try:
                    # Generate spectrogram comparison
                    spec_plot_path = plots_dir / f"{sample.id}_spectrogram.png"
                    spec_visualizer = SpectrogramVisualizer()
                    spec_visualizer.plot_spectrogram_comparison(
                        source_audio=sample.source_audio,
                        target_audio=sample.target_audio,
                        sample_rate=self.sample_rate,
                        output_path=str(spec_plot_path),
                        title=f"Spectrogram Comparison - {sample.id}"
                    )

                    viz_paths['spectrogram'] = str(spec_plot_path.relative_to(output_dir))
                    logger.debug(f"Generated spectrogram plot: {spec_plot_path}")

                except Exception as e:
                    logger.warning(f"Failed to generate spectrogram plot for {sample.id}: {e}")

        except ImportError as e:
            logger.warning(f"Visualization utilities not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to generate visualizations for sample {sample.id}: {e}")

        return viz_paths

    def _generate_all_sample_visualizations(self, results: EvaluationResults,
                                            output_dir: Path):
        """Generate visualizations for all samples in batch evaluation."""
        logger.info("Generating visualizations for all samples...")

        # Generate summary dashboard if configured
        if self.config['visualization_options']['quality_dashboard']:
            self._generate_summary_dashboard(results, output_dir)

        for sample in results.samples:
            if sample.result:
                try:
                    viz_paths = self._generate_sample_visualizations(
                        sample, sample.result, output_dir=output_dir
                    )
                    # Store visualization paths in sample for report embedding
                    if not hasattr(sample, 'visualization_paths'):
                        sample.visualization_paths = {}
                    sample.visualization_paths.update(viz_paths)
                except Exception as e:
                    logger.warning(f"Failed to generate visualizations for sample {sample.id}: {e}")

    def _generate_summary_dashboard(self, results: EvaluationResults, output_dir: Path):
        """Generate summary dashboard with aggregated visualizations."""
        try:
            from ..utils.visualization import QualityMetricsVisualizer

            dashboard_path = output_dir / 'dashboard.png'
            visualizer = QualityMetricsVisualizer()
            visualizer.create_summary_dashboard(
                summary_stats=results.summary_stats,
                output_path=str(dashboard_path),
                title="Voice Conversion Quality Dashboard"
            )
            logger.info(f"Generated summary dashboard: {dashboard_path}")
            return str(dashboard_path.relative_to(output_dir))
        except Exception as e:
            logger.warning(f"Failed to generate summary dashboard: {e}")
            return None

    def _compute_batch_summary_statistics(self, samples: List[EvaluationSample]) -> Dict[str, Any]:
        """Compute summary statistics for batch evaluation."""
        if not samples or not any(s.result for s in samples):
            return {}

        valid_results = [s.result for s in samples if s.result is not None]
        return self.metrics_aggregator.get_summary_statistics(valid_results)

    def validate_quality_targets(self, results: EvaluationResults,
                               targets: Optional[QualityTargets] = None) -> Dict[str, Any]:
        """
        Validate results against quality targets for automated checks.

        Args:
            results: Evaluation results to validate
            targets: Quality targets to validate against

        Returns:
            Dict containing validation results and pass/fail status
        """
        if targets is None:
            targets = QualityTargets(**self.config.get('quality_targets', {}))

        validation_results = {
            'overall_pass': True,
            'target_validations': {},
            'failed_targets': [],
            'sample_validations': []
        }

        summary_stats = results.summary_stats

        # Validate summary statistics against targets
        if 'pitch_accuracy' in summary_stats:
            pitch_stats = summary_stats['pitch_accuracy']
            corr_mean = pitch_stats.get('correlation', {}).get('mean', 0.0)
            rmse_hz_mean = pitch_stats.get('rmse_hz', {}).get('mean', float('inf'))

            if corr_mean < targets.min_pitch_accuracy_correlation:
                validation_results['target_validations']['min_pitch_accuracy_correlation'] = False
                validation_results['failed_targets'].append('pitch_accuracy_correlation')
                validation_results['overall_pass'] = False
            else:
                validation_results['target_validations']['min_pitch_accuracy_correlation'] = True

            # Validate Hz RMSE (new requirement)
            if rmse_hz_mean > targets.max_pitch_accuracy_rmse_hz:
                validation_results['target_validations']['max_pitch_accuracy_rmse_hz'] = False
                validation_results['failed_targets'].append('pitch_accuracy_rmse_hz')
                validation_results['overall_pass'] = False
            else:
                validation_results['target_validations']['max_pitch_accuracy_rmse_hz'] = True

        # Validate speaker similarity
        if 'speaker_similarity' in summary_stats:
            speaker_stats = summary_stats['speaker_similarity']
            cosine_sim_mean = speaker_stats.get('cosine_similarity', {}).get('mean', 0.0)

            if cosine_sim_mean < targets.min_speaker_similarity:
                validation_results['target_validations']['min_speaker_similarity'] = False
                validation_results['failed_targets'].append('speaker_similarity')
                validation_results['overall_pass'] = False
            else:
                validation_results['target_validations']['min_speaker_similarity'] = True

        # Validate intelligibility (STOI)
        if 'intelligibility' in summary_stats:
            intel_stats = summary_stats['intelligibility']
            stoi_mean = intel_stats.get('stoi', {}).get('mean', 0.0)

            if stoi_mean < targets.min_stoi_score:
                validation_results['target_validations']['min_stoi_score'] = False
                validation_results['failed_targets'].append('stoi_score')
                validation_results['overall_pass'] = False
            else:
                validation_results['target_validations']['min_stoi_score'] = True

        # Validate naturalness (MOS estimation)
        if 'naturalness' in summary_stats:
            nat_stats = summary_stats['naturalness']
            # Try to get MOS from summary stats first
            mos_mean = nat_stats.get('mos_estimation', {}).get('mean', None)

            # If not in summary, compute from per-sample results
            if mos_mean is None:
                mos_values = []
                for sample in results.samples:
                    if sample.result and hasattr(sample.result, 'naturalness'):
                        if hasattr(sample.result.naturalness, 'mos_estimation'):
                            mos_values.append(sample.result.naturalness.mos_estimation)

                if mos_values:
                    mos_mean = np.mean(mos_values)

            if mos_mean is not None:
                if mos_mean < targets.min_mos_estimate:
                    validation_results['target_validations']['min_mos_estimate'] = False
                    validation_results['failed_targets'].append('mos_estimate')
                    validation_results['overall_pass'] = False
                else:
                    validation_results['target_validations']['min_mos_estimate'] = True

        validation_results['targets_used'] = asdict(targets)

        return validation_results

    def generate_reports(self, results: EvaluationResults,
                        output_dir: Union[str, Path],
                        formats: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Generate comprehensive evaluation reports.

        Args:
            results: Evaluation results to report on
            output_dir: Directory to save reports
            formats: Report formats to generate ('markdown', 'json', 'html')

        Returns:
            Dict mapping format names to output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if formats is None:
            formats = self.config.get('output_formats', ['markdown', 'json'])

        # Generate visualizations for all samples if configured
        if self.config['visualization_options']['publish_quality_plots']:
            logger.info("Generating visualizations for evaluation samples...")
            for sample in results.samples:
                if sample.result:
                    try:
                        viz_paths = self._generate_sample_visualizations(
                            sample, sample.result, output_dir=output_dir
                        )
                        # Store visualization paths in sample metadata for report embedding
                        if not hasattr(sample, 'visualization_paths'):
                            sample.visualization_paths = {}
                        sample.visualization_paths.update(viz_paths)
                    except Exception as e:
                        logger.warning(f"Failed to generate visualizations for sample {sample.id}: {e}")

        output_files = {}

        # Generate visualizations first (if configured)
        if self.config['visualization_options']['publish_quality_plots'] or \
           self.config['reports']['include_visualizations']:
            # Generate sample visualizations
            self._generate_all_sample_visualizations(results, output_dir)

        # Generate Markdown report
        if 'markdown' in formats:
            md_file = output_dir / 'evaluation_report.md'
            self._generate_markdown_report(results, md_file, output_dir)
            output_files['markdown'] = str(md_file)

        # Generate JSON report
        if 'json' in formats:
            json_file = output_dir / 'evaluation_results.json'
            self._generate_json_report(results, json_file)
            output_files['json'] = str(json_file)

        # Generate HTML dashboard (if configured)
        if 'html' in formats and self.config['visualization_options']['quality_dashboard']:
            html_file = output_dir / 'evaluation_dashboard.html'
            self._generate_html_dashboard(results, html_file, output_dir)
            output_files['html'] = str(html_file)

        logger.info(f"Reports generated: {list(output_files.keys())}")
        return output_files

    def _generate_markdown_report(self, results: EvaluationResults, output_path: Path, output_dir: Optional[Path] = None):
        """Generate comprehensive Markdown report with embedded visualizations."""
        timestamp = datetime.fromtimestamp(results.evaluation_timestamp)
        if output_dir is None:
            output_dir = output_path.parent

        with open(output_path, 'w') as f:
            f.write("# Voice Conversion Quality Evaluation Report\n\n")
            f.write(f"**Evaluation Date:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Samples:** {len(results.samples)}\n")
            f.write(f"**Evaluation Time:** {results.total_evaluation_time:.2f} seconds\n\n")

            # Summary statistics
            if results.summary_stats:
                f.write("## Summary Statistics\n\n")

                for category, metrics in results.summary_stats.items():
                    f.write(f"### {category.replace('_', ' ').title()}\n\n")
                    if isinstance(metrics, dict):
                        for metric_name, stats in metrics.items():
                            if isinstance(stats, dict):
                                f.write(f"- **{metric_name}:**\n")
                                for stat_name, value in stats.items():
                                    f.write(f"  - {stat_name}: {value:.3f}\n")
                                f.write("\n")
                    f.write("\n")

            # Individual sample results
            f.write("## Individual Sample Results\n\n")

            for i, sample in enumerate(results.samples):
                f.write(f"### Sample {i+1}: {sample.id}\n\n")

                if sample.metadata:
                    f.write("**Metadata:**\n")
                    for k, v in sample.metadata.items():
                        if k == 'target_profile_id':
                            f.write(f"- {k}: {v}\n")
                        elif k == 'conversion_params':
                            f.write(f"- {k}: {v}\n")
                        elif k == 'reference_audio_path':
                            f.write(f"- {k}: {v}\n")
                        else:
                            # Include other metadata as-is
                            f.write(f"- {k}: {v}\n")
                    f.write("\n")

                if sample.result:
                    result = sample.result
                    f.write("**Quality Scores:**\n\n")
                    f.write("| Metric | Value | Confidence |\n")
                    f.write("|--------|-------|------------|\n")
                    f.write(f"| Pitch Accuracy | {result.pitch_accuracy.confidence_score:.3f} | {result.pitch_accuracy.confidence_score:.3f} |\n")
                    f.write(f"| Speaker Similarity | {result.speaker_similarity.confidence_score:.3f} | {result.speaker_similarity.confidence_score:.3f} |\n")
                    f.write(f"| Naturalness | {result.naturalness.confidence_score:.3f} | {result.naturalness.confidence_score:.3f} |\n")
                    f.write(f"| Intelligibility | {result.intelligibility.confidence_score:.3f} | {result.intelligibility.confidence_score:.3f} |\n")
                    f.write(f"| Overall | {result.overall_quality_score:.3f} | N/A |\n")
                    f.write("\n")
                    f.write("**Pitch Accuracy:**\n")
                    f.write(f"- RMSE (Hz): {result.pitch_accuracy.rmse_hz:.3f} Hz\n")
                    f.write(f"- RMSE (log2): {result.pitch_accuracy.rmse_log2:.3f} semitones\n")
                    f.write(f"- Correlation: {result.pitch_accuracy.correlation:.3f}\n")
                    f.write(f"- Confidence: {result.pitch_accuracy.confidence_score:.3f}\n\n")

                    f.write("**Speaker Similarity:**\n")
                    f.write(f"- Cosine Similarity: {result.speaker_similarity.cosine_similarity:.3f}\n")
                    f.write(f"- Embedding Distance: {result.speaker_similarity.embedding_distance:.3f}\n")
                    f.write("\n")

                    f.write("**Naturalness:**\n")
                    f.write(f"- Spectral Distortion: {result.naturalness.spectral_distortion:.3f} dB\n")
                    f.write(f"- MOS Estimation: {result.naturalness.mos_estimation:.3f}\n")
                    f.write(f"- Confidence: {result.naturalness.confidence_score:.3f}\n\n")

                    f.write("**Intelligibility:**\n")
                    f.write(f"- STOI: {result.intelligibility.stoi_score:.3f}\n")
                    f.write(f"- ESTOI: {result.intelligibility.estoi_score:.3f}\n")
                    f.write(f"- PESQ: {result.intelligibility.pesq_score:.3f}\n\n")

                # Embed visualizations if available
                if hasattr(sample, 'visualization_paths') and sample.visualization_paths:
                    f.write("**Visualizations:**\n\n")

                    if 'pitch_contour' in sample.visualization_paths:
                        pitch_path = sample.visualization_paths['pitch_contour']
                        f.write(f"![Pitch Contour]({pitch_path})\n\n")

                    if 'spectrogram' in sample.visualization_paths:
                        spec_path = sample.visualization_paths['spectrogram']
                        f.write(f"![Spectrogram]({spec_path})\n\n")

                f.write("---\n\n")

    def _generate_json_report(self, results: EvaluationResults, output_path: Path):
        """Generate JSON report with comprehensive data."""
        # Convert results to serializable format
        json_data = {
            'evaluation_timestamp': results.evaluation_timestamp,
            'total_evaluation_time': results.total_evaluation_time,
            'total_samples': len(results.samples),
            'config': results.evaluation_config,
            'summary_stats': results.summary_stats,
            'samples': []
        }

        for sample in results.samples:
            sample_data = {
                'id': sample.id,
                'audio_paths': {
                    'source': sample.source_audio_path,
                    'target': sample.target_audio_path
                },
                'metadata': sample.metadata or {}
            }

            if sample.result:
                sample_data['results'] = {
                    'pitch_accuracy': self._result_to_dict(sample.result.pitch_accuracy),
                    'speaker_similarity': self._result_to_dict(sample.result.speaker_similarity),
                    'naturalness': self._result_to_dict(sample.result.naturalness),
                    'intelligibility': self._result_to_dict(sample.result.intelligibility),
                    'overall_quality_score': sample.result.overall_quality_score,
                    'processing_time_seconds': sample.result.processing_time_seconds
                }

            json_data['samples'].append(sample_data)

        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)

    def _generate_html_dashboard(self, results: EvaluationResults, output_path: Path, output_dir: Optional[Path] = None):
        """Generate comprehensive HTML quality dashboard with embedded visualizations."""
        if output_dir is None:
            output_dir = output_path.parent

        timestamp = datetime.fromtimestamp(results.evaluation_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        total_samples = len(results.samples)

        # Build summary dashboard section
        summary_section = ""
        if results.summary_stats:
            summary_section += "<h2>Summary Statistics</h2>"

            # Overall quality overview
            meta_eval = results.summary_stats.get('metadata_evaluation', {})
            successful = meta_eval.get('successful_evaluations', 0)
            total_tests = meta_eval.get('total_test_cases', len(results.samples))

            success_rate = (successful / max(total_tests, 1)) * 100
            summary_section += f"""
            <div class="metric">
                <h3>Evaluation Summary</h3>
                <p><strong>Total Tests:</strong> {total_tests}</p>
                <p><strong>Successful:</strong> {successful} ({success_rate:.1f}%)</p>
                <p><strong>Evaluation Time:</strong> {results.total_evaluation_time:.2f} seconds</p>
            </div>
            """

            # Quality metrics breakdown
            for category, metrics in results.summary_stats.items():
                if category == 'metadata_evaluation':
                    continue

                summary_section += f"<h3>{category.replace('_', ' ').title()} Metrics</h3>"
                summary_section += '<div class="metric">'

                if isinstance(metrics, dict):
                    for metric_name, stats in metrics.items():
                        if isinstance(stats, dict) and 'mean' in stats:
                            mean_val = stats['mean']
                            std_val = stats.get('std', 0)
                            summary_section += f"<p><strong>{metric_name}:</strong> {mean_val:.3f} ± {std_val:.3f}</p>"
                        elif isinstance(stats, (int, float)):
                            summary_section += f"<p><strong>{metric_name}:</strong> {stats}</p>"

                summary_section += "</div>"

        # Build individual sample sections with embedded visuals
        sample_sections = ""
        for i, sample in enumerate(results.samples):
            sample_sections += f'<h2>Sample {i+1}: {sample.id}</h2>'

            # Quality metrics table
            if sample.result:
                result = sample.result
                sample_sections += """
                <div class="metric">
                <h3>Quality Metrics</h3>
                <table>
                <tr><th>Metric</th><th>Value</th><th>Confidence</th></tr>
                """
                sample_sections += f"<tr><td>Pitch Accuracy</td><td>{result.pitch_accuracy.confidence_score:.3f}</td><td>{result.pitch_accuracy.confidence_score:.3f}</td></tr>"
                sample_sections += f"<tr><td>Speaker Similarity</td><td>{result.speaker_similarity.confidence_score:.3f}</td><td>{result.speaker_similarity.confidence_score:.3f}</td></tr>"
                sample_sections += f"<tr><td>Naturalness</td><td>{result.naturalness.confidence_score:.3f}</td><td>{result.naturalness.confidence_score:.3f}</td></tr>"
                sample_sections += f"<tr><td>Intelligibility</td><td>{result.intelligibility.confidence_score:.3f}</td><td>{result.intelligibility.confidence_score:.3f}</td></tr>"
                sample_sections += f"<tr><td><strong>Overall</strong></td><td><strong>{result.overall_quality_score:.3f}</strong></td><td>N/A</td></tr>"
                sample_sections += "</table></div>"

                # Detailed metrics
                sample_sections += """
                <div class="metric">
                <h3>Detailed Pitch Metrics</h3>
                """
                sample_sections += f"<p><strong>RMSE (Hz):</strong> {result.pitch_accuracy.rmse_hz:.3f} Hz</p>"
                sample_sections += f"<p><strong>RMSE (log2):</strong> {result.pitch_accuracy.rmse_log2:.3f} semitones</p>"
                sample_sections += f"<p><strong>Correlation:</strong> {result.pitch_accuracy.correlation:.3f}</p>"
                sample_sections += "</div>"

            # Embed visualizations
            viz_section = ""
            if hasattr(sample, 'visualization_paths') and sample.visualization_paths:
                viz_section += '<div class="metric"><h3>Visualizations</h3>'

                if 'pitch_contour' in sample.visualization_paths:
                    pitch_img_path = (output_dir / sample.visualization_paths['pitch_contour']).resolve()
                    if pitch_img_path.exists():
                        try:
                            from ..utils.visualization import create_embedded_markdown_image
                            # For HTML, we embed the image directly
                            img_data = ""
                            with open(pitch_img_path, 'rb') as f:
                                import base64
                                img_data = base64.b64encode(f.read()).decode('utf-8')
                            viz_section += f'<h4>Pitch Contour Comparison</h4><img src="data:image/png;base64,{img_data}" alt="Pitch Contour" style="max-width:100%;">'
                        except Exception:
                            viz_section += f'<p><em>Pitch contour plot: {sample.visualization_paths["pitch_contour"]}</em></p>'

                if 'spectrogram' in sample.visualization_paths:
                    spec_img_path = (output_dir / sample.visualization_paths['spectrogram']).resolve()
                    if spec_img_path.exists():
                        try:
                            img_data = ""
                            with open(spec_img_path, 'rb') as f:
                                import base64
                                img_data = base64.b64encode(f.read()).decode('utf-8')
                            viz_section += f'<h4>Spectrogram Comparison</h4><img src="data:image/png;base64,{img_data}" alt="Spectrogram" style="max-width:100%;">'
                        except Exception:
                            viz_section += f'<p><em>Spectrogram plot: {sample.visualization_paths["spectrogram"]}</em></p>'

                viz_section += "</div>"

            sample_sections += viz_section

        # Full HTML content
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Voice Conversion Quality Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 40px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #2c3e50; margin-bottom: 30px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; margin-top: 50px; }}
        h3 {{ color: #34495e; }}
        .metric {{
            background: linear-gradient(135deg, #ecf0f1 0%, #f8f9fa 100%);
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        tr:hover {{ background-color: #f1f2f6; }}
        img {{
            border-radius: 5px;
            margin: 10px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .summary {{
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .summary h3 {{ margin-top: 0; }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .overview {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }}
        .stat-box {{
            text-align: center;
            padding: 15px;
            background: #ecf0f1;
            border-radius: 5px;
            flex: 1;
            margin: 0 10px;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Voice Conversion Quality Dashboard</h1>
        <div class="timestamp">Evaluation completed at: {timestamp}</div>

        <div class="overview">
            <div class="stat-box">
                <div class="stat-number">{total_samples}</div>
                <div class="stat-label">Total Samples</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{results.total_evaluation_time:.1f}</div>
                <div class="stat-label">Evaluation Time (s)</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{len([s for s in results.samples if s.result])}</div>
                <div class="stat-label">Evaluated Samples</div>
            </div>
        </div>

        {summary_section}

        {sample_sections}
    </div>
</body>
</html>
        """

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _result_to_dict(self, result) -> Dict[str, Any]:
        """Convert result object to dictionary for JSON serialization."""
        if hasattr(result, '__dict__'):
            data = asdict(result)
        elif hasattr(result, '__dataclass_fields__'):
            data = asdict(result)
        else:
            data = result.__dict__ if hasattr(result, '__dict__') else {}

        # Handle numpy arrays in the data
        def convert_value(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            elif isinstance(v, torch.Tensor):
                return v.detach().cpu().numpy().tolist()
            elif isinstance(v, np.floating):
                return float(v)
            elif isinstance(v, np.integer):
                return int(v)
            elif isinstance(v, (list, tuple)):
                return [convert_value(x) for x in v]
            elif isinstance(v, dict):
                return {k: convert_value(v2) for k, v2 in v.items()}
            else:
                return v

        return convert_value(data)

    def evaluate_test_set(self, metadata_path: str, output_report_path: Optional[str] = None):
        """
        Evaluate a test set using metadata-driven approach with pipeline conversion.

        Args:
            metadata_path: Path to JSON file containing test cases with metadata
            output_report_path: Optional path to save detailed evaluation report

        Returns:
            EvaluationResults: Batch evaluation results from metadata-driven conversion
        """
        start_time = time.time()
        logger.info(f"Starting metadata-driven evaluation from: {metadata_path}")

        # Load test metadata
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata file {metadata_path}: {e}")
            raise ValueError(f"Invalid metadata file: {e}")

        test_cases = metadata.get('test_cases', [])
        if not test_cases:
            raise ValueError("No test cases found in metadata")

        logger.info(f"Loaded {len(test_cases)} test cases for evaluation")

        # Import SingingConversionPipeline here to avoid circular imports
        try:
            from ..inference.singing_conversion_pipeline import SingingConversionPipeline
        except ImportError as e:
            logger.error(f"Could not import SingingConversionPipeline: {e}")
            raise ValueError("SingingConversionPipeline required for metadata-driven evaluation")

        # Create pipeline instance (reuse for all test cases)
        pipeline = SingingConversionPipeline()

        samples = []
        failed_tests = []

        for i, test_case in enumerate(test_cases):
            try:
                self._report_progress(i, len(test_cases), f"Processing test case: {test_case.get('id', f'test_{i}')}")

                # Extract test case data
                test_id = test_case.get('id', f'test_{i}')
                source_audio_path = test_case.get('source_audio')
                target_profile_id = test_case.get('target_profile_id')
                reference_audio_path = test_case.get('reference_audio', None)

                # Validate inputs
                if not os.path.exists(source_audio_path):
                    logger.error(f"Source audio file not found: {source_audio_path}")
                    failed_tests.append(test_id)
                    continue

                if not target_profile_id:
                    logger.error(f"No target_profile_id provided for test {test_id}")
                    failed_tests.append(test_id)
                    continue

                # Load source audio
                source_audio = self._load_audio(source_audio_path)
                if source_audio is None:
                    logger.error(f"Failed to load source audio: {source_audio_path}")
                    failed_tests.append(test_id)
                    continue

                # Use pipeline to convert audio
                try:
                    conversion_result = pipeline.convert_song(
                        song_path=source_audio_path,
                        target_profile_id=target_profile_id
                    )

                    # Extract mixed_audio from result dict
                    if isinstance(conversion_result, dict):
                        converted_audio = conversion_result['mixed_audio']
                    else:
                        converted_audio = conversion_result

                    # Convert numpy array to torch.Tensor if needed
                    if isinstance(converted_audio, np.ndarray):
                        converted_audio = torch.from_numpy(converted_audio).float()
                    elif not isinstance(converted_audio, torch.Tensor):
                        logger.error(f"Unsupported audio type from pipeline: {type(converted_audio)}")
                        failed_tests.append(test_id)
                        continue

                except Exception as e:
                    logger.error(f"Pipeline conversion failed for test {test_id}: {e}")
                    failed_tests.append(test_id)
                    continue

                # Load reference audio if provided
                reference_audio = None
                if reference_audio_path:
                    reference_audio = self._load_audio(reference_audio_path)
                    if reference_audio is None:
                        logger.warning(f"Reference audio not available for test {test_id}: {reference_audio_path}")

                # Get target speaker embedding for accurate speaker similarity evaluation
                target_speaker_embedding = None
                try:
                    # Attempt to get embedding from profile storage via VoiceCloner
                    from ..inference.voice_cloner import VoiceCloner
                    voice_cloner = VoiceCloner()
                    target_speaker_embedding = voice_cloner.get_embedding(target_profile_id)
                    logger.debug(f"Retrieved target speaker embedding from profile: {target_profile_id}")
                except Exception as e:
                    logger.warning(f"Could not retrieve target embedding from profile {target_profile_id}: {e}")
                    # Fall back to reference audio if available
                    if reference_audio is not None:
                        try:
                            from ..models.speaker_encoder import SpeakerEncoder
                            encoder = SpeakerEncoder()
                            # Ensure mono audio for embedding extraction
                            ref_mono = reference_audio.mean(dim=0) if reference_audio.dim() > 1 else reference_audio
                            target_speaker_embedding = encoder.extract_embedding(ref_mono)
                            logger.debug(f"Extracted target speaker embedding from reference audio")
                        except Exception as embed_error:
                            logger.warning(f"Failed to extract embedding from reference audio: {embed_error}")

                # Create evaluation sample
                sample = EvaluationSample(
                    id=test_id,
                    source_audio_path=source_audio_path,
                    target_audio_path=None,  # Converted audio comes from pipeline
                    source_audio=source_audio,
                    target_audio=converted_audio,
                    metadata={
                        'target_profile_id': target_profile_id,
                        'reference_audio_path': reference_audio_path,
                        'conversion_params': test_case.get('conversion_params', {}),
                        'reference_audio': reference_audio
                    }
                )

                # Evaluate quality with target speaker embedding
                try:
                    result = self.evaluate_single_conversion(
                        source_audio,
                        converted_audio,
                        target_speaker_embedding=target_speaker_embedding
                    )
                    sample.result = result
                    samples.append(sample)
                except Exception as e:
                    logger.error(f"Quality evaluation failed for test {test_id}: {e}")
                    failed_tests.append(test_id)
                    continue

            except Exception as e:
                logger.error(f"Test case processing failed for test_{i}: {e}")
                failed_tests.append(f'test_{i}')
                continue

        self._report_progress(len(test_cases), len(test_cases), "Evaluation complete")

        # Compute summary statistics
        summary_stats = self._compute_batch_summary_statistics(samples)

        # Add metadata information to summary
        summary_stats['metadata_evaluation'] = {
            'total_test_cases': len(test_cases),
            'successful_evaluations': len(samples),
            'failed_test_cases': len(failed_tests),
            'failure_rate': len(failed_tests) / len(test_cases) if test_cases else 0.0
        }

        if failed_tests:
            logger.warning(f"Failed test cases: {failed_tests}")
            summary_stats['metadata_evaluation']['failed_tests'] = failed_tests

        # Create results object
        results = EvaluationResults(
            samples=samples,
            summary_stats=summary_stats,
            evaluation_config=self.config,
            evaluation_timestamp=time.time(),
            total_evaluation_time=time.time() - start_time
        )

        logger.info(f"Metadata-driven evaluation completed in {results.total_evaluation_time:.2f} seconds")
        logger.info(f"Successfully evaluated {len(samples)}/{len(test_cases)} test cases")

        # Generate report if requested
        if output_report_path:
            report_files = self.generate_reports(results, output_report_path)
            logger.info(f"Reports generated: {list(report_files.keys())}")

        return results

    def create_test_samples_from_directory(self, source_dir: str, target_dir: str) -> List[EvaluationSample]:
        """
        Create evaluation samples from directories containing source and target audio files.

        Args:
            source_dir: Directory containing source audio files
            target_dir: Directory containing target audio files

        Returns:
            List of EvaluationSample objects
        """
        samples = []
        source_dir = Path(source_dir)
        target_dir = Path(target_dir)

        if not source_dir.exists() or not target_dir.exists():
            raise ValueError("Source or target directory does not exist")

        # Match files by name (assuming paired files have the same base name)
        source_files = {f.stem: f for f in source_dir.glob('*.wav') if f.is_file()}

        for target_file in target_dir.glob('*.wav'):
            if target_file.is_file():
                sample_id = target_file.stem
                source_file = source_files.get(sample_id)

                if source_file:
                    sample = EvaluationSample(
                        id=sample_id,
                        source_audio_path=str(source_file),
                        target_audio_path=str(target_file)
                    )
                    samples.append(sample)
                else:
                    logger.warning(f"No matching source file found for {sample_id}")

        logger.info(f"Created {len(samples)} evaluation samples from directories")
        return samples
