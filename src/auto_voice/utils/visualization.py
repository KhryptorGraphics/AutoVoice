"""
Visualization utilities for quality evaluation and results display.

Provides plotting functions for pitch contours, spectrograms, and quality metrics.
"""

import base64
import io
import os
import numpy as np
import torch
from typing import Optional, Dict, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass


try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for server environments
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, visualization features disabled")


@dataclass
class PitchContourData:
    """Container for pitch contour data."""
    f0: np.ndarray  # Fundamental frequency values (Hz)
    times: Optional[np.ndarray] = None  # Time points (seconds)
    confidence: Optional[np.ndarray] = None  # Confidence scores (0-1)


class PitchContourVisualizer:
    """Visualization class for pitch contour comparisons."""

    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        """Initialize the pitch contour visualizer.

        Args:
            figsize: Figure size as (width, height) in inches
        """
        self.figsize = figsize

    def plot_pitch_contour_comparison(
        self,
        source_pitch: PitchContourData,
        target_pitch: PitchContourData,
        output_path: Optional[str] = None,
        title: str = "Pitch Contour Comparison"
    ) -> Optional[plt.Figure]:
        """
        Plot comparative pitch contours for source and target audio.

        Args:
            source_pitch: Pitch contour data for source audio
            target_pitch: Pitch contour data for target audio
            output_path: Path to save the plot (optional)
            title: Plot title

        Returns:
            matplotlib Figure object if matplotlib is available, None otherwise
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available, skipping pitch contour visualization")
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)

        # Plot source pitch contour
        self._plot_single_pitch_contour(
            ax1, source_pitch, 'Source', 'blue', title + " - Source"
        )

        # Plot target pitch contour
        self._plot_single_pitch_contour(
            ax2, target_pitch, 'Target', 'red', title + " - Target"
        )

        # Overall title and layout
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # Save if path provided
        if output_path:
            self._save_plot(fig, output_path)

        return fig

    def _plot_single_pitch_contour(
        self,
        ax: plt.Axes,
        pitch_data: PitchContourData,
        label: str,
        color: str,
        subplot_title: str
    ):
        """Plot a single pitch contour."""
        # Handle time axis
        if pitch_data.times is not None:
            times = pitch_data.times
            xlabel = "Time (seconds)"
        else:
            # Fake time axis based on array length
            times = np.arange(len(pitch_data.f0)) * 0.01  # Assume 10ms hops
            xlabel = "Frame Index"

        # Plot pitch contour
        ax.plot(times, pitch_data.f0, color=color, linewidth=2, alpha=0.8, label=label)

        # Fill voiced regions if confidence available
        if pitch_data.confidence is not None:
            # Color based on confidence
            norm_conf = pitch_data.confidence / max(pitch_data.confidence.max(), 1.0)
            colors = plt.cm.viridis(norm_conf)
            ax.scatter(times, pitch_data.f0, c=colors[:, :3], alpha=0.6, s=1)

        # Styling
        ax.set_title(subplot_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency (Hz)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0, top=min(1000, pitch_data.f0.max() + 50) if len(pitch_data.f0) > 0 else 1000)
        ax.legend()

    def _save_plot(self, fig: plt.Figure, output_path: str):
        """Save plot to file with high quality."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(str(output_path), dpi=300, bbox_inches='tight', format='png')
        plt.close(fig)  # Free memory


class SpectrogramVisualizer:
    """Visualization class for spectrogram comparisons."""

    def __init__(self, figsize: Tuple[int, int] = (16, 8)):
        """Initialize the spectrogram visualizer.

        Args:
            figsize: Figure size as (width, height) in inches
        """
        self.figsize = figsize

    def plot_spectrogram_comparison(
        self,
        source_audio: Union[np.ndarray, torch.Tensor],
        target_audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 22050,
        output_path: Optional[str] = None,
        title: str = "Spectrogram Comparison",
        n_fft: int = 2048,
        hop_length: int = 512
    ) -> Optional[plt.Figure]:
        """
        Plot comparative spectrograms for source and target audio.

        Args:
            source_audio: Source audio waveform
            target_audio: Target audio waveform
            sample_rate: Audio sample rate
            output_path: Path to save the plot (optional)
            title: Plot title
            n_fft: FFT window size
            hop_length: Hop length for STFT

        Returns:
            matplotlib Figure object if matplotlib is available, None otherwise
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available, skipping spectrogram visualization")
            return None

        # Convert tensor to numpy if needed
        if isinstance(source_audio, torch.Tensor):
            source_audio = source_audio.cpu().numpy().flatten()
        if isinstance(target_audio, torch.Tensor):
            target_audio = target_audio.cpu().numpy().flatten()

        # Ensure mono audio
        if source_audio.ndim > 1:
            source_audio = source_audio.mean(axis=0 if source_audio.shape[0] > 1 else -1)
        if target_audio.ndim > 1:
            target_audio = target_audio.mean(axis=0 if target_audio.shape[0] > 1 else -1)

        # Compute spectrograms
        source_spec, target_spec, freqs, times = self._compute_spectrograms(
            source_audio, target_audio, sample_rate, n_fft, hop_length
        )

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)

        # Plot source spectrogram
        im1 = ax1.pcolormesh(times, freqs, 10 * np.log10(source_spec + 1e-10),
                            shading='gouraud', cmap='viridis')
        ax1.set_title(f"{title} - Source Audio", fontsize=14)
        ax1.set_ylabel("Frequency (Hz)")
        ax1.set_xlim([0, times[-1]])
        ax1.set_ylim([0, min(8000, freqs[-1])])  # Show up to 8kHz

        # Plot target spectrogram
        im2 = ax2.pcolormesh(times, freqs, 10 * np.log10(target_spec + 1e-10),
                            shading='gouraud', cmap='viridis')
        ax2.set_title(f"{title} - Target Audio", fontsize=14)
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_xlim([0, times[-1]])
        ax2.set_ylim([0, min(8000, freqs[-1])])  # Show up to 8kHz

        # Colorbar for both plots
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im1, cax=cbar_ax, label='Magnitude (dB)')

        # Overall title and layout
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.92, 0.95])

        # Save if path provided
        if output_path:
            self._save_plot(fig, output_path)

        return fig

    def _compute_spectrograms(self, source_audio, target_audio, sample_rate, n_fft, hop_length):
        """Compute STFT spectrograms for both audio signals."""
        try:
            import librosa

            # Compute STFT
            source_stft = librosa.stft(source_audio, n_fft=n_fft, hop_length=hop_length)
            target_stft = librosa.stft(target_audio, n_fft=n_fft, hop_length=hop_length)

            # Get magnitude spectrograms
            source_spec = np.abs(source_stft) ** 2
            target_spec = np.abs(target_stft) ** 2

            # Frequency and time axes
            freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
            times = librosa.times_like(source_stft, sr=sample_rate, hop_length=hop_length)

            return source_spec, target_spec, freqs, times

        except ImportError:
            # Fallback implementation using numpy
            print("librosa not available, using basic STFT implementation")

            def basic_stft(audio, n_fft, hop_length):
                """Basic STFT implementation."""
                n_frames = 1 + (len(audio) - n_fft) // hop_length
                spec = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)

                for i in range(n_frames):
                    start = i * hop_length
                    end = start + n_fft
                    frame = audio[start:end]

                    # Apply window
                    window = np.hanning(n_fft)
                    frame = frame * window

                    # FFT
                    spec[:, i] = np.fft.rfft(frame, n=n_fft)

                return spec

            source_stft = basic_stft(source_audio, n_fft, hop_length)
            target_stft = basic_stft(target_audio, n_fft, hop_length)

            source_spec = np.abs(source_stft) ** 2
            target_spec = np.abs(target_stft) ** 2

            # Simple frequency and time axes
            freqs = np.linspace(0, sample_rate // 2, source_spec.shape[0])
            frame_times = np.arange(source_spec.shape[1]) * hop_length / sample_rate

            return source_spec, target_spec, freqs, frame_times

    def _save_plot(self, fig: plt.Figure, output_path: str):
        """Save plot to file with high quality."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(str(output_path), dpi=300, bbox_inches='tight', format='png')
        plt.close(fig)  # Free memory


class QualityMetricsVisualizer:
    """Visualization class for quality metrics and summary statistics."""

    def __init__(self, figsize: Tuple[int, int] = (14, 10)):
        """Initialize the quality metrics visualizer.

        Args:
            figsize: Figure size as (width, height) in inches
        """
        self.figsize = figsize

    def create_summary_dashboard(
        self,
        summary_stats: Dict[str, Any],
        output_path: Optional[str] = None,
        title: str = "Voice Conversion Quality Dashboard"
    ) -> Optional[plt.Figure]:
        """
        Create a comprehensive quality metrics dashboard.

        Args:
            summary_stats: Summary statistics from evaluation
            output_path: Path to save the plot (optional)
            title: Dashboard title

        Returns:
            matplotlib Figure object if matplotlib is available, None otherwise
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available, skipping quality dashboard visualization")
            return None

        fig, axes = plt.subplots(2, 2, figsize=self.figsize)

        # Extract key metrics
        pitch_stats = summary_stats.get('pitch_accuracy', {})
        speaker_stats = summary_stats.get('speaker_similarity', {})
        naturalness_stats = summary_stats.get('naturalness', {})

        # Plot 1: Pitch Accuracy Metrics
        self._plot_metric_subplot(
            axes[0, 0],
            pitch_stats,
            ['rmse_hz', 'correlation'],
            "Pitch Accuracy",
            ['RMSE (Hz)', 'Correlation']
        )

        # Plot 2: Speaker Similarity Metrics
        self._plot_metric_subplot(
            axes[0, 1],
            speaker_stats,
            ['cosine_similarity', 'embedding_distance'],
            "Speaker Similarity",
            ['Cosine Similarity', 'Embedding Distance']
        )

        # Plot 3: Naturalness Metrics
        self._plot_metric_subplot(
            axes[1, 0],
            naturalness_stats,
            ['spectral_distortion', 'mos_estimation'],
            "Naturalness",
            ['Spectral Distortion (dB)', 'MOS Estimation']
        )

        # Plot 4: Overall Performance Summary
        self._plot_overall_summary(axes[1, 1], summary_stats)

        # Overall formatting
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save if path provided
        if output_path:
            self._save_plot(fig, output_path)

        return fig

    def _plot_metric_subplot(self, ax: plt.Axes, metric_data: Dict[str, Any],
                           metric_keys: list, title: str, labels: list):
        """Plot a single metric subplot with means and confidence intervals."""
        x_pos = np.arange(len(metric_keys))

        for i, key in enumerate(metric_keys):
            if key in metric_data:
                stats = metric_data[key]
                mean = stats.get('mean', 0)
                std = stats.get('std', 0)

                # Plot mean with error bars
                ax.bar(i, mean, yerr=std, capsize=5, alpha=0.7,
                      label=labels[i] if len(labels) > i else key)

        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels[:len(metric_keys)])
        ax.grid(True, alpha=0.3)

    def _plot_overall_summary(self, ax: plt.Axes, summary_stats: Dict[str, Any]):
        """Plot overall quality summary."""
        # Compute overall quality scores distribution
        quality_scores = []

        # This would typically analyze individual sample scores
        # For now, show a simple summary

        summary_text = "Quality Summary:\n\n"
        if 'metadata_evaluation' in summary_stats:
            meta = summary_stats['metadata_evaluation']
            summary_text += f"Total Tests: {meta.get('total_test_cases', 0)}\n"
            total_tests = max(meta.get('total_test_cases', 1), 1)
            success_rate = (meta.get('successful_evaluations', 0) / total_tests) * 100
            summary_text += f"Success Rate: {success_rate:.1f}%\n"

        ax.text(0.1, 0.8, summary_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_title("Evaluation Summary")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def _save_plot(self, fig: plt.Figure, output_path: str):
        """Save plot to file with high quality."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(str(output_path), dpi=300, bbox_inches='tight', format='png')
        plt.close(fig)  # Free memory


# Utility functions for report generation
def encode_plot_as_base64(fig: plt.Figure) -> str:
    """
    Encode a matplotlib figure as base64 string for embedding in HTML.

    Args:
        fig: matplotlib Figure object

    Returns:
        Base64 encoded PNG string
    """
    if not HAS_MATPLOTLIB:
        return ""

    # Save to bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)

    # Encode as base64
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return f"data:image/png;base64,{img_data}"


def create_embedded_markdown_image(image_path: str, alt_text: str = "Quality Plot") -> str:
    """
    Create markdown image tag with embedded base64 image.

    Args:
        image_path: Path to image file
        alt_text: Alt text for the image

    Returns:
        Markdown image string with embedded data
    """
    if not os.path.exists(image_path):
        return f"![{alt_text}]({image_path})"

    try:
        with open(image_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')

        ext = Path(image_path).suffix.lower().replace('.', '')
        return f"![{alt_text}](data:image/{ext};base64,{img_data})"
    except Exception:
        # Fall back to regular image link
        return f"![{alt_text}]({image_path})"
