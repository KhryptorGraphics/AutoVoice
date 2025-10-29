#!/usr/bin/env python3
"""
Voice Conversion Quality Evaluation Script.

This script provides a command-line interface for comprehensive quality evaluation
of singing voice conversion systems. It supports batch processing, various output
formats, and automated quality regression detection.

Usage:
    python examples/evaluate_voice_conversion.py --source-dir /path/to/source/audio \
                                                 --target-dir /path/to/converted/audio \
                                                 --output-dir ./evaluation_results \
                                                 --config config/evaluation_config.yaml

Example:
    python examples/evaluate_voice_conversion.py --source-dir data/test/source \
                                                 --target-dir data/test/converted \
                                                 --output-dir results/evaluation \
                                                 --formats markdown json html
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
import json
import warnings

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from auto_voice.evaluation.evaluator import (
    VoiceConversionEvaluator, EvaluationSample, EvaluationResults,
    QualityTargets
)
from auto_voice.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate quality of singing voice conversion results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input arguments - use either directory-based or metadata-driven evaluation
    parser.add_argument(
        '--source-dir',
        type=str,
        help='Directory containing source (original) audio files (directory-based mode)'
    )
    parser.add_argument(
        '--target-dir',
        type=str,
        help='Directory containing target (converted) audio files (directory-based mode)'
    )
    parser.add_argument(
        '--test-metadata',
        type=str,
        help='Path to JSON file containing test cases with metadata for metadata-driven evaluation'
    )

    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_results',
        help='Directory to save evaluation results (default: ./evaluation_results)'
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['markdown', 'json', 'html'],
        default=['markdown', 'json'],
        help='Output formats for evaluation reports (default: markdown json)'
    )

    # Configuration arguments
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to evaluation configuration YAML file'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=44100,
        help='Audio sample rate for processing (default: 44100)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='PyTorch device (auto, cpu, cuda, cuda:N) (default: auto)'
    )

    # Quality validation arguments
    parser.add_argument(
        '--validate-targets',
        action='store_true',
        help='Validate results against quality targets'
    )
    parser.add_argument(
        '--min-pitch-correlation',
        type=float,
        default=0.8,
        help='Minimum pitch correlation target (default: 0.8)'
    )
    parser.add_argument(
        '--max-pitch-rmse-hz',
        type=float,
        default=10.0,
        help='Maximum pitch RMSE in Hz (default: 10.0)'
    )
    # Deprecated alias - will be removed in a future version
    parser.add_argument(
        '--max-pitch-rmse',
        type=float,
        default=None,
        help=argparse.SUPPRESS  # Hide from help to discourage use
    )
    parser.add_argument(
        '--min-speaker-similarity',
        type=float,
        default=0.85,
        help='Minimum speaker similarity target (default: 0.85)'
    )

    # Processing arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for processing (default: 4)'
    )
    parser.add_argument(
        '--align-audio',
        action='store_true',
        default=True,
        help='Align source and target audio before evaluation'
    )
    parser.add_argument(
        '--no-align-audio',
        action='store_false',
        dest='align_audio',
        help='Disable audio alignment'
    )

    # Logging arguments
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )

    return parser.parse_args()


def setup_output_directory(output_dir: str) -> Path:
    """Ensure output directory exists."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def load_evaluation_config(config_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load evaluation configuration from YAML file."""
    if not config_path or not os.path.exists(config_path):
        return None

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded evaluation config from: {config_path}")
        return config
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        return None


def create_quality_targets(args) -> QualityTargets:
    """Create quality targets from command line arguments."""
    # Handle deprecated max-pitch-rmse argument
    if args.max_pitch_rmse is not None:
        warnings.warn(
            "--max-pitch-rmse is deprecated and will be removed in a future version. "
            "Use --max-pitch-rmse-hz instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if args.max_pitch_rmse_hz == 10.0:  # Only override if user hasn't set Hz explicitly
            args.max_pitch_rmse_hz = args.max_pitch_rmse

    return QualityTargets(
        min_pitch_accuracy_correlation=args.min_pitch_correlation,
        max_pitch_accuracy_rmse_hz=args.max_pitch_rmse_hz,
        min_speaker_similarity=args.min_speaker_similarity
    )


def setup_progress_callback(quiet: bool):
    """Set up progress callback for evaluation."""
    def progress_callback(current: int, total: int, message: str):
        if not quiet:
            if total > 0:
                percentage = (current / total) * 100
                print(f"Progress: {percentage:.1f}% - {message}")
            else:
                print(message)

    return progress_callback


def main():
    """Main evaluation workflow."""
    args = parse_arguments()

    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(level=log_level)

    logger.info("Starting voice conversion quality evaluation")

    # Validation: either use test-metadata OR source/target dirs
    if args.test_metadata:
        if args.source_dir or args.target_dir:
            logger.error("--test-metadata cannot be used with --source-dir or --target-dir")
            return 1
        logger.info(f"Using metadata-driven evaluation: {args.test_metadata}")
        logger.info(f"Output directory: {args.output_dir}")
    else:
        # Directory-based mode requires both source and target dirs
        if not args.source_dir or not args.target_dir:
            logger.error("Directory-based mode requires both --source-dir and --target-dir, or use --test-metadata for metadata-driven evaluation")
            return 1
        logger.info(f"Using directory-based evaluation")
        logger.info(f"Source directory: {args.source_dir}")
        logger.info(f"Target directory: {args.target_dir}")
        logger.info(f"Output directory: {args.output_dir}")

    try:
        # Setup output directory
        output_dir = setup_output_directory(args.output_dir)

        # Load configuration
        config = load_evaluation_config(args.config)

        # Create evaluator
        evaluator = VoiceConversionEvaluator(
            sample_rate=args.sample_rate,
            device=args.device,
            evaluation_config_path=args.config
        )

        # Setup progress callback
        progress_callback = setup_progress_callback(args.quiet)
        evaluator.add_progress_callback(progress_callback)

        # Choose evaluation mode
        if args.test_metadata:
            # Metadata-driven evaluation
            logger.info("Starting metadata-driven evaluation...")
            try:
                results = evaluator.evaluate_test_set(args.test_metadata, output_report_path=str(output_dir))

                if not results.samples:
                    logger.error("No successful evaluations in metadata-driven mode.")
                    return 1

                logger.info(f"Successfully evaluated {len(results.samples)} test cases")

            except Exception as e:
                logger.error(f"Metadata-driven evaluation failed: {e}")
                return 1

        else:
            # Directory-based evaluation
            # Create evaluation samples
            samples = evaluator.create_test_samples_from_directory(
                source_dir=args.source_dir,
                target_dir=args.target_dir
            )

            if not samples:
                logger.error("No evaluation samples found. Check source and target directories.")
                return 1

            logger.info(f"Created {len(samples)} evaluation samples")

            # Run evaluation
            logger.info("Starting evaluation...")
            start_time = time.time()

            results = evaluator.evaluate_conversions(samples)

            evaluation_time = time.time() - start_time
            logger.info(f"Evaluation completed in {evaluation_time:.2f}s")

        # Generate reports (if not already generated in metadata mode)
        if not args.test_metadata:
            logger.info("Generating reports...")
            report_files = evaluator.generate_reports(
                results=results,
                output_dir=output_dir,
                formats=args.formats
            )

            print(f"\nReports generated:")
            for format_name, file_path in report_files.items():
                print(f"  {format_name.upper()}: {file_path}")

        # Quality validation
        if args.validate_targets:
            logger.info("Validating against quality targets...")
            targets = create_quality_targets(args)
            validation_results = evaluator.validate_quality_targets(results, targets)

            # Save validation results
            validation_file = output_dir / 'quality_validation.json'
            with open(validation_file, 'w') as f:
                json.dump(validation_results, f, indent=2)

            # Print validation summary
            print(f"\nQuality Validation Results:")
            print(f"Overall Pass: {'✓' if validation_results['overall_pass'] else '✗'}")
            if validation_results['failed_targets']:
                print(f"Failed Targets: {', '.join(validation_results['failed_targets'])}")

            print(f"Validation details saved to: {validation_file}")

            # Exit with appropriate code (strict: fail on any target miss for CI gating)
            if not validation_results['overall_pass']:
                logger.error("Quality targets not met!")
                return 1

        # Print summary
        print(f"\nEvaluation Summary:")
        print(f"Total Samples: {len(results.samples)}")
        print(f"Evaluation Time: {results.total_evaluation_time:.2f} seconds")
        print(f"Processing Time: {results.total_evaluation_time:.2f} seconds")

        if results.summary_stats:
            overall_stats = results.summary_stats.get('overall', {})
            quality_score = overall_stats.get('quality_score', {}).get('mean', 0.0)
            print(f"Average Quality Score: {quality_score:.1f}")

        logger.info("Evaluation completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
