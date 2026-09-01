"""Every completed conversion must leave a quality datapoint behind.

Nothing recorded one automatically before: `record_metric` was reachable only
from an API endpoint nobody calls, so data/quality_history/ stayed empty and
the Quality page's rolling averages had no source. The practical consequence
was that "is this checkpoint better?" had no answer except opinion, and a whole
tuning session was judged by ear with no record of what any change did.

These pin the wire, and pin WHICH metrics it records: f0_correlation (melody
preservation, legitimate for singing) and rtf (measured). Explicitly not
speaker_similarity, which this project has a written calibration against on
sung audio, and not mcd, whose implementation has no DTW alignment.
"""
from unittest.mock import MagicMock, patch

from auto_voice.web.job_manager import JobManager


def _manager():
    mgr = object.__new__(JobManager)
    mgr._jobs = {}
    return mgr


def _job(**over):
    job = {'profile_id': 'p1', 'started_at': 100.0, 'completed_at': 130.0,
           'duration': 60.0, 'settings': {'resolved_pipeline': 'realtime'}}
    job.update(over)
    return job


METRICS = {'pitch_accuracy': {'correlation': 0.93, 'rmse_hz': 20.0}}


def test_a_completed_conversion_records_a_metric():
    mgr = _manager()
    mgr._jobs['j1'] = _job()
    monitor = MagicMock()
    with patch('auto_voice.monitoring.quality_monitor.get_quality_monitor',
               return_value=monitor):
        mgr._record_quality_metrics('j1', 'p1', METRICS, {'duration': 60.0})
    monitor.record_metric.assert_called_once()
    kwargs = monitor.record_metric.call_args.kwargs
    assert kwargs['profile_id'] == 'p1'
    assert kwargs['f0_correlation'] == 0.93
    assert kwargs['conversion_id'] == 'j1'
    assert kwargs['rtf'] == 0.5          # 30s elapsed / 60s audio


def test_discredited_metrics_are_not_recorded():
    """Recording these would build the feedback loop on metrics already known
    to be wrong for this project's audio."""
    mgr = _manager()
    mgr._jobs['j1'] = _job()
    monitor = MagicMock()
    with patch('auto_voice.monitoring.quality_monitor.get_quality_monitor',
               return_value=monitor):
        mgr._record_quality_metrics('j1', 'p1', METRICS, {'duration': 60.0})
    kwargs = monitor.record_metric.call_args.kwargs
    assert kwargs.get('speaker_similarity') is None
    assert kwargs.get('mcd') is None


def test_a_metrics_failure_never_breaks_the_conversion():
    """The user is waiting on audio; bookkeeping must not take it down."""
    mgr = _manager()
    mgr._jobs['j1'] = _job()
    monitor = MagicMock()
    monitor.record_metric.side_effect = RuntimeError('history store on fire')
    with patch('auto_voice.monitoring.quality_monitor.get_quality_monitor',
               return_value=monitor):
        mgr._record_quality_metrics('j1', 'p1', METRICS, {'duration': 60.0})  # must not raise


def test_nothing_recorded_without_a_profile():
    mgr = _manager()
    monitor = MagicMock()
    with patch('auto_voice.monitoring.quality_monitor.get_quality_monitor',
               return_value=monitor):
        mgr._record_quality_metrics('j1', None, METRICS, {'duration': 60.0})
    monitor.record_metric.assert_not_called()


def test_missing_metrics_records_nothing_rather_than_zeros():
    """A conversion with no measurable pitch must not poison the rolling
    average with a fabricated 0.0."""
    mgr = _manager()
    mgr._jobs['j1'] = _job(started_at=None, duration=None)
    monitor = MagicMock()
    with patch('auto_voice.monitoring.quality_monitor.get_quality_monitor',
               return_value=monitor):
        mgr._record_quality_metrics('j1', 'p1', {}, {})
    monitor.record_metric.assert_not_called()


def test_offline_lanes_do_not_report_rtf():
    """The monitor's rtf threshold is a realtime latency target (0.30). An
    offline studio render at rtf ~1.5 is healthy but would trip it on EVERY
    conversion, burying real alerts in permanent noise. The number still lives
    in the conversion history record."""
    mgr = _manager()
    mgr._jobs['j1'] = _job(settings={'resolved_pipeline': 'quality'})
    monitor = MagicMock()
    with patch('auto_voice.monitoring.quality_monitor.get_quality_monitor',
               return_value=monitor):
        mgr._record_quality_metrics('j1', 'p1', METRICS, {'duration': 60.0})
    kwargs = monitor.record_metric.call_args.kwargs
    assert kwargs['rtf'] is None
    assert kwargs['f0_correlation'] == 0.93   # still recorded


def test_realtime_lane_does_report_rtf():
    mgr = _manager()
    mgr._jobs['j1'] = _job(settings={'resolved_pipeline': 'realtime_meanvc'})
    monitor = MagicMock()
    with patch('auto_voice.monitoring.quality_monitor.get_quality_monitor',
               return_value=monitor):
        mgr._record_quality_metrics('j1', 'p1', METRICS, {'duration': 60.0})
    assert monitor.record_metric.call_args.kwargs['rtf'] == 0.5
