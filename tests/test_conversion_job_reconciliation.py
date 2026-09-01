"""Conversion jobs orphaned by a restart must not stay 'in_progress' forever.

A conversion does not survive a restart - the worker thread and its fork
subprocess die with the process - but the persisted record still read
in_progress and nothing corrected it. Three such records were found in this
deployment, two stuck since mid-July, surviving every restart since.

Not merely cosmetic: the status and history endpoints report them as live
work, so a caller polling for completion waits forever and anything gating on
"is a job running" sees phantom activity. cancel_job cannot clean them up
either - it only accepts jobs in 'queued', so an in-flight job is
uncancellable by design and a killed one has no path to a terminal state.

The training manager already had this fix; the conversion manager did not.
"""
from auto_voice.web.job_manager import JobManager


class _Store:
    def __init__(self, jobs):
        self.jobs = {j['job_id']: dict(j) for j in jobs}

    def list_training_jobs(self, profile_id=None):
        return [dict(j) for j in self.jobs.values()]

    def save_training_job(self, job):
        self.jobs[job['job_id']] = dict(job)
        return job


def _manager(store):
    mgr = object.__new__(JobManager)
    mgr.state_store = store
    return mgr


def test_orphaned_in_progress_jobs_are_failed():
    store = _Store([
        {'job_id': 'a', 'profile_id': 'p1', 'status': 'in_progress'},
        {'job_id': 'b', 'profile_id': 'p1', 'status': 'queued'},
        {'job_id': 'c', 'profile_id': 'p1', 'status': 'running'},
    ])
    _manager(store)._reconcile_orphaned_jobs()
    assert [store.jobs[k]['status'] for k in ('a', 'b', 'c')] == ['failed'] * 3
    for k in ('a', 'b', 'c'):
        assert 'restart' in store.jobs[k]['error']
        assert store.jobs[k]['completed_at'] is not None


def test_terminal_jobs_are_left_alone():
    store = _Store([
        {'job_id': 'done', 'profile_id': 'p1', 'status': 'completed',
         'error': None, 'completed_at': 123.0},
        {'job_id': 'failed', 'profile_id': 'p1', 'status': 'failed',
         'error': 'original reason', 'completed_at': 456.0},
    ])
    _manager(store)._reconcile_orphaned_jobs()
    assert store.jobs['done']['status'] == 'completed'
    assert store.jobs['done']['completed_at'] == 123.0
    assert store.jobs['failed']['error'] == 'original reason'


def test_no_store_is_a_no_op():
    _manager(None)._reconcile_orphaned_jobs()   # must not raise


def test_a_broken_store_never_blocks_startup():
    class Boom:
        def list_training_jobs(self, profile_id=None):
            raise RuntimeError('state store unreadable')
    _manager(Boom())._reconcile_orphaned_jobs()  # must not raise
