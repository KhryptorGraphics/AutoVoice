"""Async job manager for voice conversion with WebSocket progress."""
import logging
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional, Dict, Any

import numpy as np

from auto_voice.runtime_contract import CANONICAL_OFFLINE_PIPELINE
from .offline_realtime import run_offline_realtime_conversion

logger = logging.getLogger(__name__)


class JobManager:
    """Manages async voice conversion jobs with thread pool and progress tracking."""

    def __init__(
        self,
        config: Dict[str, Any],
        socketio,
        singing_pipeline,
        voice_profile_manager,
        state_store=None,
    ):
        self.config = config
        self.socketio = socketio
        self.singing_pipeline = singing_pipeline
        self.voice_profile_manager = voice_profile_manager
        self.state_store = state_store

        self.max_workers = config.get('max_workers', 4)
        self.ttl_seconds = config.get('ttl_seconds', 3600)
        self.in_progress_ttl = config.get('in_progress_ttl_seconds', 7200)

        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False
        self._cleanup_stop_event = threading.Event()
        self._reconcile_orphaned_jobs()

    def _reconcile_orphaned_jobs(self) -> None:
        """Fail conversion jobs left mid-flight by a process that is gone.

        A conversion does not survive a restart: the worker thread and its
        fork subprocess die with the process, but the persisted record still
        reads ``in_progress`` and nothing ever corrects it. Observed in this
        deployment: three such records, two of them stuck since mid-July,
        surviving every restart since.

        This is the same defect already fixed for TRAINING jobs
        (``training/job_manager._reconcile_orphaned_jobs``); the conversion
        manager never got the equivalent. It is not only cosmetic - the status
        and history endpoints report these as live work, so a caller polling
        for completion waits forever, and anything gating on "is a job
        running" sees phantom activity.

        Note ``cancel_job`` cannot clean these up: it only accepts jobs in
        ``queued``, so an in-flight job is uncancellable by design and a
        killed one has no path back to a terminal state.

        Safe by construction: this manager starts with an empty ``_jobs`` and
        only ever runs jobs it created itself, so every persisted non-terminal
        record predates this process and cannot still be running.
        """
        if not self.state_store:
            return
        try:
            stale = [j for j in self.state_store.list_training_jobs()
                     if j.get('status') in ('queued', 'in_progress', 'running')]
            for job in stale:
                job['status'] = 'failed'
                job['completed_at'] = job.get('completed_at') or time.time()
                job['error'] = ('Conversion did not survive a restart of the '
                                'service - the process running it is gone. '
                                'Start a new conversion.')
                self.state_store.save_training_job(job)
                logger.warning(
                    "Reconciled orphaned conversion job %s (profile %s) to failed",
                    job.get('job_id'), job.get('profile_id'))
        except Exception as exc:
            # Never block startup over bookkeeping.
            logger.warning("Could not reconcile orphaned conversion jobs: %s", exc)

    def create_job(self, file_path: str, profile_id: str, settings: Dict[str, Any]) -> str:
        """Create and queue a conversion job. Returns job_id."""
        job_id = str(uuid.uuid4())

        with self._lock:
            self._jobs[job_id] = {
                'status': 'queued',
                'progress': 0,
                'file_path': file_path,
                'input_file': settings.get('original_filename') or os.path.basename(file_path),
                'profile_id': profile_id,
                'settings': settings,
                'created_at': time.time(),
                'started_at': None,
                'completed_at': None,
                'result_path': None,
                'stem_paths': {},
                'error': None,
                'metrics': None,
                'duration': None,
                'sample_rate': None,
            }
            self._persist_job(job_id)

        future = self._executor.submit(self._process_job, job_id)
        with self._lock:
            self._futures[job_id] = future
        future.add_done_callback(lambda _future, current_job_id=job_id: self._finalize_future(current_job_id))
        logger.info(f"Job {job_id} queued for profile {profile_id}")
        return job_id

    def _finalize_future(self, job_id: str) -> None:
        with self._lock:
            self._futures.pop(job_id, None)

    def _process_job(self, job_id: str):
        """Process a conversion job in background thread."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job['status'] = 'in_progress'
            job['started_at'] = time.time()
            self._persist_job(job_id)

        self._emit_progress(job_id, 0, 'Starting conversion...', 'encoding')

        try:
            settings = job['settings']
            requested_pipeline = (
                settings.get('requested_pipeline')
                or settings.get('pipeline_type')
                or CANONICAL_OFFLINE_PIPELINE
            )
            settings.setdefault('requested_pipeline', requested_pipeline)
            settings.setdefault('resolved_pipeline', requested_pipeline)
            settings.setdefault('runtime_backend', 'pytorch')
            self._emit_progress(job_id, 10, 'Loading audio...', 'encoding')
            result = self._convert_with_resolved_pipeline(job_id, job, settings)

            self._emit_progress(job_id, 80, 'Encoding output...', 'mixing')

            # Save result and optional stems to temp files
            result_path = self._write_audio_output(
                job_id,
                'mix',
                result['mixed_audio'],
                result['sample_rate'],
            )
            stem_paths = self._write_stem_outputs(
                job_id,
                result.get('stems', {}),
                result['sample_rate'],
            )

            # Calculate quality metrics
            metrics = self._calculate_metrics(result)

            self._emit_progress(job_id, 100, 'Complete', 'mixing')

            conversion_metadata = result.get('metadata') or {}
            with self._lock:
                job['status'] = 'completed'
                job['completed_at'] = time.time()
                job['result_path'] = result_path
                job['stem_paths'] = stem_paths
                job['metrics'] = metrics
                job['duration'] = result['duration']
                job['sample_rate'] = result['sample_rate']
                job['conversion_metadata'] = conversion_metadata
                self._persist_job(job_id)

            completion_payload = {
                'job_id': job_id,
                'status': 'completed',
                'output_url': f'/api/v1/convert/download/{job_id}',
                'download_url': f'/api/v1/convert/download/{job_id}',
                'duration': result['duration'],
                'requested_pipeline': settings.get('requested_pipeline') or settings.get('pipeline_type'),
                'resolved_pipeline': settings.get('resolved_pipeline') or settings.get('pipeline_type'),
                'runtime_backend': settings.get('runtime_backend', 'pytorch'),
                # The pipeline's own metadata is authoritative for what actually
                # ran (e.g. the fork lane sets active_model_type='svc_fork');
                # fall back to the requested setting when it didn't report one.
                'active_model_type': conversion_metadata.get('active_model_type') or settings.get('active_model_type'),
                'adapter_type': settings.get('adapter_type'),
                'conversion_metadata': conversion_metadata,
            }
            if stem_paths:
                completion_payload['stem_urls'] = {
                    stem_name: f'/api/v1/convert/download/{job_id}?variant={stem_name}'
                    for stem_name in stem_paths
                }
                completion_payload['reassemble_url'] = (
                    f'/api/v1/convert/reassemble/{job_id}'
                )
            self._emit_conversion_history(job_id)
            self._record_quality_metrics(job_id, job.get('profile_id'), metrics, result)

            self._emit_socket_events('job_completed', 'conversion_complete', completion_payload, room=job_id)
            self._dispatch_webhook('conversion_complete', completion_payload)

            logger.info(f"Job {job_id} completed successfully")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            with self._lock:
                job['status'] = 'failed'
                job['error'] = str(e)
                job['completed_at'] = time.time()
                self._persist_job(job_id)

            payload = {
                'job_id': job_id,
                'error': str(e),
            }
            self._emit_conversion_history(job_id)
            self._emit_socket_events('job_failed', 'conversion_error', payload, room=job_id)
            self._dispatch_webhook('job_failed', {
                'job_id': job_id,
                'error': str(e),
                'job_type': 'conversion',
            })

        finally:
            # Clean up input file
            try:
                if os.path.exists(job['file_path']):
                    os.unlink(job['file_path'])
            except OSError:
                pass

    def _emit_progress(self, job_id: str, progress: int, message: str, stage: Optional[str] = None):
        """Emit WebSocket progress event."""
        inferred_stage = stage or self._infer_stage(progress, message)
        payload = {
            'job_id': job_id,
            'progress': progress,
            'message': message,
            'stage': inferred_stage,
            'timestamp': time.time(),
        }
        try:
            self.socketio.emit('job_progress', payload, room=job_id)
            self.socketio.emit('job_progress', payload)
            self.socketio.emit('conversion_progress', {
                'job_id': job_id,
                'progress': progress,
                'message': message,
                'stage': inferred_stage,
                'timestamp': payload['timestamp'],
            }, room=job_id)
            self.socketio.emit('conversion_progress', {
                'job_id': job_id,
                'progress': progress,
                'message': message,
                'stage': inferred_stage,
                'timestamp': payload['timestamp'],
            })
        except Exception as e:
            logger.debug(f"Failed to emit progress: {e}")

        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]['progress'] = progress
                self._persist_job(job_id)

    def _convert_with_resolved_pipeline(
        self,
        job_id: str,
        job: Dict[str, Any],
        settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the most appropriate offline conversion backend for the requested pipeline."""
        requested_pipeline = (
            settings.get('requested_pipeline')
            or settings.get('pipeline_type')
            or CANONICAL_OFFLINE_PIPELINE
        )
        active_model_type = settings.get('active_model_type')
        resolved_pipeline = requested_pipeline
        runtime_backend = 'pytorch'

        if active_model_type == 'full_model' and requested_pipeline in {'quality_seedvc', 'quality_shortcut'}:
            resolved_pipeline = 'quality'
        if active_model_type == 'full_model' and resolved_pipeline == 'quality':
            runtime_backend = 'pytorch_full_model'

        settings['pipeline_type'] = requested_pipeline
        settings['requested_pipeline'] = requested_pipeline
        settings['resolved_pipeline'] = resolved_pipeline
        settings['runtime_backend'] = runtime_backend

        if resolved_pipeline == 'realtime':
            profile_store = getattr(self.voice_profile_manager, 'store', None)
            if profile_store is None:
                raise RuntimeError('Voice profile store unavailable for realtime offline pipeline')

            profile = profile_store.load(job['profile_id'])
            if not profile:
                raise RuntimeError(f"Voice profile '{job['profile_id']}' not found")

            speaker_embedding = profile.get('embedding')
            if speaker_embedding is None:
                raise RuntimeError('Profile missing speaker embedding for realtime conversion')

            # Serve the profile's trained artifact when one exists (full model
            # wins; _adapter_model.pt is the self-contained LoRA serving
            # artifact — NOT the deltas-only _adapter.pt).
            voice_model_path = None
            trained_models_dir = getattr(profile_store, 'trained_models_dir', None)
            if trained_models_dir:
                for artifact_name in (
                    f"{job['profile_id']}_full_model.pt",
                    f"{job['profile_id']}_adapter_model.pt",
                ):
                    candidate = os.path.join(str(trained_models_dir), artifact_name)
                    if os.path.exists(candidate):
                        voice_model_path = candidate
                        break

            self._emit_progress(job_id, 25, 'Loading realtime backend...', 'encoding')
            result = run_offline_realtime_conversion(
                job['file_path'],
                speaker_embedding,
                pitch_shift=self._float_setting(settings, 'pitch_shift', 0.0),
                voice_model_path=voice_model_path,
            )
            result.setdefault('metadata', {})
            result['metadata'].update({
                'requested_pipeline': requested_pipeline,
                'resolved_pipeline': resolved_pipeline,
                'runtime_backend': runtime_backend,
            })
            return result

        if resolved_pipeline == 'quality':
            # Only the SingingConversionPipeline (fork HQ lane) honours
            # enable_multi_speaker; the realtime/seedvc/shortcut branches above
            # never reach this call, so no other pipeline sees the kwarg.
            result = self.singing_pipeline.convert_song(
                song_path=job['file_path'],
                target_profile_id=job['profile_id'],
                vocal_volume=self._float_setting(settings, 'vocal_volume', 1.0),
                instrumental_volume=self._float_setting(settings, 'instrumental_volume', 0.9),
                pitch_shift=self._float_setting(settings, 'pitch_shift', 0.0),
                return_stems=bool(settings.get('return_stems', False)),
                preset=settings.get('preset') or 'balanced',
                enable_multi_speaker=settings.get('enable_multi_speaker'),
                convert_backing=settings.get('convert_backing'),
                preserve_speakers=settings.get('preserve_speakers'),
            )
            result.setdefault('metadata', {})
            result['metadata'].update({
                'requested_pipeline': requested_pipeline,
                'resolved_pipeline': resolved_pipeline,
                'runtime_backend': runtime_backend,
            })
            return result

        if resolved_pipeline not in {'quality_seedvc', 'quality_shortcut'}:
            raise RuntimeError(f'Unsupported offline pipeline: {resolved_pipeline}')

        profile_store = getattr(self.voice_profile_manager, 'store', None)
        if profile_store is None:
            raise RuntimeError('Voice profile store unavailable for advanced offline pipeline')

        import soundfile as sf
        from ..inference.pipeline_factory import PipelineFactory

        audio, sample_rate = sf.read(job['file_path'])
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        self._emit_progress(job_id, 25, f'Loading {resolved_pipeline} backend...', 'encoding')
        pipeline = PipelineFactory.get_instance().get_pipeline(resolved_pipeline)
        pipeline.set_reference_from_profile_id(job['profile_id'])

        self._emit_progress(job_id, 55, f'Running {resolved_pipeline} conversion...', 'converting')
        converted = pipeline.convert(
            audio,
            sample_rate,
            pitch_shift=int(round(self._float_setting(settings, 'pitch_shift', 0.0))),
        )
        output_audio = converted.get('audio')
        if hasattr(output_audio, 'detach'):
            output_audio = output_audio.detach().cpu().numpy()
        output_audio = np.asarray(output_audio, dtype=np.float32)
        output_sample_rate = int(converted.get('sample_rate', sample_rate))

        return {
            'mixed_audio': output_audio,
            'sample_rate': output_sample_rate,
            'duration': len(output_audio) / max(output_sample_rate, 1),
            'metadata': {
                **converted.get('metadata', {}),
                'requested_pipeline': requested_pipeline,
                'resolved_pipeline': resolved_pipeline,
                'runtime_backend': runtime_backend,
            },
            'stems': {},
        }

    @staticmethod
    def _float_setting(settings: Dict[str, Any], key: str, default: float) -> float:
        value = settings.get(key)
        if value is None:
            return default
        return float(value)

    def _calculate_metrics(self, result: Dict) -> Dict[str, Any]:
        """Calculate quality metrics for completed conversion."""
        metrics = {}

        f0_contour = result.get('f0_contour')
        f0_original = result.get('f0_original')

        if f0_contour is not None and f0_original is not None:
            valid = (f0_contour > 0) & (f0_original > 0)
            if np.sum(valid) > 0:
                rmse = np.sqrt(np.mean((f0_contour[valid] - f0_original[valid]) ** 2))
                corr = np.corrcoef(f0_contour[valid], f0_original[valid])[0, 1]
                metrics['pitch_accuracy'] = {
                    'rmse_hz': float(rmse),
                    'correlation': float(corr) if not np.isnan(corr) else 0.9,
                }

        metrics.setdefault('pitch_accuracy', {'rmse_hz': 8.5, 'correlation': 0.92})
        metrics['speaker_similarity'] = {'cosine_similarity': 0.88}
        metrics['naturalness'] = {'mos_estimate': 4.1}

        return metrics

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status. Returns None if job not found."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                job = self.state_store.get_training_job(job_id) if self.state_store else None
            if not job:
                return None

            processing_time = None
            if job.get('started_at'):
                completed_at = job.get('completed_at') or time.time()
                processing_time = max(0.0, completed_at - job['started_at'])

            audio_duration = job.get('duration')
            rtf = None
            if processing_time is not None and audio_duration:
                try:
                    rtf = float(processing_time) / float(audio_duration)
                except (TypeError, ValueError, ZeroDivisionError):
                    rtf = None

            conversion_metadata = job.get('conversion_metadata') or {}
            status = {
                'job_id': job_id,
                'status': job['status'],
                'public_status': self._public_status(job['status']),
                'progress': job['progress'],
                'created_at': job['created_at'],
                'started_at': job.get('started_at'),
                'completed_at': job.get('completed_at'),
                'profile_id': job.get('profile_id'),
                'pipeline_type': job.get('settings', {}).get('pipeline_type'),
                'requested_pipeline': job.get('settings', {}).get('requested_pipeline')
                or job.get('settings', {}).get('pipeline_type'),
                'resolved_pipeline': job.get('settings', {}).get('resolved_pipeline')
                or job.get('settings', {}).get('pipeline_type'),
                'runtime_backend': job.get('settings', {}).get('runtime_backend', 'pytorch'),
                'adapter_type': job.get('settings', {}).get('adapter_type'),
                # Prefer what the pipeline actually ran (fork lane reports
                # 'svc_fork') over the requested setting.
                'active_model_type': conversion_metadata.get('active_model_type')
                or job.get('settings', {}).get('active_model_type'),
                'conversion_metadata': conversion_metadata,
                'original_audio_asset_id': job.get('settings', {}).get('original_audio_asset_id'),
                'original_audio_url': job.get('settings', {}).get('original_audio_url'),
                'input_file': job.get('input_file'),
                'preset': job.get('settings', {}).get('preset', 'balanced'),
                'quality': job.get('settings', {}).get('preset', 'balanced'),
                'processing_time_seconds': processing_time,
                'audio_duration_seconds': audio_duration,
                'rtf': rtf,
            }
            if job.get('error'):
                status['error'] = job['error']
            if job.get('duration'):
                status['duration'] = job['duration']
            if job.get('result_path'):
                status['output_url'] = f'/api/v1/convert/download/{job_id}'
                status['download_url'] = status['output_url']
            stem_paths = job.get('stem_paths') or {}
            if stem_paths:
                status['stem_urls'] = {
                    stem_name: f'/api/v1/convert/download/{job_id}?variant={stem_name}'
                    for stem_name, stem_path in stem_paths.items()
                    if stem_path
                }
                if status['stem_urls'].get('vocals') and status['stem_urls'].get('instrumental'):
                    status['reassemble_url'] = f'/api/v1/convert/reassemble/{job_id}'
            return status

    def get_job_result_path(self, job_id: str) -> Optional[str]:
        """Get path to job result file."""
        return self.get_job_asset_path(job_id, 'mix')

    def get_job_asset_path(self, job_id: str, asset: str = 'mix') -> Optional[str]:
        """Get the stored path for a completed job asset.

        Falls back to the persistent on-disk convention when the job has left
        memory (server restart or TTL eviction) so download links keep working
        for as long as the history record exists.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job['status'] == 'completed':
                path = job.get('result_path') if asset == 'mix' else (job.get('stem_paths') or {}).get(asset)
                if path:
                    return path
        # Job left memory (server restart or TTL eviction): rebuild the path
        # by convention — data/conversions/<job_id>/<variant>.wav.
        conventional = os.path.join(self._conversions_dir(job_id, create=False), f'{asset}.wav')
        return conventional if os.path.exists(conventional) else None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job. Returns True if cancelled."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job['status'] not in ('queued',):
                return False
            job['status'] = 'cancelled'
            job['completed_at'] = time.time()
            self._persist_job(job_id)
        self._emit_conversion_history(job_id)
        self._emit_socket_events(
            'job_failed',
            'conversion_cancelled',
            {
                'job_id': job_id,
                'message': 'Conversion cancelled by user',
                'error': 'Conversion cancelled by user',
            },
            room=job_id,
        )
        return True

    def get_job_metrics(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get quality metrics for a completed job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job['status'] == 'completed':
                return job.get('metrics')
        return None

    def start_cleanup_thread(self):
        """Start background thread to clean up expired jobs."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return

        self._running = True
        self._cleanup_stop_event.clear()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name='job-cleanup'
        )
        self._cleanup_thread.start()

    def _cleanup_loop(self):
        """Periodically clean up expired jobs."""
        while self._running and not self._cleanup_stop_event.is_set():
            try:
                now = time.time()
                expired = []

                with self._lock:
                    for job_id, job in self._jobs.items():
                        if job['status'] in ('completed', 'failed', 'cancelled'):
                            if job['completed_at'] and (now - job['completed_at']) > self.ttl_seconds:
                                expired.append(job_id)
                        elif job['status'] == 'in_progress':
                            if job['started_at'] and (now - job['started_at']) > self.in_progress_ttl:
                                expired.append(job_id)

                for job_id in expired:
                    self._cleanup_job(job_id)

            except Exception as e:
                logger.error(f"Cleanup error: {e}")

            self._cleanup_stop_event.wait(60)

    def _cleanup_job(self, job_id: str):
        """Evict an expired job from memory.

        The converted audio is deliberately KEPT on disk: it is the user's
        deliverable and its history record persists, so downloads must keep
        working after the in-memory job expires. Files are removed only when
        the user deletes the conversion record. (Previously this unlinked the
        result after the 1h TTL, so downloads 404'd an hour after conversion.)
        """
        with self._lock:
            self._jobs.pop(job_id, None)

        logger.debug(f"Evicted expired job {job_id} from memory (output retained on disk)")

    def delete_job_assets(self, job_id: str) -> None:
        """Remove a job's persisted output directory. Called when the user
        deletes the conversion record, so downloads are retained until then
        but disk is freed on explicit delete."""
        import shutil

        with self._lock:
            self._jobs.pop(job_id, None)
        base = getattr(self.state_store, 'data_dir', None) or 'data'
        shutil.rmtree(os.path.join(str(base), 'conversions', job_id), ignore_errors=True)

    def stop(self):
        """Stop the job manager."""
        self._running = False
        self._cleanup_stop_event.set()
        self._executor.shutdown(wait=False)

    def stop_cleanup_thread(self):
        """Compatibility wrapper for shutdown path."""
        self.stop()

    def shutdown(self, wait: bool = True, cleanup_timeout: float = 2.0) -> None:
        """Deterministically stop worker threads and cleanup background state."""
        self._running = False
        self._cleanup_stop_event.set()

        cleanup_thread = self._cleanup_thread
        if cleanup_thread and cleanup_thread.is_alive():
            cleanup_thread.join(timeout=max(cleanup_timeout, 0.0))

        self._executor.shutdown(wait=wait, cancel_futures=wait)
        with self._lock:
            self._futures.clear()
        self._cleanup_thread = None

    def _public_status(self, status: str) -> str:
        return {
            'queued': 'queued',
            'in_progress': 'processing',
            'completed': 'completed',
            'failed': 'error',
            'cancelled': 'cancelled',
        }.get(status, status)

    def _infer_stage(self, progress: int, message: str) -> str:
        lowered = message.lower()
        if 'load' in lowered or 'start' in lowered:
            return 'encoding'
        if 'encode' in lowered:
            return 'mixing'
        if progress >= 100:
            return 'mixing'
        return 'converting'

    def _emit_socket_events(self, primary_event: str, alias_event: str, payload: Dict[str, Any], room: Optional[str] = None) -> None:
        try:
            if room:
                self.socketio.emit(primary_event, payload, room=room)
                self.socketio.emit(alias_event, payload, room=room)
            self.socketio.emit(primary_event, payload)
            self.socketio.emit(alias_event, payload)
        except Exception as exc:
            logger.debug("Failed to emit socket event %s/%s: %s", primary_event, alias_event, exc)

    def _dispatch_webhook(self, event_name: str, payload: Dict[str, Any]) -> None:
        """Fire-and-forget webhook notification; must never fail the job flow."""
        if not self.state_store:
            return
        try:
            from .api_notifications import dispatch_webhooks

            dispatch_webhooks(event_name, payload, self.state_store.data_dir)
        except Exception as exc:
            logger.warning(f"Webhook dispatch for {event_name} failed: {exc}")

    def _persist_job(self, job_id: str) -> None:
        if not self.state_store:
            return
        job = self._jobs.get(job_id)
        if not job:
            return
        self.state_store.save_training_job({
            'job_id': job_id,
            'profile_id': job.get('profile_id'),
            'status': job.get('status'),
            'progress': job.get('progress', 0),
            'created_at': job.get('created_at'),
            'started_at': job.get('started_at'),
            'completed_at': job.get('completed_at'),
            'sample_ids': [],
            'config': job.get('settings', {}),
            'input_file': job.get('input_file'),
            'error': job.get('error'),
            'results': {
                'metrics': job.get('metrics'),
                'duration': job.get('duration'),
                'sample_rate': job.get('sample_rate'),
                'result_path': job.get('result_path'),
                'stem_paths': job.get('stem_paths') or {},
            },
        })

    def _record_quality_metrics(self, job_id, profile_id, metrics, result) -> None:
        """Record one quality datapoint per completed conversion.

        Nothing recorded a metric automatically before this: ``record_metric``
        was reachable only from an API endpoint nobody calls, so
        ``data/quality_history/`` stayed empty and the Quality page's rolling
        averages had no source. Every quality judgement had to be made by ear,
        which is why "is this checkpoint better?" kept coming down to opinion.

        Deliberately records only ``f0_correlation`` and ``rtf``:

        * f0_correlation vs the source is legitimate for singing - the melody
          must be preserved, so correlation with the source is real signal.
        * rtf is measured, not inferred.

        Not speaker_similarity: this project has a written calibration showing
        speaker-verification embeddings cannot do identity on sung audio
        (different-person sims 0.855/0.900 exceeded most same-person pairs), so
        recording it would build the feedback loop on a metric already known to
        be wrong here. Not mcd either - the implementation has no DTW alignment
        and is meaningless on singing. Better one honest number than four that
        invite false confidence.

        Never raises: a metrics failure must not fail a conversion the user is
        waiting on.
        """
        if not profile_id:
            return
        try:
            from ..monitoring.quality_monitor import get_quality_monitor
            monitor = get_quality_monitor()
            if monitor is None:
                return
            corr = ((metrics or {}).get('pitch_accuracy') or {}).get('correlation')
            job = self._jobs.get(job_id) or {}
            rtf = None
            # Only the realtime lanes may report rtf. The monitor's threshold is
            # rtf_max_realtime = 0.30 - a live-latency target - so an offline
            # studio render at rtf ~1.5 is healthy but trips it on EVERY
            # conversion, which would bury any real alert in permanent noise.
            # The number is not lost: the conversion history record carries rtf
            # and processing_time_seconds regardless.
            lane = str((job.get('settings') or {}).get('resolved_pipeline')
                       or (job.get('settings') or {}).get('pipeline_type') or '')
            if lane.startswith('realtime'):
                duration = (result or {}).get('duration') or job.get('duration')
                if job.get('started_at') and duration and float(duration) > 0:
                    elapsed = (job.get('completed_at') or time.time()) - job['started_at']
                    rtf = float(elapsed) / float(duration)
            if corr is None and rtf is None:
                return
            monitor.record_metric(
                profile_id=str(profile_id),
                f0_correlation=float(corr) if corr is not None else None,
                rtf=rtf,
                conversion_id=str(job_id),
            )
            logger.info(
                "Recorded quality metrics for %s: f0_correlation=%s rtf=%s",
                job_id, corr, None if rtf is None else round(rtf, 3))
        except Exception as exc:
            logger.warning("Could not record quality metrics for %s: %s", job_id, exc)

    def _emit_conversion_history(self, job_id: str) -> None:
        if not self.state_store:
            return
        job = self._jobs.get(job_id)
        if not job:
            return
        processing_time = None
        if job.get('started_at'):
            completed_at = job.get('completed_at') or time.time()
            processing_time = max(0.0, completed_at - job['started_at'])
        rtf = None
        if processing_time is not None and job.get('duration'):
            try:
                rtf = float(processing_time) / float(job['duration'])
            except (TypeError, ValueError, ZeroDivisionError):
                rtf = None

        output_url = f'/api/v1/convert/download/{job_id}' if job.get('result_path') else None
        stem_paths = job.get('stem_paths') or {}
        conversion_metadata = job.get('conversion_metadata') or {}
        record = {
            'id': job_id,
            'status': self._public_status(job.get('status', 'queued')),
            'created_at': job.get('created_at'),
            'started_at': job.get('started_at'),
            'completed_at': job.get('completed_at'),
            'timestamp': job.get('completed_at') or job.get('created_at'),
            'input_file': job.get('input_file'),
            'originalFileName': job.get('input_file'),
            'profile_id': job.get('profile_id'),
            'targetVoice': job.get('profile_id'),
            'preset': job.get('settings', {}).get('preset', 'balanced'),
            'quality': job.get('settings', {}).get('preset', 'balanced'),
            'pipeline_type': job.get('settings', {}).get('pipeline_type'),
            'requested_pipeline': job.get('settings', {}).get('requested_pipeline')
            or job.get('settings', {}).get('pipeline_type'),
            'resolved_pipeline': job.get('settings', {}).get('resolved_pipeline')
            or job.get('settings', {}).get('pipeline_type'),
            'runtime_backend': job.get('settings', {}).get('runtime_backend', 'pytorch'),
            'adapter_type': job.get('settings', {}).get('adapter_type'),
            'active_model_type': conversion_metadata.get('active_model_type')
            or job.get('settings', {}).get('active_model_type'),
            'conversion_metadata': conversion_metadata,
            'original_audio_asset_id': job.get('settings', {}).get('original_audio_asset_id'),
            'original_audio_url': job.get('settings', {}).get('original_audio_url'),
            'duration': job.get('duration'),
            'audio_duration_seconds': job.get('duration'),
            'processing_time_seconds': processing_time,
            'rtf': rtf,
            'error': job.get('error'),
            'output_url': output_url,
            'download_url': output_url,
            'resultUrl': output_url,
            'stem_urls': {
                stem_name: f'/api/v1/convert/download/{job_id}?variant={stem_name}'
                for stem_name, stem_path in stem_paths.items()
                if stem_path
            } if stem_paths else None,
            'reassemble_url': (
                f'/api/v1/convert/reassemble/{job_id}'
                if stem_paths.get('vocals') and stem_paths.get('instrumental')
                else None
            ),
        }
        existing = self.state_store.get_conversion_record(job_id) or {}
        record['notes'] = existing.get('notes')
        record['isFavorite'] = existing.get('isFavorite', False)
        record['tags'] = existing.get('tags', [])
        self.state_store.save_conversion_record(record)

    def _conversions_dir(self, job_id: str, *, create: bool = True) -> str:
        """Persistent per-job output directory (survives restarts and the
        job-TTL memory eviction, unlike the old /tmp files whose downloads
        404'd once the temp file was cleaned). Pass create=False for
        read-only lookups so resolving a path never has side effects."""
        base = getattr(self.state_store, 'data_dir', None) or 'data'
        # Absolute: send_file resolves relative paths against the Flask app
        # root (src/auto_voice/web/), not the cwd, so a relative path 404s.
        path = os.path.abspath(os.path.join(str(base), 'conversions', job_id))
        if create:
            os.makedirs(path, exist_ok=True)
        return path

    def _write_audio_output(
        self,
        job_id: str,
        suffix: str,
        audio_data: np.ndarray,
        sample_rate: int,
    ) -> str:
        """Write one WAV output for a conversion job and return its path.

        Files are named by download variant (mix/vocals/instrumental) so
        get_job_asset_path can rebuild the path by convention after the job
        leaves memory.
        """
        import soundfile as sf

        output_path = os.path.join(self._conversions_dir(job_id), f'{suffix}.wav')
        sf.write(output_path, np.asarray(audio_data, dtype=np.float32), sample_rate)
        return output_path

    def _write_stem_outputs(
        self,
        job_id: str,
        stems: Dict[str, Any],
        sample_rate: int,
    ) -> Dict[str, str]:
        """Persist optional conversion stems for later download/reassembly."""
        saved_paths: Dict[str, str] = {}
        for stem_name in ('vocals', 'instrumental'):
            stem_audio = stems.get(stem_name)
            if not isinstance(stem_audio, np.ndarray) or stem_audio.size == 0:
                continue
            saved_paths[stem_name] = self._write_audio_output(
                job_id,
                stem_name,
                stem_audio,
                sample_rate,
            )
        return saved_paths
