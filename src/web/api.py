"""FastAPI interface for AutoVoice."""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import os
import tempfile
import logging
from typing import Optional, List
import asyncio
from ..inference import VoiceSynthesizer
from ..audio import AudioProcessor, VoiceAnalyzer

logger = logging.getLogger(__name__)


class SynthesisRequest(BaseModel):
    """Request model for synthesis."""
    text: str
    speaker_id: Optional[int] = 0
    pitch_scale: Optional[float] = 1.0
    speed_scale: Optional[float] = 1.0
    emotion: Optional[str] = "neutral"


class VoiceCloneRequest(BaseModel):
    """Request model for voice cloning."""
    text: str
    preserve_prosody: Optional[bool] = True


class VoiceConversionRequest(BaseModel):
    """Request model for voice conversion."""
    target_speaker_id: int


class BatchSynthesisRequest(BaseModel):
    """Request model for batch synthesis."""
    texts: List[str]
    speaker_ids: Optional[List[int]] = None


class VoiceAPI:
    """FastAPI interface for voice synthesis."""

    def __init__(self, model_path: str, vocoder_path: str):
        """Initialize API.

        Args:
            model_path: Path to acoustic model
            vocoder_path: Path to vocoder model
        """
        self.app = FastAPI(title="AutoVoice API", version="1.0.0")

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Initialize models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.synthesizer = VoiceSynthesizer(model_path, vocoder_path, self.device)
        self.audio_processor = AudioProcessor(device=self.device)
        self.voice_analyzer = VoiceAnalyzer(device=self.device)

        # Temporary directory for files
        self.temp_dir = tempfile.mkdtemp()

        # Register routes
        self._register_routes()

    def _register_routes(self):
        """Register API routes."""

        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {"message": "AutoVoice API", "version": "1.0.0"}

        @self.app.post("/synthesize")
        async def synthesize(request: SynthesisRequest):
            """Synthesize speech from text."""
            try:
                # Synthesize
                waveform = await asyncio.to_thread(
                    self.synthesizer.synthesize,
                    text=request.text,
                    speaker_id=request.speaker_id,
                    pitch_scale=request.pitch_scale,
                    speed_scale=request.speed_scale
                )

                # Save to temporary file
                output_path = os.path.join(self.temp_dir, f"synthesis_{os.getpid()}.wav")
                self.synthesizer.save_audio(waveform, output_path)

                return FileResponse(output_path, media_type="audio/wav",
                                  filename="synthesized.wav")

            except Exception as e:
                logger.error(f"Synthesis error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/batch_synthesize")
        async def batch_synthesize(request: BatchSynthesisRequest,
                                  background_tasks: BackgroundTasks):
            """Batch synthesis of multiple texts."""
            try:
                # Generate job ID
                job_id = f"batch_{os.getpid()}_{len(request.texts)}"

                # Start background task
                background_tasks.add_task(
                    self._process_batch_synthesis,
                    job_id, request.texts, request.speaker_ids
                )

                return {"job_id": job_id, "status": "processing",
                       "num_texts": len(request.texts)}

            except Exception as e:
                logger.error(f"Batch synthesis error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/clone")
        async def clone_voice(text: str = None, audio: UploadFile = File(...)):
            """Clone voice from reference audio."""
            try:
                # Save uploaded file
                temp_path = os.path.join(self.temp_dir, audio.filename)
                content = await audio.read()
                with open(temp_path, "wb") as f:
                    f.write(content)

                # Clone voice
                waveform = await asyncio.to_thread(
                    self.synthesizer.clone_voice,
                    temp_path, text
                )

                # Save output
                output_path = os.path.join(self.temp_dir, f"cloned_{os.getpid()}.wav")
                self.synthesizer.save_audio(waveform, output_path)

                # Clean up
                os.remove(temp_path)

                return FileResponse(output_path, media_type="audio/wav",
                                  filename="cloned.wav")

            except Exception as e:
                logger.error(f"Voice cloning error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/convert")
        async def convert_voice(target_speaker: int, audio: UploadFile = File(...)):
            """Convert voice to target speaker."""
            try:
                # Save uploaded file
                temp_path = os.path.join(self.temp_dir, audio.filename)
                content = await audio.read()
                with open(temp_path, "wb") as f:
                    f.write(content)

                # Convert voice
                waveform = await asyncio.to_thread(
                    self.synthesizer.convert_voice,
                    temp_path, target_speaker
                )

                # Save output
                output_path = os.path.join(self.temp_dir, f"converted_{os.getpid()}.wav")
                self.synthesizer.save_audio(waveform, output_path)

                # Clean up
                os.remove(temp_path)

                return FileResponse(output_path, media_type="audio/wav",
                                  filename="converted.wav")

            except Exception as e:
                logger.error(f"Voice conversion error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/analyze")
        async def analyze_voice(audio: UploadFile = File(...)):
            """Analyze voice characteristics."""
            try:
                # Save uploaded file
                temp_path = os.path.join(self.temp_dir, audio.filename)
                content = await audio.read()
                with open(temp_path, "wb") as f:
                    f.write(content)

                # Load and analyze
                waveform, _ = self.audio_processor.load_audio(temp_path)
                characteristics = await asyncio.to_thread(
                    self.voice_analyzer.analyze_voice_characteristics,
                    waveform
                )

                # Clean up
                os.remove(temp_path)

                return JSONResponse(content=characteristics)

            except Exception as e:
                logger.error(f"Voice analysis error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/compare")
        async def compare_voices(audio1: UploadFile = File(...),
                                audio2: UploadFile = File(...)):
            """Compare two voice samples."""
            try:
                # Save uploaded files
                temp_path1 = os.path.join(self.temp_dir, f"1_{audio1.filename}")
                temp_path2 = os.path.join(self.temp_dir, f"2_{audio2.filename}")

                content1 = await audio1.read()
                content2 = await audio2.read()

                with open(temp_path1, "wb") as f:
                    f.write(content1)
                with open(temp_path2, "wb") as f:
                    f.write(content2)

                # Load audio
                waveform1, _ = self.audio_processor.load_audio(temp_path1)
                waveform2, _ = self.audio_processor.load_audio(temp_path2)

                # Compare
                comparison = await asyncio.to_thread(
                    self.voice_analyzer.compare_voices,
                    waveform1, waveform2
                )

                # Clean up
                os.remove(temp_path1)
                os.remove(temp_path2)

                return JSONResponse(content=comparison)

            except Exception as e:
                logger.error(f"Voice comparison error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/speakers")
        async def list_speakers():
            """List available speakers."""
            speakers = [
                {"id": i, "name": f"Speaker {i+1}",
                 "language": "en", "gender": ["female", "male"][i % 2]}
                for i in range(10)
            ]
            return speakers

        @self.app.get("/job/{job_id}")
        async def get_job_status(job_id: str):
            """Get batch job status."""
            # Check if output files exist
            output_dir = os.path.join(self.temp_dir, job_id)
            if os.path.exists(output_dir):
                files = os.listdir(output_dir)
                return {
                    "job_id": job_id,
                    "status": "completed",
                    "files": files
                }
            else:
                return {
                    "job_id": job_id,
                    "status": "processing"
                }

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "device": str(self.device),
                "models_loaded": True
            }

    async def _process_batch_synthesis(self, job_id: str, texts: List[str],
                                      speaker_ids: Optional[List[int]] = None):
        """Process batch synthesis in background.

        Args:
            job_id: Job identifier
            texts: List of texts to synthesize
            speaker_ids: Optional speaker IDs
        """
        try:
            # Create output directory
            output_dir = os.path.join(self.temp_dir, job_id)
            os.makedirs(output_dir, exist_ok=True)

            # Process each text
            for i, text in enumerate(texts):
                speaker_id = speaker_ids[i] if speaker_ids else 0

                # Synthesize
                waveform = self.synthesizer.synthesize(
                    text=text,
                    speaker_id=speaker_id
                )

                # Save
                output_path = os.path.join(output_dir, f"audio_{i:03d}.wav")
                self.synthesizer.save_audio(waveform, output_path)

            logger.info(f"Batch synthesis completed: {job_id}")

        except Exception as e:
            logger.error(f"Batch synthesis error for {job_id}: {e}")

    def run(self, host="0.0.0.0", port=8000):
        """Run the API server.

        Args:
            host: Host address
            port: Port number
        """
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)