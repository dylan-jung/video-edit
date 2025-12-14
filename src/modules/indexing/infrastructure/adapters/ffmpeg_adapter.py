import os
import subprocess
import logging
from src.modules.indexing.application.ports.media_processing_port import MediaProcessingPort

logger = logging.getLogger(__name__)

class FFmpegAdapter(MediaProcessingPort):
    def extract_audio(self, video_path: str, audio_path: str) -> None:
        if not os.path.exists(video_path):
             raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Extracting audio from {video_path} to {audio_path}")
        
        # ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 audio.wav
        command = [
            "ffmpeg", 
            "-i", video_path,
            "-vn", 
            "-acodec", "pcm_s16le", 
            "-ar", "16000", 
            "-ac", "1", 
            "-y", # Overwrite output files without asking
            "-loglevel", "error", # Less verbose
            audio_path
        ]
        
        try:
            subprocess.run(
                command, 
                check=True, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.PIPE
            )
            logger.info("Audio extraction completed successfully.")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else "Unknown error"
            logger.error(f"FFmpeg failed: {error_msg}")
            raise RuntimeError(f"Audio extraction failed: {error_msg}") from e
