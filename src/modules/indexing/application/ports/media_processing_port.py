from abc import ABC, abstractmethod

class MediaProcessingPort(ABC):
    @abstractmethod
    def extract_audio(self, video_path: str, audio_path: str) -> None:
        """
        Extracts audio from a video file.
        
        Args:
            video_path: Path to the source video file.
            audio_path: Path where the extracted audio should be saved.
        
        Raises:
            IOError: If input file doesn't exist or output cannot be written.
            RuntimeError: If extraction fails.
        """
        pass
