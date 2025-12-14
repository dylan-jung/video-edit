from abc import ABC, abstractmethod
from typing import Any

class AIServicePort(ABC):
    @abstractmethod
    def transcribe_audio(self, audio_input: Any) -> Any:
        """Transcribe audio using Whisper model."""
        pass
