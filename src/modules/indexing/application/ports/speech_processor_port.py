from abc import ABC, abstractmethod
from typing import Dict, List, Any

class SpeechProcessorPort(ABC):
    @abstractmethod
    def process_audio(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Process audio file (split, transcribe) and return chunks.
        """
        pass

    @abstractmethod
    def enhance_transcription(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance transcription chunks with AI analysis.
        """
        pass
