from abc import ABC, abstractmethod

class SpeechIndexerPort(ABC):
    @abstractmethod
    def index_speech(self, project_id: str, video_id: str, speech_analysis_path: str, vector_db_url: str) -> None:
        """
        Index speech from analysis results.
        """
        pass
