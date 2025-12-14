from abc import ABC, abstractmethod
from typing import List

class VideoRepositoryPort(ABC):
    @abstractmethod
    def get_video_metadata(self, project_id: str, video_id: str) -> dict:
        pass
    
    @abstractmethod
    def get_video_scenes(self, project_id: str, video_id: str) -> List[dict]:
        pass

    @abstractmethod
    def get_video_audio(self, project_id: str, video_id: str) -> bytes:
        pass

    @abstractmethod
    def list_videos(self, project_id: str) -> List[str]:
        pass

    @abstractmethod
    def get_video_scene_descriptions(self, project_id: str, video_id: str) -> dict:
        pass

    @abstractmethod
    def download_video(self, project_id: str, video_id: str, destination_path: str) -> None:
        """Download the original video file to local path."""
        pass

    @abstractmethod
    def save_vision_vector_db(self, project_id: str, file_path: str) -> None:
        """Upload vision vector DB file."""
        pass

    @abstractmethod
    def save_speech_vector_db(self, project_id: str, file_path: str) -> None:
        """Upload speech vector DB file."""
        pass

    @abstractmethod
    def download_vision_vector_db(self, project_id: str, destination_path: str) -> None:
        """Download vision vector DB file."""
        pass

    @abstractmethod
    def download_speech_vector_db(self, project_id: str, destination_path: str) -> None:
        """Download speech vector DB file."""
        pass

    @abstractmethod
    def save_project_file(self, project_id: str, file_path: str) -> None:
        """Upload project.json file."""
        pass

    @abstractmethod
    def download_project_file(self, project_id: str, destination_path: str) -> None:
        """Download project.json file."""
        pass
