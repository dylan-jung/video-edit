from abc import ABC, abstractmethod


class Repository(ABC):
    @abstractmethod
    def push_file(self, project_id: str, video_id: str, file_path: str, file_name: str):
        pass
    
    @abstractmethod
    def file_exists(self, project_id: str, video_id: str, file_name: str) -> bool:
        pass
    
    @abstractmethod
    def get_temp_download_link(self, project_id: str, video_id: str, file_name: str, expires_in: int = 60 * 60 * 24) -> str:
        pass
    
    @abstractmethod
    def get_video_metadata(self, project_id: str, video_id: str) -> bytes:
        pass
    
    @abstractmethod
    def get_video_scenes(self, project_id: str, video_id: str) -> bytes:
        pass

    @abstractmethod
    def get_video_scene_descriptions(self, project_id: str, video_id: str) -> bytes:
        pass

    @abstractmethod
    def get_video_audio(self, project_id: str, video_id: str) -> bytes:
        pass

    @abstractmethod
    def get_video_video(self, project_id: str, video_id: str) -> bytes:
        pass

    @abstractmethod
    def get_video_transcription(self, project_id: str, video_id: str) -> bytes:
        pass

    @abstractmethod
    def list_videos(self, project_id: str) -> list[str]:
        pass

    @abstractmethod
    def read_file(self, file_path: str) -> bytes:
        """Read a file from the storage."""
        pass

    @abstractmethod
    def write_file(self, file_path: str, content: bytes) -> None:
        """Write content to a file in the storage."""
        pass

    @abstractmethod
    def get_editing_state(self, project_id: str) -> bytes:
        """Get the editing state for a project."""
        pass

    @abstractmethod
    def save_editing_state(self, project_id: str, state_data: bytes) -> None:
        """Save the editing state for a project."""
        pass

    @abstractmethod
    def generate_upload_signed_url(self, project_id: str, video_id: str, file_name: str, content_type: str, expires_in: int = 3600, sub_path: str = None) -> str:
        pass