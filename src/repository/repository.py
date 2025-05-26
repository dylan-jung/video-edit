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
    def get_video_audio(self, project_id: str, video_id: str) -> bytes:
        pass

    @abstractmethod
    def get_video_video(self, project_id: str, video_id: str) -> bytes:
        pass

    @abstractmethod
    def list_videos(self, project_id: str) -> list[str]:
        pass