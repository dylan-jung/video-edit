import os

import supabase

from .repository import Repository


class SupabaseRepository(Repository):
    def __init__(self):
        self.client = supabase.create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY"),
        )

    def create_video_bucket(self, video_id: str) -> dict:
        self.client.storage.create_bucket(
            "video",
            "video"
        )

    def push_file(self, project_id: str, video_id: str, upload_file_path: str, file_name: str = None) -> dict:
        if file_name is None:
            file_name = os.path.basename(upload_file_path)

        with open(upload_file_path, "rb") as f:
            self.client.storage.from_("video").upload(
                path=f"{project_id}/{video_id}/{file_name}",
                file=f,
            )
    
    def file_exists(self, project_id: str, video_id: str, file_name: str) -> bool:
        """
        Check if a file exists in storage.
        
        Args:
            project_id: The project ID
            video_id: The video ID
            file_name: The file name to check
            
        Returns:
            bool: True if the file exists, False otherwise
        """
        try:
            # List files in the directory and check if the file exists
            files = self.client.storage.from_("video").list(f"{project_id}/{video_id}")
            return any(file.get("name") == file_name for file in files)
        except Exception:
            # If any error occurs, assume the file doesn't exist
            return False
    
    def get_temp_download_link(self, project_id: str, video_id: str, file_name: str, expires_in: int = 60 * 60 * 24) -> str:
        response = self.client.storage.from_("video").create_signed_url(f"{project_id}/{video_id}/{file_name}", expires_in)
        return response["signedURL"]

    def get_video_metadata(self, project_id: str, video_id: str) -> bytes:
        response = self.client.storage.from_("video").download(f"{project_id}/{video_id}/metadata.json")
        return response
    
    def get_video_scenes(self, project_id: str, video_id: str) -> bytes:
        response = self.client.storage.from_("video").download(f"{project_id}/{video_id}/scenes.json")
        return response
    
    def get_video_audio(self, project_id: str, video_id: str) -> bytes:
        response = self.client.storage.from_("video").download(f"{project_id}/{video_id}/audio.wav")
        return response
    
    def get_video_video(self, project_id: str, video_id: str) -> bytes:
        response = self.client.storage.from_("video").download(f"{project_id}/{video_id}/video.mp4")
        return response