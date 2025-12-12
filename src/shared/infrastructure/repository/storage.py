import datetime
import os

from google.cloud import storage

from src.config.settings import Settings, get_settings
from .repository import Repository


class CloudStorageRepository(Repository):
    def __init__(self, settings: Settings = None):
        self.settings = settings or get_settings()
        self.storage_client = storage.Client()
        if self.settings.GOOGLE_APPLICATION_CREDENTIALS:
             self.storage_client = storage.Client.from_service_account_json(self.settings.GOOGLE_APPLICATION_CREDENTIALS)
        
        self.project = self.settings.GCP_PROJECT_ID
        self.bucket = self.storage_client.bucket(self.settings.BUCKET_NAME, user_project=self.project)

    def push_file(self, project_id: str, video_id: str, file_path: str, file_name: str, sub_path: str = None):
        if sub_path:
            blob = self.bucket.blob(f"{project_id}/{video_id}/{sub_path}/{file_name}")
        else:
            blob = self.bucket.blob(f"{project_id}/{video_id}/{file_name}")
        blob.upload_from_filename(file_path)

    def file_exists(self, project_id: str, video_id: str, file_name: str, sub_path: str = None) -> bool:
        if sub_path:
            blob = self.bucket.blob(f"{project_id}/{video_id}/{sub_path}/{file_name}")
        else:
            blob = self.bucket.blob(f"{project_id}/{video_id}/{file_name}")
        return blob.exists()
    
    def get_temp_download_link(self, project_id: str, video_id: str, file_name: str, expires_in: int = 60 * 60 * 24, sub_path: str = None) -> str:
        if sub_path:
            blob = self.bucket.blob(f"{project_id}/{video_id}/{sub_path}/{file_name}")
        else:
            blob = self.bucket.blob(f"{project_id}/{video_id}/{file_name}")
        return blob.generate_signed_url(expiration=datetime.timedelta(seconds=expires_in))

    def generate_upload_signed_url(self, project_id: str, video_id: str, file_name: str, content_type: str, expires_in: int = 3600, sub_path: str = None) -> str:
        if sub_path:
            blob = self.bucket.blob(f"{project_id}/{video_id}/{sub_path}/{file_name}")
        else:
            blob = self.bucket.blob(f"{project_id}/{video_id}/{file_name}")
        
        return blob.generate_signed_url(
            expiration=datetime.timedelta(seconds=expires_in),
            method="PUT",
            content_type=content_type
        )

    def list_videos(self, project_id: str) -> list[str]:
        root = f"{project_id}/"
        blobs = self.bucket.list_blobs(prefix=root, delimiter="/")
        list(blobs) # iterate blobs to get the prefixes(gcp need it)
        return [p[len(root):].rstrip("/") for p in blobs.prefixes]

    def get_video_metadata(self, project_id: str, video_id: str) -> bytes:
        blob = self.bucket.blob(f"{project_id}/{video_id}/metadata.json")
        return blob.download_as_bytes()
    
    def get_video_scenes(self, project_id: str, video_id: str) -> bytes:
        blob = self.bucket.blob(f"{project_id}/{video_id}/scenes.json")
        return blob.download_as_bytes()
    
    def get_video_scene_descriptions(self, project_id: str, video_id: str) -> bytes:
        blob = self.bucket.blob(f"{project_id}/{video_id}/scene_descriptions.json")
        return blob.download_as_bytes()

    def get_video_audio(self, project_id: str, video_id: str) -> bytes:
        blob = self.bucket.blob(f"{project_id}/{video_id}/audio.wav")
        return blob.download_as_bytes()
    
    def get_video_video(self, project_id: str, video_id: str) -> bytes:
        blob = self.bucket.blob(f"{project_id}/{video_id}/video.mp4")
        return blob.download_as_bytes()
    
    def get_video_transcription(self, project_id: str, video_id: str) -> bytes:
        blob = self.bucket.blob(f"{project_id}/{video_id}/transcription.json")
        return blob.download_as_bytes()
    
    def read_file(self, file_path: str) -> bytes:
        """Read a file from the storage."""
        blob = self.bucket.blob(file_path)
        if not blob.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return blob.download_as_bytes()

    def write_file(self, file_path: str, content: bytes) -> None:
        """Write content to a file in the storage."""
        blob = self.bucket.blob(file_path)
        blob.upload_from_string(content)

    def get_editing_state(self, project_id: str) -> bytes:
        """Get the editing state for a project."""
        blob = self.bucket.blob(f"projects/{project_id}/editing_state.json")
        if not blob.exists():
            raise FileNotFoundError(f"Editing state not found for project: {project_id}")
        return blob.download_as_bytes()

    def save_editing_state(self, project_id: str, state_data: bytes) -> None:
        """Save the editing state for a project."""
        blob = self.bucket.blob(f"projects/{project_id}/editing_state.json")
        blob.upload_from_string(state_data)
    
    