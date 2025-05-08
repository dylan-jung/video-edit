import datetime
import os

from google.cloud import storage

from .repository import Repository


class CloudStorageRepository(Repository):
    def __init__(self):
        self.storage_client = storage.Client()
        self.storage_client.from_service_account_json(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        self.project = os.getenv("GCP_PROJECT_ID")
        self.bucket = self.storage_client.bucket("attenz", user_project=self.project)

    def push_file(self, project_id: str, video_id: str, file_path: str, file_name: str):
        blob = self.bucket.blob(f"{project_id}/{video_id}/{file_name}")
        blob.upload_from_filename(file_path)

    def file_exists(self, project_id: str, video_id: str, file_name: str) -> bool:
        blob = self.bucket.blob(f"{project_id}/{video_id}/{file_name}")
        return blob.exists()
    
    def get_temp_download_link(self, project_id: str, video_id: str, file_name: str, expires_in: int = 60 * 60 * 24) -> str:
        blob = self.bucket.blob(f"{project_id}/{video_id}/{file_name}")
        return blob.generate_signed_url(expiration=datetime.timedelta(seconds=expires_in))

    def get_video_metadata(self, project_id: str, video_id: str) -> bytes:
        blob = self.bucket.blob(f"{project_id}/{video_id}/metadata.json")
        return blob.download_as_bytes()
    
    def get_video_scenes(self, project_id: str, video_id: str) -> bytes:
        blob = self.bucket.blob(f"{project_id}/{video_id}/scenes.json")
        return blob.download_as_bytes()
    
    def get_video_audio(self, project_id: str, video_id: str) -> bytes:
        blob = self.bucket.blob(f"{project_id}/{video_id}/audio.wav")
        return blob.download_as_bytes()
    
    def get_video_video(self, project_id: str, video_id: str) -> bytes:
        blob = self.bucket.blob(f"{project_id}/{video_id}/video.mp4")
        return blob.download_as_bytes()
    
    