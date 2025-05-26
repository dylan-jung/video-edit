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
    
    