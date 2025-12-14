import json
import logging
import os
from typing import List

from google.cloud import storage

from src.config.settings import Settings, get_settings
from src.shared.application.ports.video_repository import VideoRepositoryPort

logger = logging.getLogger(__name__)

class GCPVideoRepository(VideoRepositoryPort):
    def __init__(self, settings: Settings = None):
        self.settings = settings or get_settings()
        self.storage_client = storage.Client()
        if self.settings.GOOGLE_APPLICATION_CREDENTIALS:
             self.storage_client = storage.Client.from_service_account_json(self.settings.GOOGLE_APPLICATION_CREDENTIALS)
        
        self.project = self.settings.GCP_PROJECT_ID
        self.bucket_name = self.settings.BUCKET_NAME
        self.bucket = self.storage_client.bucket(self.bucket_name, user_project=self.project)

    def _read_blob_as_bytes(self, object_name: str) -> bytes:
        blob = self.bucket.blob(object_name)
        if not blob.exists():
             raise FileNotFoundError(f"File not found: {object_name}")
        return blob.download_as_bytes()

    def _get_artifact_prefix(self, project_id: str, video_id: str) -> str:
        return f"projects/{project_id}/{video_id}"

    def get_video_metadata(self, project_id: str, video_id: str) -> dict:
        path = f"{self._get_artifact_prefix(project_id, video_id)}/metadata.json"
        content = self._read_blob_as_bytes(path)
        return json.loads(content)
    
    def get_video_scenes(self, project_id: str, video_id: str) -> List[dict]:
        path = f"{self._get_artifact_prefix(project_id, video_id)}/scenes.json"
        content = self._read_blob_as_bytes(path)
        return json.loads(content)

    def get_video_audio(self, project_id: str, video_id: str) -> bytes:
        path = f"{self._get_artifact_prefix(project_id, video_id)}/audio.wav"
        return self._read_blob_as_bytes(path)

    def list_videos(self, project_id: str) -> List[str]:
        # Videos are stored at projects/{project_id}/videos/{video_id}.mp4
        prefix = f"projects/{project_id}/videos/"
        blobs = self.bucket.list_blobs(prefix=prefix)
        
        video_ids = []
        for blob in blobs:
            # projects/{project_id}/videos/{video_id}.mp4
            name = blob.name
            if name.endswith(".mp4"):
                # Extract video_id
                if name.startswith(prefix):
                    remainder = name[len(prefix):]
                    video_id = remainder.replace(".mp4", "")
                    video_ids.append(video_id)
        return video_ids

    def get_video_scene_descriptions(self, project_id: str, video_id: str) -> dict:
        path = f"{self._get_artifact_prefix(project_id, video_id)}/scene_descriptions.json"
        content = self._read_blob_as_bytes(path)
        return json.loads(content)

    def download_video(self, project_id: str, video_id: str, destination_path: str) -> None:
        # Path: projects/{project_id}/videos/{video_id}.mp4
        object_name = f"projects/{project_id}/videos/{video_id}.mp4"
        blob = self.bucket.blob(object_name)
        if not blob.exists():
            raise FileNotFoundError(f"Video not found: {object_name}")
        
        blob.download_to_filename(destination_path)
        logger.info(f"Downloaded video {object_name} to {destination_path}")

    def save_vision_vector_db(self, project_id: str, file_path: str) -> None:
        # Path: projects/{project_id}/vision_vector_db.faiss
        object_name = f"projects/{project_id}/vision_vector_db.faiss"
        blob = self.bucket.blob(object_name)
        blob.upload_from_filename(file_path)
        logger.info(f"Uploaded vision vector DB to {object_name}")
        
        # Upload Metadata Sidecar
        metadata_path = f"{file_path}.metadata"
        if os.path.exists(metadata_path):
            meta_object_name = f"{object_name}.metadata"
            blob_meta = self.bucket.blob(meta_object_name)
            blob_meta.upload_from_filename(metadata_path)
            logger.info(f"Uploaded vision vector DB metadata to {meta_object_name}")

    def save_speech_vector_db(self, project_id: str, file_path: str) -> None:
        # Path: projects/{project_id}/speech_vector_db.faiss
        object_name = f"projects/{project_id}/speech_vector_db.faiss"
        blob = self.bucket.blob(object_name)
        blob.upload_from_filename(file_path)
        logger.info(f"Uploaded speech vector DB to {object_name}")
        
        # Upload Metadata Sidecar
        metadata_path = f"{file_path}.metadata"
        if os.path.exists(metadata_path):
            meta_object_name = f"{object_name}.metadata"
            blob_meta = self.bucket.blob(meta_object_name)
            blob_meta.upload_from_filename(metadata_path)
            logger.info(f"Uploaded speech vector DB metadata to {meta_object_name}")

    def download_vision_vector_db(self, project_id: str, destination_path: str) -> None:
        object_name = f"projects/{project_id}/vision_vector_db.faiss"
        blob = self.bucket.blob(object_name)
        if blob.exists():
            blob.download_to_filename(destination_path)
            logger.info(f"Downloaded vision vector DB from {object_name}")
            
            # Download Sidecar
            meta_object_name = f"{object_name}.metadata"
            blob_meta = self.bucket.blob(meta_object_name)
            if blob_meta.exists():
                blob_meta.download_to_filename(f"{destination_path}.metadata")
                logger.info(f"Downloaded vision vector DB metadata from {meta_object_name}")
        else:
             logger.info(f"Vision vector DB not found at {object_name}, skipping download.")

    def download_speech_vector_db(self, project_id: str, destination_path: str) -> None:
        object_name = f"projects/{project_id}/speech_vector_db.faiss"
        blob = self.bucket.blob(object_name)
        if blob.exists():
            blob.download_to_filename(destination_path)
            logger.info(f"Downloaded speech vector DB from {object_name}")
            
            # Download Sidecar
            meta_object_name = f"{object_name}.metadata"
            blob_meta = self.bucket.blob(meta_object_name)
            if blob_meta.exists():
                blob_meta.download_to_filename(f"{destination_path}.metadata")
                logger.info(f"Downloaded speech vector DB metadata from {meta_object_name}")
        else:
             logger.info(f"Speech vector DB not found at {object_name}, skipping download.")

    def save_project_file(self, project_id: str, file_path: str) -> None:
        # Path: projects/{project_id}/project.json
        object_name = f"projects/{project_id}/project.json"
        blob = self.bucket.blob(object_name)
        blob.upload_from_filename(file_path)
        logger.info(f"Uploaded project.json to {object_name}")

    def download_project_file(self, project_id: str, destination_path: str) -> None:
        object_name = f"projects/{project_id}/project.json"
        blob = self.bucket.blob(object_name)
        if blob.exists():
            blob.download_to_filename(destination_path)
            logger.info(f"Downloaded project.json from {object_name}")
        else:
             logger.info(f"project.json not found at {object_name}, skipping download.")
