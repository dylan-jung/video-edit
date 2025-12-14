import datetime
import logging
import os
from google.cloud import storage

from src.config.settings import Settings, get_settings
from src.shared.application.ports.file_storage import FileStoragePort

logger = logging.getLogger(__name__)

class GCPFileStorage(FileStoragePort):
    def __init__(self, settings: Settings = None):
        self.settings = settings or get_settings()
        self.storage_client = storage.Client()
        if self.settings.GOOGLE_APPLICATION_CREDENTIALS:
             self.storage_client = storage.Client.from_service_account_json(self.settings.GOOGLE_APPLICATION_CREDENTIALS)
        
        self.project = self.settings.GCP_PROJECT_ID
        self.bucket_name = self.settings.BUCKET_NAME
        self.bucket = self.storage_client.bucket(self.bucket_name, user_project=self.project)

    def push_file(self, object_name: str, file_path: str) -> None:
        blob = self.bucket.blob(object_name)
        blob.upload_from_filename(file_path)
        logger.info(f"Uploaded {file_path} to gs://{self.bucket_name}/{object_name}")

    def file_exists(self, object_name: str) -> bool:
        blob = self.bucket.blob(object_name)
        return blob.exists()

    def read_file(self, object_name: str) -> bytes:
        blob = self.bucket.blob(object_name)
        if not blob.exists():
            raise FileNotFoundError(f"File not found: {object_name}")
        return blob.download_as_bytes()

    def generate_upload_signed_url(self, object_name: str, content_type: str, expires_in: int = 3600) -> str:
        blob = self.bucket.blob(object_name)
        return blob.generate_signed_url(
            expiration=datetime.timedelta(seconds=expires_in),
            method="PUT",
            content_type=content_type
        )

    def download_file(self, object_name: str, destination_path: str) -> None:
        blob = self.bucket.blob(object_name)
        if not blob.exists():
            raise FileNotFoundError(f"File not found in storage: {object_name}")
        
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        blob.download_to_filename(destination_path)
        logger.info(f"Downloaded gs://{self.bucket_name}/{object_name} to {destination_path}")
