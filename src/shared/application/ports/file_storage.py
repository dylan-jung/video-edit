from abc import ABC, abstractmethod
from typing import Optional

class FileStoragePort(ABC):
    @abstractmethod
    def push_file(self, object_name: str, file_path: str) -> None:
        """Upload a file to storage."""
        pass

    @abstractmethod
    def file_exists(self, object_name: str) -> bool:
        """Check if a file exists in storage."""
        pass

    @abstractmethod
    def read_file(self, object_name: str) -> bytes:
        """Read a file's content directly."""
        pass

    @abstractmethod
    def generate_upload_signed_url(self, object_name: str, content_type: str, expires_in: int = 3600) -> str:
        """Generate a signed URL for uploading."""
        pass

    @abstractmethod
    def download_file(self, object_name: str, destination_path: str) -> None:
        """Download a file from storage to local path."""
        pass
