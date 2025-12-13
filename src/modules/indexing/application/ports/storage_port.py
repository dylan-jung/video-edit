from abc import ABC, abstractmethod
from typing import List

class StoragePort(ABC):
    @abstractmethod
    def push_file(self, project_id: str, video_id: str, file_path: str, file_name: str, sub_path: str = None) -> None:
        pass

    @abstractmethod
    def file_exists(self, project_id: str, video_id: str, file_name: str, sub_path: str = None) -> bool:
        pass
    
    @abstractmethod
    def read_file(self, file_path: str) -> bytes:
        pass
