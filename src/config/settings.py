from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional
import os

class CommonSettings(BaseSettings):
    PROJECT_ID: str = "test"
    DEBUG: bool = False
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

class GCPSettings(BaseSettings):
    GCP_PROJECT_ID: Optional[str] = None
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    BUCKET_NAME: str = "attenz"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

class DBSettings(BaseSettings):
    MONGO_URI: str = "mongodb://localhost:27017"
    DB_NAME: str = "video_edit"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

class Settings(CommonSettings, GCPSettings, DBSettings):
    """
    Main settings class.
    """
    
    def get_cloud_storage_path(self, project_id: str, video_id: str, file_name: str, sub_path: str = None) -> str:
        if sub_path:
            return f"{project_id}/{video_id}/{sub_path}/{file_name}"
        return f"{project_id}/{video_id}/{file_name}"

@lru_cache
def get_settings() -> Settings:
    return Settings()
