import os
import shutil
from typing import Dict, Type, Optional

from src.shared.infrastructure.ai.vector_db import VectorDB
from src.shared.infrastructure.ai.speech_vector_db import SpeechVectorDB
from src.shared.application.ports.video_repository import VideoRepositoryPort
from src.shared.infrastructure.storage.gcp_video_repository import GCPVideoRepository


class VectorDBRegistry:
    _instance = None
    
    def __init__(self, video_repository: VideoRepositoryPort = None):
        if VectorDBRegistry._instance is not None:
             raise Exception("This class is a singleton!")
        
        self.video_repository = video_repository or GCPVideoRepository()
        self._vision_cache: Dict[str, VectorDB] = {}
        self._speech_cache: Dict[str, SpeechVectorDB] = {}
        
        # Local cache directory setup
        self.base_dir = "projects"
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    @classmethod
    def get_instance(cls) -> 'VectorDBRegistry':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """For testing purposes"""
        cls._instance = None

    def _get_local_path(self, project_id: str, filename: str) -> str:
        project_dir = os.path.join(self.base_dir, project_id)
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)
        return os.path.join(project_dir, filename)

    def get_vision_db(self, project_id: str) -> VectorDB:
        # 1. Check Memory Cache
        if project_id in self._vision_cache:
            return self._vision_cache[project_id]

        # 2. Check Local Disk & Download if missing
        db_filename = "vision_vector_db.faiss"
        local_path = self._get_local_path(project_id, db_filename)
        
        # We try to load even if file missing (VectorDB.load handles fallback/creation)
        # But for robustness in distributed env, we should try download if completely missing
        if not os.path.exists(local_path):
            print(f"📥 Vision DB not found locally for {project_id}, attempting download...")
            try:
                self.video_repository.download_vision_vector_db(project_id, local_path)
            except Exception as e:
                print(f"⚠️ Failed to download Vision DB: {e}. Creating new empty DB.")
        
        # 3. Load from Disk
        try:
            vector_db = VectorDB.load(local_path)
        except Exception as e:
            print(f"❌ Failed to load Vision DB, creating empty: {e}")
            vector_db = VectorDB()
            
        # 4. Update Memory Cache
        self._vision_cache[project_id] = vector_db
        return vector_db

    def get_speech_db(self, project_id: str) -> SpeechVectorDB:
        # 1. Check Memory Cache
        if project_id in self._speech_cache:
            return self._speech_cache[project_id]

        # 2. Check Local Disk & Download if missing
        db_filename = "speech_vector_db.faiss"
        local_path = self._get_local_path(project_id, db_filename)
        
        if not os.path.exists(local_path):
            print(f"📥 Speech DB not found locally for {project_id}, attempting download...")
            try:
                self.video_repository.download_speech_vector_db(project_id, local_path)
            except Exception as e:
                print(f"⚠️ Failed to download Speech DB: {e}. Creating new empty DB.")
        
        # 3. Load from Disk
        try:
            vector_db = SpeechVectorDB.load(local_path)
        except Exception as e:
            print(f"❌ Failed to load Speech DB, creating empty: {e}")
            vector_db = SpeechVectorDB()

        # 4. Update Memory Cache
        self._speech_cache[project_id] = vector_db
        return vector_db

    def invalidate(self, project_id: str):
        """Invalidate cache for a project (e.g., after indexing updates)"""
        if project_id in self._vision_cache:
            del self._vision_cache[project_id]
        if project_id in self._speech_cache:
            del self._speech_cache[project_id]
            
        # Optional: Clean up local files to force re-download on next access?
        # For now, let's keep local files as they might be updated by this same process (if used in indexer)
        # But if this is purely a query service, we might want to delete them.
        # Given "distributed" concern, assume writer is elsewhere. So we SHOULD delete local files to ensure freshness.
        
        project_dir = os.path.join(self.base_dir, project_id)
        if os.path.exists(project_dir):
            try:
                shutil.rmtree(project_dir)
                print(f"🧹 Cleared local cache for project {project_id}")
            except Exception as e:
                print(f"⚠️ Failed to clear local cache: {e}")
        
        print(f"🔄 Invalidated Registry cache for {project_id}")
