# tools/read_video_metadata.py
import json
from typing import Any, Dict, Union

from langchain_core.tools import Tool

from src.repository.cloud_storage_repository import CloudStorageRepository
from src.repository.repository import Repository
from src.server.agent.config import PROJECT_ID
from src.utils.get_video_id import get_video_id


class ReadVideoMetadataTool:
    """
    Cloud Storage 에 저장된 동영상의 메타데이터(JSON)를 읽어 옵니다.
    """
    name = "read_video_metadata"
    description = (
        "Read metadata information for a specific video. "
        "Returns comprehensive video metadata as JSON including duration, resolution, "
        "frame rate (fps), creation time, file size, and other technical properties. "
        "This metadata is essential for understanding video characteristics before editing. "
        "Input: video_id (str) - the id of the video to read metadata from "
        "Output: metadata (JSON) - the metadata of the video"
    )
    
    def __init__(self):
        self.repository: Repository = CloudStorageRepository()

    def call(self, video_id: str) -> str:
        metadata_bytes = self.repository.get_video_metadata(PROJECT_ID, video_id)
        metadata = json.loads(metadata_bytes)

        # Tool-calling 프로토콜에 맞춰 항상 문자열(JSON)로 반환
        return json.dumps(metadata, ensure_ascii=False)
    
    def as_tool(self) -> Tool:
        def tool_func(*args, **kwargs) -> str:
            # LangChain이 video_id를 args[0]이나 kwargs에서 전달할 수 있음
            video_id = args[0] if args else kwargs.get('video_id', '')
            return self.call(video_id)
        
        return Tool(
            name=self.name,
            description=self.description,
            func=tool_func
        )
