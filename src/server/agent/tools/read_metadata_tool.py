# tools/read_video_metadata.py
import json
from typing import Any, Dict, Union

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
        "Return metadata (as JSON) for a video.",
        "metadata is a JSON file that contains video metadata information. this file has information about the video's duration, resolution, fps, creation_time etc.",
        "Output: metadata (as JSON)"
    )
    parameters = {
        "type": "object",
        "properties": {
            "video_id": {
                "type": "string",
                "description": "the id of the video file"
            }
        },
        "required": ["video_id"]
    }
    def __init__(self):
        self.repository: Repository = CloudStorageRepository()

    def call(self, video_id: str) -> str:
        metadata_bytes = self.repository.get_video_metadata(PROJECT_ID, video_id)
        metadata = json.loads(metadata_bytes)

        # Tool-calling 프로토콜에 맞춰 항상 문자열(JSON)로 반환
        return json.dumps(metadata, ensure_ascii=False)
    
    @staticmethod
    def as_tool() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": ReadVideoMetadataTool.name,
                "description": "\n".join(ReadVideoMetadataTool.description),
                "parameters": ReadVideoMetadataTool.parameters
            }
        }
