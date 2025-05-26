import json
from typing import Any, Dict, Union

from src.repository.cloud_storage_repository import CloudStorageRepository
from src.repository.repository import Repository
from src.server.agent.config import PROJECT_ID
from src.utils.get_video_id import get_video_id


class ReadVideoSceneDescriptionsTool:
    """
    Cloud Storage 에 저장된 동영상의 장면 설명(scene_descriptions, JSON)을 읽어 옵니다.
    """
    name = "read_video_scene_descriptions"
    description = (
        "Return scene descriptions (as JSON) for a video.",
        "scene_descriptions is a JSON file that contains scene descriptions information. this file has information about the video's scenes, including the start and end time of each scene.",
        "Output: scene_descriptions (as JSON)"
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
        scene_descriptions_bytes = self.repository.get_video_scene_descriptions(PROJECT_ID, video_id)
        scene_descriptions = json.loads(scene_descriptions_bytes)

        return json.dumps(scene_descriptions, ensure_ascii=False)
        
    @staticmethod
    def as_tool() -> Dict[str, Any]:
        """Convert the tool to a LangGraph-compatible tool format."""
        return {
            "type": "function",
            "function": {
                "name": ReadVideoSceneDescriptionsTool.name,
                "description": "\n".join(ReadVideoSceneDescriptionsTool.description),
                "parameters": ReadVideoSceneDescriptionsTool.parameters
            }
        }
