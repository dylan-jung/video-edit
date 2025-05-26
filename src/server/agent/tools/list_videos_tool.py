import json
from typing import Any, Dict

from src.repository.cloud_storage_repository import CloudStorageRepository


class ListVideosTool:
    """
    프로젝트에 속한 모든 동영상 비디오 id를 읽어옵니다.
    """
    name = "list_videos"
    description = (
        "Reads the video IDs of all videos in the project.",
        "Output: video_ids (list[str]) - the ids of the videos in the project"
    )
    parameters = {
        "type": "object",
        "properties": {
            "project_id": {
                "type": "string",
                "description": "the id of the project"
            }
        },
        "required": ["project_id"]
    }

    def __init__(self):
        self.repository = CloudStorageRepository()

    def call(self, project_id: str) -> str:
        video_ids = self.repository.list_videos(project_id)
        print(f"video_ids: {video_ids}")
        return json.dumps(video_ids, ensure_ascii=False)

    @staticmethod
    def as_tool() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": ListVideosTool.name,
                "description": "\n".join(ListVideosTool.description),
                "parameters": ListVideosTool.parameters
            }
        }