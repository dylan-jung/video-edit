import json
from typing import Any, Dict

from langchain_core.tools import Tool

from src.shared.infrastructure.storage.gcp_video_repository import GCPVideoRepository

class ListVideosTool:
    """
    프로젝트에 속한 모든 동영상 비디오 id를 읽어옵니다.
    """
    name = "list_videos"
    description = (
        "List all video IDs in the current project. "
        "This tool retrieves the identifiers of all videos that belong to the project. "
        "These video IDs can be used with other tools to access specific video data "
        "such as metadata, transcriptions, or scene descriptions. "
        "Input: none "
        "Output: video_ids (list[str]) - the ids of the videos in the project"
    )
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.repository = GCPVideoRepository()

    def call(self) -> str:
        video_ids = self.repository.list_videos(self.project_id)
        print(f"video_ids: {video_ids}")
        return json.dumps(video_ids, ensure_ascii=False)

    def as_tool(self) -> Tool:
        def tool_func(*args, **kwargs) -> str:
            return self.call()
        
        return Tool(
            name=self.name,
            description=self.description,
            func=tool_func
        )