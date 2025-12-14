import json
from typing import Annotated, Any, Dict
from langgraph.prebuilt import InjectedState
from langchain_core.tools import StructuredTool, Tool

from src.shared.infrastructure.storage.gcp_video_repository import GCPVideoRepository
from src.shared.application.ports.video_repository import VideoRepositoryPort

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
    
    def __init__(self):
        self.repository: VideoRepositoryPort = GCPVideoRepository()

    def call(self, project_id: str) -> str:
        video_ids = self.repository.list_videos(project_id)
        print(f"video_ids: {video_ids}")
        return json.dumps(video_ids, ensure_ascii=False)

    def as_tool(self) -> StructuredTool:
        def tool_func(project_id: Annotated[str, InjectedState("project_id")]) -> str:
            return self.call(project_id)
        
        return StructuredTool.from_function(
            func=tool_func,
            name=self.name,
            description=self.description,
        )