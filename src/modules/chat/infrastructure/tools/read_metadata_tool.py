# tools/read_video_metadata.py
import json
from typing import Annotated, Any, Dict, Union

from langchain_core.tools import StructuredTool, Tool
from langgraph.prebuilt import InjectedState

from src.shared.infrastructure.storage.gcp_video_repository import GCPVideoRepository
from src.shared.application.ports.video_repository import VideoRepositoryPort
from src.shared.utils.get_video_id import get_video_id


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
        self.repository: VideoRepositoryPort = GCPVideoRepository()

    def call(self, video_id: str, project_id: str) -> str:
        metadata_bytes = self.repository.get_video_metadata(project_id, video_id)
        metadata = json.loads(metadata_bytes)

        # Tool-calling 프로토콜에 맞춰 항상 문자열(JSON)로 반환
        return json.dumps(metadata, ensure_ascii=False)
    
    def as_tool(self) -> StructuredTool:
        def tool_func(video_id: str, project_id: Annotated[str, InjectedState("project_id")]) -> str:
            return self.call(video_id, project_id)
        
        return StructuredTool.from_function(
            func=tool_func,
            name=self.name,
            description=self.description,
        )
