import json
from typing import Annotated, Any, Dict, Union

from langchain_core.tools import StructuredTool, Tool
from langgraph.prebuilt import InjectedState

from src.shared.infrastructure.storage.gcp_video_repository import GCPVideoRepository
from src.shared.application.ports.video_repository import VideoRepositoryPort
# from src.modules.chat.config import PROJECT_ID
from src.shared.utils.get_video_id import get_video_id


class ReadVideoTranscriptionTool:
    """
    Cloud Storage 에 저장된 동영상의 트랜스크립션(JSON)을 읽어 옵니다.
    """
    name = "read_video_transcription"
    description = (
        "Read detailed transcription data for a specific video. "
        "Returns comprehensive transcription information as JSON, including the complete "
        "transcribed text with precise timestamps (start and end times) for each segment. "
        "This enables accurate synchronization between speech and video timeline for editing. "
        "Note that transcriptions are AI-generated and may not be 100% accurate. "
        "Input: video_id (str) - the id of the video to read transcription from "
        "Output: transcription (JSON) - the transcription of the video"
    )

    def __init__(self):
        self.repository: VideoRepositoryPort = GCPVideoRepository()

    def call(self, video_id: str, project_id: str) -> str:
        transcription_bytes = self.repository.get_video_transcription(project_id, video_id)
        transcription_json = json.loads(transcription_bytes)
        transcription = transcription_json["transcription"] 
        return json.dumps(transcription, ensure_ascii=False)
        
    def as_tool(self) -> StructuredTool:
        """Convert the tool to a LangGraph-compatible tool format."""
        def tool_func(video_id: str, project_id: Annotated[str, InjectedState("project_id")]) -> str:
            return self.call(video_id, project_id)
        
        return StructuredTool.from_function(
            func=tool_func,
            name=self.name,
            description=self.description,
        )
