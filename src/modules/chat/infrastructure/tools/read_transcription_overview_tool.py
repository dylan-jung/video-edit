import json
from typing import Any, Dict, Union

from langchain_core.tools import Tool

from src.shared.infrastructure.repository.storage import CloudStorageRepository
from src.shared.infrastructure.repository.repository import Repository
from src.modules.chat.config import PROJECT_ID
from src.shared.utils.get_video_id import get_video_id


class ReadVideoTranscriptionOverviewTool:
    name = "read_video_transcription_overview"
    description = (
        "Read a simplified transcription overview for a specific video. "
        "Returns the transcribed speech content as plain text without timestamp information. "
        "This is useful for quickly understanding the spoken content and dialogue in the video. "
        "Note that transcriptions are AI-generated and may not be 100% accurate. "
        "For detailed transcription with timestamps, use read_video_transcription instead. "
        "Input: video_id (str) - the id of the video to read transcription overview from "
        "Output: transcription_overview (str) - the transcription overview of the video"
    )

    def __init__(self):
        self.repository: Repository = CloudStorageRepository()

    def call(self, video_id: str) -> str:
        transcription_bytes = self.repository.get_video_transcription(PROJECT_ID, video_id)
        transcription_json = json.loads(transcription_bytes)
        transcription_overview = transcription_json["transcription"]
        return transcription_overview
        
    def as_tool(self) -> Tool:
        """Convert the tool to a LangGraph-compatible tool format."""
        def tool_func(*args, **kwargs) -> str:
            # LangChain이 video_id를 args[0]이나 kwargs에서 전달할 수 있음
            video_id = args[0] if args else kwargs.get('video_id', '')
            return self.call(video_id)
        
        return Tool(
            name=self.name,
            description=self.description,
            func=tool_func
        )
