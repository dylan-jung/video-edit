from functools import partial
from typing import Any, Dict, List, Union

from langchain_core.tools import StructuredTool, Tool

from src.server.agent.config import PROJECT_ID
from src.server.agent.tools.read_transcription_overview_tool import \
    ReadVideoTranscriptionOverviewTool
from src.server.agent.tools.read_video_tool import ReadVideoTool
from src.server.agent.tools.sementic_speech_search_tool import \
    SementicSpeechSearchTool

from .list_videos_tool import ListVideosTool
from .read_editing_state_tool import ReadEditingStateTool
from .read_metadata_tool import ReadVideoMetadataTool
from .read_scene_descriptions_tool import ReadVideoSceneDescriptionsTool
from .read_transcription_tool import ReadVideoTranscriptionTool
from .sementic_vision_search_tool import SementicVisionSearchTool
from .write_editing_state_tool import WriteEditingStateTool

available_tools = [
    ListVideosTool(),
    ReadVideoMetadataTool(),
    ReadEditingStateTool(),
    WriteEditingStateTool(),
    ReadVideoSceneDescriptionsTool(),
    # ReadVideoTranscriptionOverviewTool(),
    ReadVideoTranscriptionTool(),
    SementicVisionSearchTool(),
    SementicSpeechSearchTool(),
    ReadVideoTool(),
]

class ToolExecutor:
    """A class for executing tools based on tool calls from LangGraph."""
    
    def __init__(self):
        """Initialize the tool executor with a list of tools."""
        self.tools_dict = {}

        for tool in available_tools:
            tool_name = tool.name
            self.tools_dict[tool_name] = tool
            print(f"✅ Registered tool: {tool_name}")

    def execute_tool(self, tool_call: Dict[str, Any]) -> str:
        """Execute a tool based on a tool call."""
        tool_name: str = tool_call["name"]
        tool_args: Dict[str, Any] = tool_call.get("parameters", {})
        
        if tool_name not in self.tools_dict:
            return f"Tool {tool_name} not found."
        try:
            result = self.tools_dict[tool_name].call(**tool_args)
            return result
        except Exception as e:
            return f"Error executing tool {tool_name}: {str(e)}"

    def as_tool(self) -> List[Union[Tool, StructuredTool]]:
        """Return LangChain Tool objects."""
        return [tool.as_tool() for tool in available_tools]

__all__ = ["ToolExecutor"]