import json
import os
from typing import Any, Dict, Union

from langchain_core.tools import Tool

from src.server.agent.config import BASE_PATH, PROJECT_ID


class ReadEditingStateTool:
    """
    현재 프로젝트의 프로젝트 파일을 읽어 옵니다.
    """
    name = "read_editing_state"
    description = (
        "Read the current editing state of the video project. "
        "This tool returns the complete editing state as JSON, which contains "
        "timeline information, clips, and configurations "
        "This file drives the entire video editing process. "
        "Input: none "
        "Output: editing_state (JSON) - the current project state"
    )

    def _get_editing_state_path(self, project_id: str) -> str:
        """Get the editing state file path for the project."""
        return f"projects/{project_id}/editing_state.json"

    def call(self) -> str:
        try:
            # Get the editing state file path
            editing_state_path = self._get_editing_state_path(PROJECT_ID)
            full_path = editing_state_path
            
            # Check if file exists
            if not os.path.exists(full_path):
                with open(full_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "project_id": PROJECT_ID,
                        "tracks": []
                    }, f, ensure_ascii=False)
            
            # Read and parse the editing state file
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    project_data = json.load(f)
                
                return json.dumps(project_data, ensure_ascii=False)
                
            except json.JSONDecodeError as e:
                return json.dumps({
                    "error": f"Invalid JSON in editing state file: {str(e)}",
                    "project_id": PROJECT_ID,
                    "tracks": []
                }, ensure_ascii=False)
                
        except Exception as e:
            return json.dumps({
                "error": f"Error reading editing state: {str(e)}",
                "project_id": PROJECT_ID,
                "tracks": []
            }, ensure_ascii=False)
        
    def as_tool(self) -> Tool:
        """Convert the tool to a LangGraph-compatible tool format."""
        def tool_func(*args, **kwargs) -> str:
            return self.call()
        
        return Tool(
            name=self.name,
            description=self.description,
            func=tool_func
        )
