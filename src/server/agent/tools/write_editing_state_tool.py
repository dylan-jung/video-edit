import difflib
import json
import os
from typing import Any, Dict, Union

from langchain_core.tools import Tool

from src.server.agent.config import BASE_PATH, PROJECT_ID


class WriteEditingStateTool:
    """
    프로젝트 파일을 쓰기 위한 도구입니다.
    """
    name = "write_editing_state"
    description = (
        "Write or update the video editing state file with JSON diff/patch. "
        "The editing state contains timeline information, clips, and configurations "
        "that drive the video editing process.\n\n"
        "Required parameter: 'diff' - JSON changes to apply to the editing state.\n\n"
        "Editing state structure:\n"
        "{\n"
        "  \"project_id\": string,\n"
        "  \"tracks\": [\n"
        "    {\n"
        "      \"src\": string,       // video ID\n"
        "      \"start\": string,     // start time in track (hh:mm:ss)\n"
        "      \"end\": string,       // end time in track (hh:mm:ss)\n"
        "      \"duration\": string,  // playback duration (hh:mm:ss)\n"
        "      \"trimIn\": string,    // trim start from source (hh:mm:ss)\n"
        "      \"trimOut\": string    // trim end from source (hh:mm:ss)\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Example diff to add a clip:\n"
        "{\n"
        "  \"tracks\": [\n"
        "    {\n"
        "      \"src\": \"ea48283a31baa560\",\n"
        "      \"start\": \"00:00:00\",\n"
        "      \"end\": \"00:00:10\",\n"
        "      \"duration\": \"00:00:10\",\n"
        "      \"trimIn\": \"00:00:05\",\n"
        "      \"trimOut\": \"00:00:15\"\n"
        "    }\n"
        "  ]\n"
        "}"
    )

    def _get_editing_state_path(self, project_id: str) -> str:
        """Get the editing state file path for the project."""
        return f"projects/{project_id}/editing_state.json"

    def call(self, diff: str) -> str:
        try:
            # Parse diff string to JSON
            if not diff:
                return "Error: No diff provided"
                
            try:
                diff_data = json.loads(diff)
            except json.JSONDecodeError:
                return "Error: Invalid JSON format in diff"

            # Get the editing state file path
            editing_state_path = self._get_editing_state_path(PROJECT_ID)
            full_path = editing_state_path
            
            # Try to get existing editing state
            current_state = {}
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        current_state = json.load(f)
                except Exception as e:
                    return f"Error reading existing editing state: {str(e)}"
            else:
                print(f"No existing editing state found at {full_path}, creating new one")
                # Start with empty state if file doesn't exist
                current_state = {
                    "project_id": PROJECT_ID,
                    "tracks": [],  # Track[] - 각 트랙은 Clip[]
                }
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Apply diff to current state
            updated_state = self._apply_diff(current_state, diff_data)
            
            # Save updated state
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(updated_state, f, ensure_ascii=False, indent=2)
            
            return f"Successfully updated editing state at {full_path}"
            
        except Exception as e:
            return f"Error updating editing state: {str(e)}"

    def _apply_diff(self, current_state: Dict[str, Any], diff_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply diff to the current state.
        
        Args:
            current_state: Current editing state
            diff_data: Diff to apply
            
        Returns:
            Updated state
        """
        # Deep copy to avoid modifying original
        updated_state = json.loads(json.dumps(current_state))
        
        # Apply diff - this is a simple merge, can be made more sophisticated
        self._deep_merge(updated_state, diff_data)
        
        return updated_state
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Deep merge source into target dictionary.
        
        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
        
    def as_tool(self) -> Tool:
        """Convert the tool to a LangGraph-compatible tool format."""
        def tool_func(diff: str) -> str:
            return self.call(diff)
        
        return Tool(
            name=self.name,
            description=self.description,
            func=tool_func
        )