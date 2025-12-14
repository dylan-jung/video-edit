import base64
import json
import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.modules.indexing.infrastructure.scene_analyzer.prompt.scene_base import scene_base_prompt
from src.modules.indexing.infrastructure.scene_analyzer.prompt.scene_describer import scene_describer_prompt
from src.modules.indexing.infrastructure.scene_analyzer.prompt.scene_merger import scene_merger_prompt
from src.shared.infrastructure.video.service import (
    extract_frames_from_video,
    extract_video_chunk_frames
)

logger = logging.getLogger(__name__)

class GPTSceneAnalyzer:
    def __init__(self, model_name: str = "gpt-4.1-mini", temperature: float = 0.1):
        """
        Initialize GPT Scene Analyzer.
        
        Args:
            model_name: GPT model name (default: gpt-4.1-mini).
            temperature: Creativity control.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.model = self._init_gpt_model()

    def _init_gpt_model(self) -> ChatOpenAI:
        if "OPENAI_API_KEY" not in os.environ:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
        
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=10000,
            max_retries=5,
            api_key=os.environ["OPENAI_API_KEY"]
        )

    def _create_message_with_frames(self, base64_frames: List[str], prompt_text: str) -> HumanMessage:
        content = [{"type": "text", "text": prompt_text}]
        for frame in base64_frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
            })
        return HumanMessage(content=content)

    def analyze_chunk(self, video_path: str, chunk_index: int) -> List[Dict[str, Any]]:
        """
        Analyze a specific video chunk.
        
        Args:
            video_path: Path to the sampled video chunk.
            chunk_index: Index of the chunk.
            
        Returns:
            List of scene dictionaries.
        """
        try:
            logger.info(f"Extracting frames for chunk {chunk_index} from {video_path}")
            base64_frames = extract_frames_from_video(video_path)
            
            if not base64_frames:
                logger.warning(f"No frames extracted for chunk {chunk_index}")
                return []

            # Step 1: Group Scenes
            logger.info(f"Analyzing chunk {chunk_index} grouping...")
            grouping_prompt = f"This is chunk {chunk_index + 1}. First, identify and group similar scenes."
            messages = [
                SystemMessage(content=scene_base_prompt),
                self._create_message_with_frames(base64_frames, grouping_prompt)
            ]
            
            response_grouping = self.model.invoke(messages)
            
            # Step 2: Describe Scenes
            logger.info(f"Analyzing chunk {chunk_index} description...")
            messages.append(AIMessage(content=response_grouping.content))
            messages.append(SystemMessage(content=scene_describer_prompt))
            messages.append(HumanMessage(content="Now analyze the scenes detailedly based on the grouping."))
            
            response_desc = self.model.invoke(messages)
            return self._parse_json_response(response_desc.content)
            
        except Exception as e:
            logger.error(f"Failed to analyze chunk {chunk_index}: {e}")
            raise e

    def merge_chunks(self, video_path: str, analysis_results: List[Dict[str, Any]], chunk_duration: int = 300, fps: int = 1) -> List[Dict[str, Any]]:
        """
        Merge scenes across chunks.
        """
        if not analysis_results:
            return []

        responses = []
        # Calculate boundaries and compare last scene of current with first of next
        for i in range(len(analysis_results) - 1):
            boundary_time = chunk_duration * (i + 1)
            this_last_scene = analysis_results[i][-1] if analysis_results[i] else {}
            next_first_scene = analysis_results[i+1][0] if analysis_results[i+1] else {}
            
            if not this_last_scene or not next_first_scene:
                responses.append({"is_same_scene": False})
                continue

            # Extractframes around boundary
            # 3 seconds before and after boundary
            # Need to pass video_path of the FULL video or relevant chunks? 
            # The 'video_path' arg should be the FULL video path if we want to extract from valid timestamps.
            # Assuming 'video_path' provided here is the original full video.
            
            try:
                this_frames = extract_video_chunk_frames(video_path, max(0, boundary_time - 3), boundary_time, fps)
                next_frames = extract_video_chunk_frames(video_path, boundary_time, boundary_time + 3, fps)
                
                this_prompt = f"Scene A Analysis:\n{json.dumps(this_last_scene, ensure_ascii=False)}\n\nVisuals attached."
                next_prompt = f"Scene B Analysis:\n{json.dumps(next_first_scene, ensure_ascii=False)}\n\nVisuals attached."
                
                messages = [
                    SystemMessage(content=scene_merger_prompt),
                    self._create_message_with_frames(this_frames, this_prompt),
                    self._create_message_with_frames(next_frames, next_prompt),
                    HumanMessage(content="Determine if these two scenes are visually and contextually connected.")
                ]
                
                response = self.model.invoke(messages)
                merge_decision = self._parse_json_response(response.content)
                responses.append(merge_decision)
                
            except Exception as e:
                logger.error(f"Error merging boundary {i}: {e}")
                responses.append({"is_same_scene": False})

        return self._apply_merge_logic(analysis_results, responses)

    def _parse_json_response(self, content: str) -> Any:
        try:
            cleaned = content.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0]
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1]
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {content}")
            raise e

    def _apply_merge_logic(self, analysis_results: List[List[Dict]], responses: List[Dict]) -> List[Dict]:
        """
        Merge scenes based on LLM decisions.
        """
        merged_scenes = []
        
        # Add all scenes from first chunk EXCEPT the last one initially
        if not analysis_results: return []
        
        # We process chunk by chunk
        current_chunk_idx = 0
        
        # Add scenes from chunk 0, up to -1
        # Then check boundary.
        # This logic is complex because multiple chunks.
        
        # Simplification:
        # Flatten everything? No, boundaries matter.
        
        # Reusing the logic provided in original code but cleaning it up.
        
        # Start with chunk 0
        current_chunk = analysis_results[0]
        if not current_chunk:
             pass # Should handle emptiness
             
        for i in range(len(analysis_results) - 1):
            chunk = analysis_results[i]
            next_chunk = analysis_results[i+1]
            decision = responses[i]
            
            # Add all scenes of current chunk except last
            for scene in chunk[:-1]:
                merged_scenes.append(scene)
            
            last_scene = chunk[-1]
            
            if decision.get("is_same_scene"):
                # Merge last_scene and next_chunk[0]
                merged_scene = decision.get("merged_scene")
                if merged_scene:
                    merged_scenes.append(merged_scene)
                    # Modify next_chunk to start from index 1 potentially?
                    # The loop continues to next chunk `i+1`.
                    # But `analysis_results[i+1]` still has the first scene.
                    # We MUST mutate or skip the first scene of next chunk when processing it next iteration.
                    # Actually, better to accumulate `pending_scenes` logic.
                    
                    # Hack: Remove the first scene from next chunk in the list reference so next iteration ignores it?
                    # Side effects are dangerous.
                    
                    # Better Approach:
                    # Just skip adding the first scene of next chunk in next iteration if merged.
                    # But we are in a loop `for i`. 
                    # Let's clean up logic:
                    
                    # We have `decision` for boundary i->i+1.
                    # If merged: `merged_scenes` gets `merged_scene`.
                    # Next iteration (i+1) needs to know its first scene was consumed.
                    
                    # Since we can't easily look ahead/behind cleanly with side effects:
                    # Let's modify `analysis_results[i+1][0]` to be None? No.
                    pass
                else:
                    # Fallback
                    merged_scenes.append(last_scene)
                    # Next chunk first scene remains.
            
            else:
                # Not merged
                merged_scenes.append(last_scene)
                # Next chunk first scene remains.

            # Wait, if we merged, `merged_scene` REPLACES `chunk[-1]` AND `next_chunk[0]`.
            # So `next_chunk[0]` should NOT be added in the NEXT iteration's `chunk[:-1]` loop.
            
            if decision.get("is_same_scene"):
                 # Pop the first element of next chunk to "consume" it
                 if analysis_results[i+1]:
                     analysis_results[i+1].pop(0)

        # Handle last chunk
        last_chunk = analysis_results[-1]
        for scene in last_chunk:
            merged_scenes.append(scene)
            
        return merged_scenes