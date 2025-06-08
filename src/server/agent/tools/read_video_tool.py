import json
import os
from typing import List

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.server.agent.config import PROJECT_ID
from src.server.utils.video_control import extract_video_chunk_frames


class ReadVideoInput(BaseModel):
    """Input schema for ReadVideoTool"""
    video_id: str = Field(description="ID/filename of the video file")
    start_time: str = Field(description="Start time in format 'hh:mm:ss'")
    end_time: str = Field(description="End time in format 'hh:mm:ss'")
    fps: float = Field(default=0.5, description="Frames per second to extract (less than 1)")


class ReadVideoTool:
    name = "read_video"
    description = (
        "Read and extract frames from a specific time range of a video file from the project. "
        "This tool trims the video to the specified time range and samples frames at the given fps. "
        "If user wants to get more accurate results, you should use this tool. "
        "IMPORTANT: This tool consumes many tokens due to image processing. "
        "Use small time ranges (< 30 seconds) and fps=0.5 for most cases. "
        "If you want to handle long video, you should use small fps first."
        # "For longer videos, break into smaller chunks. "
        "Maximum recommended: 30 frames per call to avoid rate limits. "
        "FPS should be less than 1. Use 0.5 for faster processing and 1 when you need more accurate results. "
        "In the video, the timestamp is written in the right upper corner in the format of hh:mm:ss.fff "
        "Input: video_id (str), start_time (str in hh:mm:ss format), end_time (str in hh:mm:ss format), fps (float, default=0.5) "
        "Output: List of base64 encoded frame strings from the specified video segment"
    )

    def __init__(self):
        # Token estimation constants (approximate)
        self.tokens_per_frame = 1500  # Estimated tokens per image frame
        self.max_frames_per_call = 30  # Limit to avoid rate limits
        pass

    def call(self, video_id: str, start_time: str, end_time: str, fps: float = 1) -> str:
        """
        Extract frames from a specific time range of the video.
        
        Args:
            video_id: ID/filename of the video file
            start_time: Start time in format "hh:mm:ss"
            end_time: End time in format "hh:mm:ss"
            fps: Frames per second to extract (default: 1)
        
        Returns:
            List of base64 encoded frame strings
        """
        # Find video file in project root
        video_path = None
        
        # Check if video_id is a direct filename in project root
        # TODO: 비디오 파일 cloud storage에서 읽어오기
        video_path = self._validate_video_path(video_id)
        self._validate_fps(fps)
        
        # Validate time format (basic check)
        if not self._validate_time_format(start_time) or not self._validate_time_format(end_time):
            raise ValueError("시간 형식이 잘못되었습니다. hh:mm:ss 형식을 사용해주세요.")
        
        # Estimate number of frames to prevent rate limiting
        duration_seconds = self._calculate_duration(start_time, end_time)
        estimated_frames = duration_seconds * fps
        
        if estimated_frames > self.max_frames_per_call:
            suggested_end = self._calculate_max_end_time(start_time, fps)
            raise ValueError(
                f"⚠️ 요청된 프레임 수({estimated_frames})가 너무 많습니다. "
                f"Rate Limit을 피하기 위해 {self.max_frames_per_call}프레임 이하로 제한해주세요. "
                f"제안: end_time을 {suggested_end}로 줄이거나 fps를 낮춰주세요."
            )
        
        # Extract frames from the specified time range using the new utility function
        try:
            frames = extract_video_chunk_frames(video_path, start_time, end_time, fps)
            
            # Additional safety check on actual frame count
            if len(frames) > self.max_frames_per_call:
                frames = frames[:self.max_frames_per_call]
                print(f"⚠️ 프레임 수를 {self.max_frames_per_call}개로 제한했습니다.")
            
            # Estimate token usage
            estimated_tokens = len(frames) * self.tokens_per_frame
            print(f"📊 예상 토큰 사용량: ~{estimated_tokens:,} tokens ({len(frames)} frames)")
            
            content = [{
                "type": "text",
                "text": f"비디오 '{video_id}'에서 {start_time}-{end_time} 구간, {fps}fps로 {len(frames)}개 프레임을 추출했습니다. (예상 토큰: ~{estimated_tokens:,})"
            }]
            for frame in frames:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame}"
                    }
                })
            print(f"✅ 비디오 '{video_id}'에서 {start_time}-{end_time} 구간, {fps}fps로 {len(frames)}개 프레임을 추출했습니다.")
            return json.dumps(content, ensure_ascii=False)
        except Exception as e:
            raise ValueError(f"비디오 청크 추출 실패: {str(e)}")

    def _calculate_duration(self, start_time: str, end_time: str) -> int:
        """Calculate duration in seconds between start and end time"""
        def time_to_seconds(time_str: str) -> int:
            parts = time_str.split(':')
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        
        return time_to_seconds(end_time) - time_to_seconds(start_time)
    
    def _calculate_max_end_time(self, start_time: str, fps: float) -> str:
        """Calculate maximum end time for given fps to stay within frame limit"""
        max_duration = self.max_frames_per_call / fps
        
        def time_to_seconds(time_str: str) -> int:
            parts = time_str.split(':')
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        
        def seconds_to_time(seconds: float) -> str:
            seconds = int(seconds)  # Convert float to int to avoid formatting error
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        
        start_seconds = time_to_seconds(start_time)
        max_end_seconds = start_seconds + max_duration
        return seconds_to_time(max_end_seconds)

    def _validate_video_path(self, video_id: str) -> str:
        """
        Validate video path
        """
        project_root = os.getcwd()
        if os.path.exists(os.path.join(project_root, "projects", PROJECT_ID, video_id, "video.mp4")):
            return os.path.join(project_root, "projects", PROJECT_ID, video_id, "video.mp4")
        else:
            raise ValueError(f"찾을 수 없는 비디오 파일: {video_id}")

    def _validate_fps(self, fps: float) -> bool:
        """
        Validate fps
        """
        if fps <= 1.001:
            return True
        else:
            raise ValueError(f"fps: {fps} is not valid. fps should be less than 1.")

    def _validate_time_format(self, time_str: str) -> bool:
        """
        Validate time format hh:mm:ss
        """
        try:
            parts = time_str.split(':')
            if len(parts) != 3:
                return False
            
            hours, minutes, seconds = parts
            if not (0 <= int(hours) <= 23 and 0 <= int(minutes) <= 59 and 0 <= int(seconds) <= 59):
                return False
            
            return True
        except (ValueError, AttributeError):
            return False

    def as_tool(self) -> StructuredTool:
        def tool_func(video_id: str, start_time: str, end_time: str, fps: float = 1) -> str:
            print(f"🔍 Tool called with video_id: {video_id}, start_time: {start_time}, end_time: {end_time}, fps: {fps}")
            return self.call(video_id, start_time, end_time, fps)
        
        return StructuredTool(
            name=self.name,
            description=self.description,
            func=tool_func,
            args_schema=ReadVideoInput
        )