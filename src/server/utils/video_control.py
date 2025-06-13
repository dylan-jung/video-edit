import base64
import os
import shutil
import subprocess
import tempfile
from typing import List

import cv2

from src.server.utils.cache_manager import get_cache_path


def sample_video_some_fps(video_path: str, fps: int = 1, use_cache: bool = True) -> str:
    """
    Sample video at 1 fps and return the sampled video path in temp directory.
    """
    # Generate temp file path
    temp_dir = os.path.dirname(video_path)
    filename = os.path.basename(video_path)
    name_without_ext = os.path.splitext(filename)[0]
    output_path = os.path.join(temp_dir, f"{name_without_ext}_fps.mp4")
    
    # Check cache first
    cache_args = {
        "operation": "sample_fps",
        "fps": fps,
        "input_path": video_path
    }
    cache_exists, cache_path = get_cache_path(video_path, cache_args)
    
    if cache_exists and use_cache:
        # Copy from cache to temp directory
        shutil.copy2(cache_path, output_path)
        print(f"FPS 샘플링 캐시에서 로드됨: {output_path}")
        return output_path
    
    # Create sampled video if not in cache
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        output_path
    ]
    
    result = subprocess.run(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    if os.path.exists(output_path) and use_cache:
        # Save to cache
        shutil.copy2(output_path, cache_path)
        return output_path
    elif os.path.exists(output_path):
        return output_path
    else:
        raise ValueError(f"1 FPS 샘플링 생성 실패: {result}")


def trim_and_sample_video(video_path: str, start_time: str, end_time: str, fps: float = 1, use_cache: bool = True) -> str:
    """
    Trim video to specific time range and sample at specified fps using single ffmpeg command.
    
    Args:
        video_path: Path to the input video file
        start_time: Start time in format "hh:mm:ss"
        end_time: End time in format "hh:mm:ss" 
        fps: Frames per second to sample (default: 1)
        use_cache: Whether to use caching (default: True)
    
    Returns:
        Path to the trimmed and sampled video file
    """
    # Generate temp file path
    temp_dir = os.path.dirname(video_path)
    filename = os.path.basename(video_path)
    name_without_ext = os.path.splitext(filename)[0]
    output_path = os.path.join(temp_dir, f"{name_without_ext}_trimmed_{start_time.replace(':', '')}-{end_time.replace(':', '')}_fps{fps}.mp4")
    
    # Check cache first
    cache_args = {
        "operation": "trim_and_sample",
        "start_time": start_time,
        "end_time": end_time,
        "fps": fps,
        "input_path": video_path
    }
    cache_exists, cache_path = get_cache_path(video_path, cache_args)
    
    if cache_exists and use_cache:
        # Copy from cache to temp directory
        shutil.copy2(cache_path, output_path)
        print(f"Trim & Sample 캐시에서 로드됨: {output_path}")
        return output_path
    
    print(f"🔍 Trim & Sample 비디오 생성 시작: {video_path} -> {output_path}")
    # Create trimmed and sampled video using single ffmpeg command
    cmd = [
        "ffmpeg", "-i", video_path,
        "-ss", start_time,
        "-to", end_time,
        "-vf", f"fps={fps}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        output_path
    ]
    
    result = subprocess.run(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    print(result)
    if os.path.exists(output_path) and use_cache:
        # Save to cache
        shutil.copy2(output_path, cache_path)
        return output_path
    elif os.path.exists(output_path):
        return output_path
    else:
        raise ValueError(f"Trim & Sample 생성 실패: {result.stderr}")


def extract_video_chunk_frames(video_path: str, start_time: str | int, end_time: str | int, fps: float = 1) -> List[str]:
    """
    Extract frames from a specific time range of the video.
    
    Args:
        video_path: Path to the video file
        start_time: Start time in format "hh:mm:ss"
        end_time: End time in format "hh:mm:ss"
        fps: Frames per second to extract (default: 1)
    
    Returns:
        List of base64 encoded frame strings
    """
    # First trim and sample the video
    if isinstance(start_time, int):
        start_time = f"{start_time // 3600:02d}:{start_time % 3600 // 60:02d}:{start_time % 60:02d}"
    if isinstance(end_time, int):
        end_time = f"{end_time // 3600:02d}:{end_time % 3600 // 60:02d}:{end_time % 60:02d}"

    print(start_time, end_time)
    trimmed_video_path = trim_and_sample_video(video_path, start_time, end_time, fps, use_cache=False)
    
    try:
        # Extract frames from the trimmed video
        frames = extract_frames_from_video(trimmed_video_path)
        return frames
    finally:
        pass
        # Clean up the temporary trimmed video file
        if os.path.exists(trimmed_video_path):
            try:
                os.remove(trimmed_video_path)
            except OSError:
                pass  # File might be in use or already deleted


def extract_frames_from_video(video_path: str) -> List[str]:
    """
    Extract frames from video and convert to base64 encoded strings.
    
    Args:
        video_path: Path to the video file
        sample_rate: Extract every nth frame (default: 30, roughly 1 frame per second for 30fps video)
    
    Returns:
        List of base64 encoded frame strings
    """
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        # raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
        print(f"비디오 파일을 열 수 없습니다: {video_path}")
        return []
    
    base64_frames = []
    frame_count = 0
    
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
            
        # Encode frame to JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        # Convert to base64
        base64_frame = base64.b64encode(buffer).decode("utf-8")
        base64_frames.append(base64_frame)
        
        frame_count += 1
    
    video.release()
    print(f"{len(base64_frames)} 프레임을 추출했습니다.")
    return base64_frames