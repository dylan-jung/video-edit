import base64
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from PIL import Image
from prompt.scene_describer import sys_prompt

from src.server.utils.cache_manager import get_cache_path, get_file_id
from src.server.utils.cookie_utils import get_code_server_cookies

# TODO: Code server 세션 쿠키 때문에 만들어 둠
cookies = get_code_server_cookies()

httpx_client = httpx.Client(
    cookies=cookies,
    timeout=180.0
)

def split_scenes(video_path: str, scenes: List[Dict], ffmpeg_path: str = "ffmpeg") -> List[str]:
    """
    Split a video into scenes based on provided timestamps.

    Args:
        video_path: Path to the input video file
        scenes: List of (start_time, end_time) tuples in seconds or "HH:MM:SS" format
        output_dir: Directory to save extracted scenes
        use_cache: Whether to use cached scenes if available
        ffmpeg_path: Path to ffmpeg executable

    Returns:
        List of paths to the extracted scene video files
    """
    hit, cache_path = get_cache_path(video_path, {})
    video_path = os.path.join(os.environ.get('PYTHONPATH'), video_path)

    # Create output directory based on cache path
    output_dir = os.path.join(os.path.dirname(cache_path), os.path.basename(cache_path).split('.')[0])
    os.makedirs(output_dir, exist_ok=True)

    if hit:
        print(f"🔍 Using cached scenes: {cache_path}")
        with open(cache_path, 'r') as f:
            scene_paths = json.load(f)
        return scene_paths
    else:
        base = os.path.splitext(os.path.basename(video_path))[0]
        scene_paths = []
        for i, item in enumerate(scenes, 1):
            s = item["start_time"]
            e = item["end_time"]
            
            # Use time values directly - they can be either seconds (float) or "HH:MM:SS" (string)
            # ffmpeg accepts both formats
            out = os.path.join(output_dir, f"{base}_scene{i:03d}.mp4")
            scene_paths.append(out)
            cmd = [ffmpeg_path, '-y', '-ss', str(s), '-to', str(e),
                   '-i', video_path, '-c', 'copy', out]
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            if r.returncode:
                print(f"Error scene {i}: {r.stderr.decode()}")
                raise Exception(f"Error scene {i}: {r.stderr.decode()}")
            else:
                print(f"✅ Saved scene {i}: {out}")
        with open(cache_path, 'w') as f:
            json.dump(scene_paths, f)
        return scene_paths

def extract_video_frames(video_path: str, fps: float = 1.0, max_frames: int = 10) -> List[str]:
    """
    Extract frames from video using ffmpeg and return base64 encoded frames.
    
    Args:
        video_path: Path to the video file
        fps: Frames per second to extract
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of base64 encoded frame strings
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract frames using ffmpeg
        frame_pattern = os.path.join(temp_dir, "frame_%04d.jpg")
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf', f'fps={fps}',
            '-frames:v', str(max_frames),
            '-q:v', '2',  # High quality
            frame_pattern
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            print(f"Error extracting frames from {video_path}")
            return []
        
        # Get all extracted frames
        frame_files = sorted(glob.glob(os.path.join(temp_dir, "frame_*.jpg")))
        
        # Convert frames to base64
        base64_frames = []
        for frame_file in frame_files:
            try:
                with open(frame_file, 'rb') as f:
                    frame_data = f.read()
                    base64_str = base64.b64encode(frame_data).decode('utf-8')
                    base64_frames.append(base64_str)
            except Exception as e:
                print(f"Error processing frame {frame_file}: {e}")
                continue
                
        return base64_frames

def seconds_to_hh_mm_ss(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        String in HH:MM:SS format
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def describe_scene_v2(video_path: str, start_time: float, end_time: float, model: str = "gemini-2.5-flash-preview-05-20", use_cache: bool = True) -> Dict:
    """
    Uses Langchain's ChatOpenAI to describe the content of a video scene.

    Args:
        video_path: Path to the video scene file
        start_time: Start time of the scene in seconds or "HH:MM:SS" format
        end_time: End time of the scene in seconds or "HH:MM:SS" format
        model: Model to use for description
        use_cache: Whether to use cached descriptions

    Returns:
        Dictionary containing the scene description and metadata
    """
    retry_count = 0
    max_retries = 3
    
    # Initialize Langchain ChatOpenAI client
    api_url = os.getenv("VISION_API_URL")
    api_key = os.getenv("VISION_API_KEY")
    
    if not api_url or not api_key:
        raise ValueError("VISION_API_URL and VISION_API_KEY environment variables must be set")
    
    # Extract frames from video
    base64_frames = extract_video_frames(video_path, fps=1.0, max_frames=8)
    
    if not base64_frames:
        raise ValueError(f"Could not extract frames from video: {video_path}")
    
    # Initialize ChatOpenAI with custom base URL
    llm = ChatOpenAI(
        model=model,
        openai_api_base=api_url,
        openai_api_key=api_key,
        temperature=0.1
    )
    
    # Prepare messages with video frames
    content = [
        {"type": "text", "text": "Please describe this video scene in detail based on the following frames."}
    ]
    
    # Add frames as images
    for i, frame_b64 in enumerate(base64_frames):
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame_b64}"
            }
        })
    
    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=content)
    ]
    
    while retry_count < max_retries:
        try:
            response = llm.invoke(messages)
            description_text = response.content
            
            # Try to parse as JSON, fallback to plain text
            try:
                description = json.loads(description_text)
            except json.JSONDecodeError:
                description = {"description": description_text}
            
            break
        except Exception as e:
            print(f"Error on attempt {retry_count + 1}: {e}")
            retry_count += 1
            if retry_count >= max_retries:
                raise e
            time.sleep(1)

    # Convert time to hh:mm:ss format if it's a float (seconds)
    start_time_formatted = seconds_to_hh_mm_ss(start_time) if isinstance(start_time, (int, float)) else start_time
    end_time_formatted = seconds_to_hh_mm_ss(end_time) if isinstance(end_time, (int, float)) else end_time

    # Create response object
    description_data = {
        "description": description,
        "start_time": start_time_formatted,
        "end_time": end_time_formatted,
        # "timestamp": result.get("created", None)
    }

    return description_data

def sample_video_fps(input_path: str, target_fps: float = 1.0, use_cache: bool = True) -> str:
        """
        Sample video to reduce size by taking 1 frame per second.
        
        Args:
            input_path: Path to input video
            target_fps: Target frames per second (default: 1.0)
            use_cache: Whether to use cached results
            
        Returns:
            Path to the sampled video file
        """
        hit, cache_path = get_cache_path(input_path, { "target_fps": target_fps })
        if use_cache and hit:
            if os.path.exists(cache_path):
                return cache_path
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-filter:v', f'fps={target_fps}',
            '-c:v', 'libx264',
            '-crf', '28',  # Higher CRF = lower quality, smaller file
            '-preset', 'ultrafast',
            cache_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return cache_path

def divide_scenes(video_path: str, scenes: List[Dict],
                         use_cache: bool = True) -> List[Dict]:
    """
    Process a video by splitting it into scenes and describing each scene.

    Args:
        video_path: Path to the input video
        use_cache: Whether to use cached results

    Returns:
        List of dictionaries containing scene descriptions
    """

    # fps sampling
    sampled_path = sample_video_fps(video_path, 1.0, use_cache)

    # Split video into scenes
    scene_paths = split_scenes(sampled_path, scenes)
    return scene_paths


def describe_scenes(scene_paths: List[str], scenes: List[Dict], use_cache: bool = True) -> List[Dict]:
    """
    Describe each scene in the given list of scene paths.
    
    Args:
        scene_paths: List of paths to scene video files
        scenes: List of dictionaries with start_time and end_time (in seconds or "HH:MM:SS" format)
        use_cache: Whether to use cached descriptions
        
    Returns:
        List of dictionaries containing scene descriptions
    """
    # Describe each scene
    scene_descriptions = []
    for scene_path, scene in zip(scene_paths, scenes):
        print(f"Describing scene {scene_path}")
        description = describe_scene_v2(scene_path, scene["start_time"], scene["end_time"], use_cache=use_cache)
        scene_descriptions.append(description)

    return scene_descriptions
