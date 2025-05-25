import base64
import json
import os
import subprocess
import sys
import time
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
from openai import OpenAI
from PIL import Image
from prompt.scene_describer import sys_prompt
from qwen_vl_utils import process_vision_info

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

def prepare_message_for_vllm(content_messages):
    """
    The frame extraction logic for videos in `vLLM` differs from that of `qwen_vl_utils`.
    Here, we utilize `qwen_vl_utils` to extract video frames, with the `media_typ`e of the video explicitly set to `video/jpeg`.
    By doing so, vLLM will no longer attempt to extract frames from the input base64-encoded images.
    """
    vllm_messages, fps_list = [], []
    for message in content_messages:
        message_content_list = message["content"]
        if not isinstance(message_content_list, list):
            vllm_messages.append(message)
            continue

        new_content_list = []
        for part_message in message_content_list:
            if 'video' in part_message:
                video_message = [{'content': [part_message]}]
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    video_message, return_video_kwargs=True)
                assert video_inputs is not None, "video_inputs should not be None"
                video_input = (video_inputs.pop()).permute(
                    0, 2, 3, 1).numpy().astype(np.uint8)
                fps_list.extend(video_kwargs.get('fps', []))

                # encode image with base64
                base64_frames = []
                for frame in video_input:
                    img = Image.fromarray(frame)
                    output_buffer = BytesIO()
                    img.save(output_buffer, format="jpeg")
                    byte_data = output_buffer.getvalue()
                    base64_str = base64.b64encode(byte_data).decode("utf-8")
                    base64_frames.append(base64_str)

                part_message = {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
                }
            new_content_list.append(part_message)
        message["content"] = new_content_list
        vllm_messages.append(message)

    return vllm_messages, {'fps': fps_list}

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

def describe_scene_v2(video_path: str, start_time: float, end_time: float, model: str = "Qwen/Qwen2.5-VL-7B-Instruct", use_cache: bool = True) -> Dict:
    """
    Uses the VISION API endpoint to describe the content of a video scene.
    This function has the same interface as describe_scene but sends a POST request to the VISION API endpoint.

    Args:
        video_path: Path to the video scene file
        start_time: Start time of the scene in seconds or "HH:MM:SS" format
        end_time: End time of the scene in seconds or "HH:MM:SS" format
        model: Model to use for description (not used in this version)
        use_cache: Whether to use cached descriptions (not used in this version)

    Returns:
        Dictionary containing the scene description and metadata
    """
    retry_count = 0
    
    api_url = os.getenv("VISION_API_URL")
    api_key = os.getenv("VISION_API_KEY")

    print(api_url)
    endpoint = f"{api_url}/chat"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Please describe this video scene in detail."},
                {"type": "video", "video": video_path},
            ]
        }
    ]

    payload = {"messages": messages}

    while retry_count < 3:
        response = httpx_client.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        try:
            description = json.loads(result['response'])
        except json.JSONDecodeError:
            print(f"Failed to parse description: {result['response']}")
            continue
        break

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

def sample_video_fps(input_path: str, target_fps: float = 1.0, use_cache: bool = True) -> None:
        """
        Sample video to reduce size by taking 1 frame per second.
        
        Args:
            input_path: Path to input video
            output_path: Path to save sampled video
            target_fps: Target frames per second (default: 1.0)
            use_cache: Whether to use cached results
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
