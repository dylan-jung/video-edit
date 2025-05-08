import base64
import json
import os
import subprocess
import sys
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
from openai import OpenAI
from PIL import Image
from prompt.scene_describer import sys_prompt
from qwen_vl_utils import process_vision_info

from src.server.utils.cache_manager import get_cache_path, get_file_id

# TODO: Code server 세션 쿠키 때문에 만들어 둠
cookies = httpx.Cookies()
cookies.set("code-server-session",
            """%24argon2id%24v%3D19%24m%3D65536%2Ct%3D3%2Cp%3D4%24PH%2B%2FvHn4uG3JWvs5wWFSWQ%24Sl2b696Jv%2FO528n5NVQlakXBkt5MTfV8ODf8HiZAww0""",
            domain=".1a40432.tunnel.myubai.uos.ac.kr",
            path="/")

httpx_client = httpx.Client(
    cookies=cookies,
)

def split_scenes(video_path: str, scenes: List[Dict], ffmpeg_path: str = "ffmpeg") -> List[str]:
    """
    Split a video into scenes based on provided timestamps.

    Args:
        video_path: Path to the input video file
        scenes: List of (start_time, end_time) tuples in seconds
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


def describe_scene(video_path: str,
                   start_time: float,
                   end_time: float,
                   model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                   use_cache: bool = True) -> Dict:
    """
    Uses OpenAI API to describe the content of a video scene.

    Args:
        video_path: Path to the video scene file
        model: Model to use for description
        use_cache: Whether to use cached descriptions

    Returns:
        Dictionary containing the scene description and metadata
    """
    # Check cache first

    cache_args = {"model": model}
    cache_exists, cache_path = get_cache_path(video_path, cache_args)

    if use_cache and cache_exists:
        print(f"Using cached description: {cache_path}")
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"Error reading cache, regenerating description")

    # Create OpenAI client
    api_url = os.getenv("VISION_API_URL")
    api_key = os.getenv("VISION_API_KEY")
    client = OpenAI(base_url=api_url, api_key=api_key, http_client=httpx_client)
    
    try:
        # Read the video file and convert to base64
        total_pixels = 20480 * 28 * 28
        min_pixels = 16 * 28 * 28
        max_tokens = 2048

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Please describe this video scene in detail."},
                {"type": "video", "video": video_path, "total_pixels": total_pixels,
                    "min_pixels": min_pixels},
            ]
            }
        ]

        vllm_messages, video_kwargs = prepare_message_for_vllm(messages)

        # Print the request details
        response = client.chat.completions.create(
            model=model,
            messages=vllm_messages,
            max_tokens=max_tokens,
            extra_body={
                "mm_processor_kwargs": video_kwargs
            }
        )

        # Extract description from response
        description = response.choices[0].message.content
        try:
            description = json.loads(description)
        except json.JSONDecodeError:
            print(f"Failed to parse description: {description}")
            raise Exception(f"Failed to parse description: {description}")

        # Create response object
        description_data = {
            "video_id": get_file_id(video_path),
            "description": description,
            "start_time": start_time,
            "end_time": end_time,
            "timestamp": response.created
        }

        # Save to cache
        if use_cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(description_data, f)

        return description_data

    except Exception as e:
        raise e

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
def process_video_scenes(video_path: str, scenes: List[Dict],
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

    # Describe each scene
    scene_descriptions = []
    for scene_path, scene in zip(scene_paths, scenes):
        print(f"Describing scene {scene_path}")
        description = describe_scene(scene_path, scene["start_time"], scene["end_time"], use_cache=use_cache)
        scene_descriptions.append(description)

    return scene_descriptions
