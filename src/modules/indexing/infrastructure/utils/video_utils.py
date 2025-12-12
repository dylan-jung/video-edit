import os
import subprocess
import tempfile
import shutil
from typing import List, Tuple
from src.shared.infrastructure.video import extract_frames_from_video, extract_video_chunk_frames, sample_video_some_fps
from src.shared.infrastructure.cache.cache import get_cache_path

__all__ = ['extract_frames_from_video', 'extract_video_chunk_frames', 'sample_video_some_fps', 'split_video_into_chunks']

def split_video_into_chunks(video_path: str, chunk_duration: int = 300) -> List[str]:
    """
    Split video into chunks of specified duration (default 5 minutes = 300 seconds).
    Returns list of chunk file paths in temp directory.
    """
    chunk_paths = []
    
    # Get video duration
    cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    total_duration = float(result.stdout.strip())
    
    # Calculate number of chunks
    num_chunks = int(total_duration / chunk_duration) + (1 if total_duration % chunk_duration > 0 else 0)
    
    # Create temp directory for chunks
    temp_dir = tempfile.mkdtemp(prefix="video_chunks_")
    
    for i in range(num_chunks):
        start_time = i * chunk_duration
        chunk_filename = f"chunk_{i:03d}.mp4"
        
        # Check cache first
        cache_args = {
            "operation": "split_chunk",
            "chunk_index": i,
            "start_time": start_time,
            "duration": chunk_duration,
            "video_path": video_path
        }
        cache_exists, cache_path = get_cache_path(video_path, cache_args)
        
        if cache_exists:
            # Copy from cache to temp directory
            chunk_path = os.path.join(temp_dir, chunk_filename)
            shutil.copy2(cache_path, chunk_path)
            chunk_paths.append(chunk_path)
            print(f"청크 {i} 캐시에서 로드됨: {chunk_path}")
            continue
        
        # Create chunk if not in cache
        chunk_path = os.path.join(temp_dir, chunk_filename)
        cmd = [
            "ffmpeg", "-i", video_path,
            "-ss", str(start_time),
            "-t", str(chunk_duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            chunk_path
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        if os.path.exists(chunk_path):
            # Save to cache
            shutil.copy2(chunk_path, cache_path)
            chunk_paths.append(chunk_path)
            print(f"청크 {i} 생성 및 캐시 저장됨: {chunk_path}")
        else:
            print(f"청크 {i} 생성 실패")
    
    return chunk_paths
