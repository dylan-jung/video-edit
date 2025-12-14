import logging
import os
import shutil
import subprocess
import tempfile
from typing import List

from src.modules.indexing.domain.scene import Scene
from src.modules.indexing.infrastructure.scene_analyzer.gpt_scene_analyzer import GPTSceneAnalyzer
from src.shared.infrastructure.video.service import sample_video_some_fps

logger = logging.getLogger(__name__)

def split_video_into_chunks(video_path: str, chunk_duration: int = 300) -> List[str]:
    """
    Split video into chunks of specified duration.
    """
    chunk_paths = []
    
    # Get duration
    cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        total_duration = float(result.stdout.strip())
    except ValueError:
        logger.error(f"Failed to get video duration for {video_path}")
        raise

    num_chunks = int(total_duration / chunk_duration) + (1 if total_duration % chunk_duration > 0 else 0)
    temp_dir = tempfile.mkdtemp(prefix="video_chunks_")
    
    for i in range(num_chunks):
        start_time = i * chunk_duration
        chunk_filename = f"chunk_{i:03d}.mp4"
        chunk_path = os.path.join(temp_dir, chunk_filename)
        
        cmd = [
            "ffmpeg", "-i", video_path,
            "-ss", str(start_time),
            "-t", str(chunk_duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            chunk_path
        ]
        subprocess.run(cmd, capture_output=True)
        
        if os.path.exists(chunk_path):
            chunk_paths.append(chunk_path)
            logger.info(f"Chunk {i} created")
        else:
            logger.error(f"Failed to create chunk {i}")

    return chunk_paths

def analyze_video_scenes(video_path: str, model: str = "gpt-4o-mini", chunk_duration: int = 300) -> List[Scene]:
    """
    Analyze video scenes
    """
    logger.info(f"Starting Scene Analysis for {video_path}")
    analyzer = GPTSceneAnalyzer(model_name=model)
    temp_chunks_dir = None
    
    try:
        # 1. Split Video
        chunk_paths = split_video_into_chunks(video_path, chunk_duration)
        if chunk_paths:
            temp_chunks_dir = os.path.dirname(chunk_paths[0])

        # 2. Analyze Chunks
        analysis_results = []
        for i, chunk_path in enumerate(chunk_paths):
            logger.info(f"Processing chunk {i+1}/{len(chunk_paths)}")
            
            # Sample 1 FPS
            sampled_path = sample_video_some_fps(chunk_path, fps=1)

            # Analyze
            chunk_analysis = analyzer.analyze_chunk(sampled_path, i)
            if chunk_analysis:
                 analysis_results.append(chunk_analysis)
            else:
                logger.warning(f"Chunk {i} analysis returned empty")
                analysis_results.append([])

        # 3. Merge Scenes
        merged_raw_scenes = analyzer.merge_chunks(video_path, analysis_results, chunk_duration)
        
        # Convert to Domain Model
        scenes = [Scene(**s) for s in merged_raw_scenes]
        logger.info(f"Scene Analysis Complete. Total Scenes: {len(scenes)}")
        return scenes

    finally:
        if temp_chunks_dir and os.path.exists(temp_chunks_dir):
            shutil.rmtree(temp_chunks_dir)