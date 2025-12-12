import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from src.modules.indexing.domain.scene import Scene

from .gemini_scene_analyzer import \
    analyze_video_with_gemini
from .gpt_scene_analyzer import (
    analyze_video_with_gpt, merge_scenes)
from src.shared.infrastructure.cache.cache import get_cache_path
from src.shared.infrastructure.video import sample_video_some_fps


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


def analyze_video_chunk_with_gemini_cached(video_path: str, chunk_index: int, model: str = "gemini-2.0-flash") -> List[Scene]:
    """
    Analyze a video chunk using Gemini with caching support.
    """
    # Check cache first for analysis results
    cache_args = {
        "operation": "gemini_analysis",
        "chunk_index": chunk_index,
        "model": model,
        "video_path": video_path
    }
    cache_exists, cache_path = get_cache_path(video_path, cache_args)
    
    # Cache path for JSON results (change extension)
    json_cache_path = cache_path.replace(os.path.splitext(cache_path)[1], ".json")
    
    if os.path.exists(json_cache_path):
        try:
            with open(json_cache_path, 'r', encoding='utf-8') as f:
                cached_result_json = json.load(f)
            print(f"청크 {chunk_index} Gemini 분석 결과 캐시에서 로드됨")
            
            # Convert dict/list to List[Scene]
            if isinstance(cached_result_json, dict):
                 cached_result_json = [cached_result_json]
            return [Scene(**item) for item in cached_result_json]
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"청크 {chunk_index} 캐시 파일 손상됨, 새로 분석")
    
    # Analyze with Gemini if not in cache
    try:
        analysis_result = analyze_video_with_gemini(video_path, chunk_index, model)
        
        # Save to cache
        with open(json_cache_path, 'w', encoding='utf-8') as f:
            json.dump([scene.model_dump(mode='json') for scene in analysis_result], f, ensure_ascii=False, indent=2)
        print(f"청크 {chunk_index} Gemini 분석 결과 캐시 저장됨")
        
        return analysis_result
    except Exception as e:
        print(f"청크 {chunk_index} Gemini 분석 실패: {str(e)}")
        raise


def analyze_video_scenes(video_path: str, model: str = "gemini-2.0-flash", chunk_duration: int = 300) -> List[Scene]:
    """
    Main function to analyze video scenes.
    1. Split video into 5-minute chunks (using temp files and caching)
    2. Sample each chunk at 1 fps (using temp files and caching)
    3. Analyze each chunk with Gemini (using caching)
    4. Return combined results as a list
    """
    print(f"동영상 장면 분석 시작: {video_path}")
    temp_chunks_dir = None
    
    try:
        chunk_paths = split_video_into_chunks(video_path, chunk_duration=chunk_duration)
        print(f"총 {len(chunk_paths)}개의 청크로 분할 완료")
        
        if chunk_paths:
            temp_chunks_dir = os.path.dirname(chunk_paths[0])
        
        # Step 2 & 3: Sample at 1 fps and analyze each chunk
        analysis_results = []
        
        for i, chunk_path in enumerate(chunk_paths):
            print(f"\n2. 청크 {i+1}/{len(chunk_paths)} 처리 중...")
            
            # Sample at 1 fps
            sampled_path = sample_video_some_fps(chunk_path, fps=1)
            print(f"FPS 샘플링 완료: {sampled_path}")
            
            # Analyze with Gemini (with caching)
            if "gemini" in model:
                analysis: List[Scene] = analyze_video_chunk_with_gemini_cached(sampled_path, i, model)
                if len(analysis) == 0:
                    print(f"청크 {i+1} 분석 결과가 비어있습니다.")
                    continue
                analysis_results.append(analysis)
            else:
                analysis: List[Scene] = analyze_video_with_gpt(sampled_path, i, model)
                if len(analysis) == 0:
                    print(f"청크 {i+1} 분석 결과가 비어있습니다.")
                    continue
                analysis_results.append(analysis)
            print(f"청크 {i+1} 분석 완료")
        
        # Step 4: Merge scenes
        merged_scenes = merge_scenes(video_path, analysis_results, chunk_duration, 1, model)
        with open("merged_scenes.json", "w", encoding="utf-8") as f:
            json.dump([scene.model_dump(mode='json') for scene in merged_scenes], f, ensure_ascii=False, indent=2)

        print(f"\n전체 동영상 장면 분석 완료. 총 {len(merged_scenes)}개 청크 분석됨")
        return merged_scenes
    
    finally:
        # Clean up temp directory
        if temp_chunks_dir and os.path.exists(temp_chunks_dir):
            try:
                shutil.rmtree(temp_chunks_dir)
                print("임시 청크 파일들 정리 완료")
            except Exception as e:
                print(f"임시 파일 정리 실패: {str(e)}")

# For backward compatibility - clustering based analysis
def analyze_video_scenes_clustering(video_path: str) -> List[Scene]:
    """
    Placeholder for clustering-based video scene analysis.
    Currently redirects to Gemini-based analysis.
    """
    print("클러스터링 기반 분석이 요청되었으나 Gemini 기반 분석으로 대체됩니다.")
    return analyze_video_scenes(video_path) 