#!/usr/bin/env python3
"""
Test script for refactored scene analysis functionality.
Usage: python test_scene_analysis.py <video_path>
"""

import json
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from src.server.indexing.scene_analyzer import analyze_video_scenes


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_scene_analysis.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    print(f"Testing refactored scene analysis with video: {video_path}")
    print("=" * 60)
    print("Features:")
    print("  - Separated Gemini API calls to gemini_scene_analysis.py")
    print("  - Using temp files for video processing")
    print("  - Caching enabled with cache_manager.py")
    print("=" * 60)
    
    try:
        # Test the scene analysis
        results = analyze_video_scenes(video_path, chunk_duration=300)
        
        print("\n=== 분석 결과 ===")
        print(f"총 {len(results)}개의 청크가 분석되었습니다.")
        print("\n=== 캐시 정보 ===")
        print("두 번째 실행 시 캐시된 결과를 사용하여 더 빠르게 실행됩니다.")
        print("캐시는 시스템의 임시 디렉토리에 저장됩니다.")
        
        with open("results.json", "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error during scene analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 