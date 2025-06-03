#!/usr/bin/env python3
"""
GPT Scene Analyzer 테스트 스크립트
"""

import json
import os

from src.server.indexing.gpt_scene_analyzer import analyze_video_with_gpt
from src.server.indexing.scene_analyzer import analyze_video_scenes


def test_gpt_scene_analyzer():
    """GPT Scene Analyzer 기본 기능 테스트"""
    
    # 환경 변수 확인
    if "OPENAI_API_KEY" not in os.environ:
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("export OPENAI_API_KEY='your-api-key-here' 를 실행하세요.")
        return
    
    # 테스트할 비디오 파일들 확인
    test_videos = [
        "video.mp4",
        "small.mp4",
        "1.mp4",
        "2.mp4",
        "3.mp4"
    ]
    
    available_videos = [video for video in test_videos if os.path.exists(video)]
    
    if not available_videos:
        print("❌ 테스트할 비디오 파일이 없습니다.")
        print(f"다음 파일 중 하나를 준비하세요: {test_videos}")
        return
    
    print(f"✅ 사용 가능한 비디오 파일: {available_videos}")
    
    # 첫 번째 비디오로 테스트
    test_video = available_videos[0]
    print(f"\n🎬 '{test_video}' 파일로 GPT Scene Analysis 테스트 시작...")
    
    try:
        # 기본 scene analysis 테스트
        results = analyze_video_scenes(video_path=test_video, model="gpt-4.1-mini", chunk_duration=180)
        
        print("✅ GPT Scene Analysis 성공!")
        print("📊 분석 결과:")
        print(json.dumps(results, ensure_ascii=False, indent=2))
        
        # 결과를 파일로 저장
        output_file = f"gpt_analysis_result_{test_video.replace('.mp4', '')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 결과가 '{output_file}'에 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 기본 분석 실패: {str(e)}")
        return

if __name__ == "__main__":
    test_gpt_scene_analyzer()