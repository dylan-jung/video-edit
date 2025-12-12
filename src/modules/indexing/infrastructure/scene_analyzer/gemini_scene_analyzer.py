import json
import os
import time
from typing import Any, Dict, List
from src.modules.indexing.domain.scene import Scene

import google.generativeai as genai

from ..prompt.scene_base import scene_base_prompt
from ..prompt.scene_describer import scene_describer_prompt


def configure_gemini():
    """Configure Gemini API with the API key from environment variables."""
    if "GOOGLE_API_KEY" not in os.environ:
        raise EnvironmentError(
            "환경변수 GOOGLE_API_KEY 가 설정되어 있지 않습니다. "
            "https://ai.google.dev/ 에서 API 키를 발급받아 설정하세요."
        )
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def init_model(model: str):
    generation_config = {
        "max_output_tokens": 10000,
        "candidate_count": 1,
        "temperature": 0.1,
        "top_p": 0.95,
        "response_mime_type": "application/json",
    }
    safety_settings = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]
    return genai.GenerativeModel(model, safety_settings=safety_settings, generation_config=generation_config)


def analyze_video_with_gemini(video_path: str, chunk_index: int, model: str = "gemini-2.0-flash") -> List[Scene]:
    """
    Analyze a video chunk using Gemini and return the analysis results.
    This function handles all Gemini API interactions in two stages like GPT version.
    """
    configure_gemini()

    # Upload video file using File API
    print(f"비디오 청크 {chunk_index} 업로드 중: {video_path}")
    video_file = genai.upload_file(path=video_path)
    print(f"업로드 완료. 파일 URI: {video_file.uri}")

    # Wait for the file to be processed
    while video_file.state.name == "PROCESSING":
        print(f"청크 {chunk_index} 파일 처리 중...")
        time.sleep(2)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(f"비디오 파일 처리 실패: {video_file.state.name}")

    # Initialize the model
    model_instance = init_model(model=model)

    # Stage 1: Scene grouping using scene_base_prompt
    print(f"청크 {chunk_index} 1단계: 씬 그룹화 분석 중...")
    first_prompt = f"""
    이 동영상은 전체 동영상을 5분씩 나눈 청크 중 {chunk_index + 1}번째 청크입니다.
    먼저 비슷한 장면을 찾아 묶어주세요.
    """
    
    first_response = model_instance.generate_content([
        scene_base_prompt,
        first_prompt,
        video_file
    ])
    
    # Save first response for debugging
    with open(f"response_{chunk_index}.txt", "w", encoding="utf-8") as f:
        f.write(first_response.text)
    
    print(f"청크 {chunk_index} 1단계 완료")
    
    # Stage 2: Detailed scene analysis using scene_describer_prompt
    print(f"청크 {chunk_index} 2단계: 상세 씬 분석 중...")
    second_prompt = f"""
    이제 씬을 분석해주세요. 이전 단계에서 나눈 씬을 주의깊게 참고해서 분석해주세요.
    
    이전 단계 결과:
    {first_response.text}
    
    JSON 형식으로 응답해주세요.
    """
    
    second_response = model_instance.generate_content([
        scene_base_prompt,
        first_prompt,
        first_response.text,
        scene_describer_prompt,
        second_prompt,
        video_file
    ])
    
    # Clean up: delete the uploaded file
    genai.delete_file(video_file.name)
    print(f"청크 {chunk_index} 업로드된 파일 삭제 완료")
    
    try:
        if "```json" in second_response.text:
            analysis_result_json = json.loads(second_response.text.split("```json")[1].split("```")[0])
        else:
            analysis_result_json = json.loads(second_response.text)
        
        # Ensure it is a list
        if isinstance(analysis_result_json, dict):
             analysis_result_json = [analysis_result_json]
             
        analysis_result = [Scene(**item) for item in analysis_result_json]
    except json.JSONDecodeError:
        raise ValueError(f"Gemini API 응답을 JSON으로 파싱할 수 없습니다: {second_response.text}")
    except Exception as e:
        raise ValueError(f"Scene 모델 변환 실패: {str(e)}")

    return analysis_result
