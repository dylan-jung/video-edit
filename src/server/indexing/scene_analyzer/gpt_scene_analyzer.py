import base64
import json
import os
from typing import Any, Dict, List

import cv2
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.server.indexing.prompt.scene_base import scene_base_prompt
from src.server.indexing.prompt.scene_describer import scene_describer_prompt
from src.server.indexing.prompt.scene_merger import scene_merger_prompt
from src.server.utils.video_control import (extract_frames_from_video,
                                            extract_video_chunk_frames)


def init_gpt_model(model_name: str = "gpt-4o", temperature: float = 0.1) -> ChatOpenAI:
    """
    Initialize GPT model with vision capabilities.
    
    Args:
        model_name: Model name (default: gpt-4o for vision capabilities)
        temperature: Model temperature for creativity
    
    Returns:
        Configured ChatOpenAI instance
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError(
            "환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다. "
            "OpenAI API 키를 설정하세요."
        )
    
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=10000,
        max_retries=5,
        api_key=os.environ["OPENAI_API_KEY"]
    )

def create_message_with_frames(base64_frames: List[str], prompt: str) -> HumanMessage:
    """
    Create a message with frames for GPT analysis.
    
    Args:
        base64_frames: List of base64 encoded frames
        chunk_index: Index of the video chunk
    
    Returns:
        HumanMessage with text and images
    """
    # Create the text prompt
    prompt_text = f"""
    {prompt}
    """
    
    # Create content list starting with text
    content = [
        {
            "type": "text",
            "text": prompt_text
        }
    ]
    
    for frame in base64_frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame}"
            }
        })
    
    return HumanMessage(content=content)


def analyze_video_with_gpt(video_path: str, chunk_index: int, model_name: str = "gpt-4o") -> Dict[str, Any]:
    """
    Analyze a video chunk using GPT Vision and return the analysis results.
    This function handles all GPT API interactions.
    
    Args:
        video_path: Path to the video file
        chunk_index: Index of the video chunk
        model_name: GPT model to use (default: gpt-4o)
    
    Returns:
        Dictionary containing the analysis results
    """
    try:
        # Extract frames from video
        print(f"비디오 청크 {chunk_index} 프레임 추출 중: {video_path}")
        base64_frames = extract_frames_from_video(video_path)
        
        if len(base64_frames) == 0:
            print(f"비디오가 비어있습니다: {video_path}")
            return []

        # Initialize GPT model
        model = init_gpt_model(model_name=model_name)
        
        messages = [
            SystemMessage(content=scene_base_prompt),
            create_message_with_frames(base64_frames, f"이 동영상은 전체 동영상을 나눈 청크 중 {chunk_index + 1}번째 청크입니다. 먼저 비슷한 장면을 찾아 묶어주세요.")
        ]
        
        # Send request to GPT
        print(f"청크 {chunk_index} GPT 분석 중...")
        response = model.invoke([*messages])
        with open(f"response_{chunk_index}.txt", "w", encoding="utf-8") as f:
            f.write(response.content)

        messages.append(AIMessage(content=response.content))
        response = model.invoke([*messages, SystemMessage(content=scene_describer_prompt), HumanMessage(content="이제 씬을 분석해주세요. 이전 단계에서 나눈 씬을 주의깊게 참고해서 분석해주세요.")])
        
        # Parse response
        response_text = response.content
        print(f"청크 {chunk_index} 분석 완료")
        
        # Try to parse JSON response
        try:
            # Handle potential markdown formatting
            if "```json" in response_text:
                json_part = response_text.split("```json")[1].split("```")[0].strip()
                analysis_result = json.loads(json_part)
            elif "```" in response_text:
                # Handle other code block formats
                json_part = response_text.split("```")[1].strip()
                analysis_result = json.loads(json_part)
            else:
                analysis_result = json.loads(response_text)
                
        except json.JSONDecodeError:
            raise ValueError(f"GPT API 응답을 JSON으로 파싱할 수 없습니다: {response_text}")
        
        return analysis_result
        
    except Exception as e:
        print(f"청크 {chunk_index} 분석 중 오류 발생: {str(e)}")
        raise

def _merge_scenes(analysis_results: List[Dict[str, Any]], responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge scenes across video chunks based on boundary analysis.
    
    Args:
        analysis_results: List of analysis results for each chunk
        responses: List of boundary comparison results between chunks
    
    Returns:
        List of merged scenes
    """
    if not analysis_results:
        return []
    
    merged_scenes = []
    
    # 첫 번째 청크의 모든 씬을 시작점으로 추가 (마지막 씬 제외)
    for scene in analysis_results[0][:-1]:
        merged_scenes.append(scene.copy())
    
    # 각 청크 경계를 처리
    for i in range(len(analysis_results) - 1):
        current_chunk_last_scene = analysis_results[i][-1]  # 현재 청크의 마지막 씬
        next_chunk_first_scene = analysis_results[i + 1][0]  # 다음 청크의 첫 번째 씬
        
        # 경계 씬이 같은지 판단
        is_same_scene = False
        if i < len(responses):
            response = responses[i]
            is_same_scene = response.get('is_same_scene', False)
        
        if is_same_scene:
            # 씬을 병합: 현재 청크의 마지막 씬과 다음 청크의 첫 번째 씬을 합침
            merged_scene = response.get("merged_scene", None)
            if merged_scene is None:
                raise ValueError(f"병합된 씬이 없습니다: {response}")
            
            # 병합된 씬 추가
            merged_scenes.append(merged_scene)
            
            # 다음 청크의 나머지 씬들 추가 (첫 번째 씬 제외, 마지막 씬은 조건부)
            next_scenes = analysis_results[i + 1][1:]
            if i == len(analysis_results) - 2:  # 마지막 청크인 경우 모든 씬 추가
                for scene in next_scenes:
                    merged_scenes.append(scene.copy())
            else:  # 마지막 청크가 아닌 경우 마지막 씬 제외
                for scene in next_scenes[:-1]:
                    merged_scenes.append(scene.copy())
        else:
            # 씬을 병합하지 않음: 각각 별도 씬으로 추가
            merged_scenes.append(current_chunk_last_scene.copy())
            
            # 다음 청크의 씬들 추가
            next_scenes = analysis_results[i + 1]
            if i == len(analysis_results) - 2:  # 마지막 청크인 경우 모든 씬 추가
                for scene in next_scenes:
                    merged_scenes.append(scene.copy())
            else:  # 마지막 청크가 아닌 경우 마지막 씬 제외
                for scene in next_scenes[:-1]:
                    merged_scenes.append(scene.copy())
    
    # 마지막 청크가 하나뿐인 경우 처리
    if len(analysis_results) == 1:
        for scene in analysis_results[0]:
            merged_scenes.append(scene.copy())
    
    return merged_scenes

def merge_scenes(video_path: str, analysis_results: List[Dict[str, Any]], chunk_duration: int = 300, fps: int = 1, model_name: str = "gpt-4o") -> List[Dict[str, Any]]:
    """
    Merge scenes from analysis results.
    """
    model = init_gpt_model(model_name=model_name)
    responses = []

    for i, this_analysis in enumerate(analysis_results):
        if i == len(analysis_results) - 1:
            continue
        
        boundary_time = chunk_duration * (i + 1)
        next_analysis = analysis_results[i + 1]

        this_base64_frames = extract_video_chunk_frames(video_path, boundary_time - 3, boundary_time, fps)
        next_base64_frames = extract_video_chunk_frames(video_path, boundary_time, boundary_time + 3, fps)
        
        this_prompt = f"다음은 AI가 분석한 씬의 내용입니다:\n{json.dumps(this_analysis[-1], ensure_ascii=False)}\n\n또한 적절한 판단을 위해 영상의 일부분을 첨부합니다."
        next_prompt = f"다음은 AI가 분석한 씬의 내용입니다:\n{json.dumps(next_analysis[0], ensure_ascii=False)}\n\n또한 적절한 판단을 위해 영상의 일부분을 첨부합니다."

        messages = [
            SystemMessage(content=scene_merger_prompt),
            create_message_with_frames(this_base64_frames, this_prompt),
            create_message_with_frames(next_base64_frames, next_prompt),
            HumanMessage(content="이제 이 두 씬이 같은 씬인지 판단해주세요.")
        ]

        response = model.invoke(messages)
        if "```json" in response.content:
            json_part = response.content.split("```json")[1].split("```")[0].strip()
        else:
            json_part = response.content
        response_json = json.loads(json_part)
        responses.append(response_json)

    merged_scenes = _merge_scenes(analysis_results, responses)
    return merged_scenes