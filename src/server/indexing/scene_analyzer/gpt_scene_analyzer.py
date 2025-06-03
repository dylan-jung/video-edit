import base64
import json
import os
from typing import Any, Dict, List

import cv2
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.server.indexing.prompt.scene_base import scene_base_prompt
from src.server.indexing.prompt.scene_describer import scene_describer_prompt


def extract_frames_from_video(video_path: str) -> List[str]:
    """
    Extract frames from video and convert to base64 encoded strings.
    
    Args:
        video_path: Path to the video file
        sample_rate: Extract every nth frame (default: 30, roughly 1 frame per second for 30fps video)
    
    Returns:
        List of base64 encoded frame strings
    """
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
    
    base64_frames = []
    frame_count = 0
    
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
            
        # Encode frame to JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        # Convert to base64
        base64_frame = base64.b64encode(buffer).decode("utf-8")
        base64_frames.append(base64_frame)
        
        frame_count += 1
    
    video.release()
    print(f"{len(base64_frames)} 프레임을 추출했습니다.")
    return base64_frames


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
        api_key=os.environ["OPENAI_API_KEY"]
    )

def create_message_with_frames(base64_frames: List[str], prompt: str, chunk_index: int) -> HumanMessage:
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
    이 동영상은 전체 동영상을 나눈 청크 중 {chunk_index + 1}번째 청크입니다.
    
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
        
        if not base64_frames:
            raise ValueError(f"비디오에서 프레임을 추출할 수 없습니다: {video_path}")
        
        # Initialize GPT model
        model = init_gpt_model(model_name=model_name)
        
        messages = [
            SystemMessage(content=scene_base_prompt),
            create_message_with_frames(base64_frames, "먼저 비슷한 장면을 찾아 묶어주세요.", chunk_index)
        ]
        # # Create message with frames
        # message = create_message_with_frames(base64_frames, chunk_index)
        
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
