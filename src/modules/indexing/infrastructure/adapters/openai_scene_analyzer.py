import os
import json
import shutil
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.modules.indexing.domain.scene import Scene
from src.modules.indexing.application.ports.scene_analyzer_port import SceneAnalyzerPort
from src.modules.indexing.infrastructure.utils.video_utils import (
    split_video_into_chunks, sample_video_some_fps, 
    extract_frames_from_video, extract_video_chunk_frames
)
from src.modules.indexing.infrastructure.prompt.scene_base import scene_base_prompt
from src.modules.indexing.infrastructure.prompt.scene_describer import scene_describer_prompt
from src.modules.indexing.infrastructure.prompt.scene_merger import scene_merger_prompt
from src.shared.infrastructure.cache.cache import get_cache_path

class OpenAISceneAnalyzer(SceneAnalyzerPort):
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self._check_api_key()

    def _check_api_key(self):
        if "OPENAI_API_KEY" not in os.environ:
             raise EnvironmentError("OPENAI_API_KEY missing - required for OpenAISceneAnalyzer")

    def _init_gpt_model(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=10000,
            max_retries=5,
            api_key=os.environ["OPENAI_API_KEY"]
        )

    def _create_message_with_frames(self, base64_frames: List[str], prompt: str) -> HumanMessage:
        content = [{"type": "text", "text": prompt}]
        for frame in base64_frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
            })
        return HumanMessage(content=content)

    def _analyze_chunk(self, video_path: str, chunk_index: int) -> List[Scene]:
        """Analyze a single video chunk using GPT Vision."""
        print(f"비디오 청크 {chunk_index} 프레임 추출 중: {video_path}")
        base64_frames = extract_frames_from_video(video_path)
        
        if not base64_frames:
            print(f"비디오가 비어있습니다: {video_path}")
            return []

        model = self._init_gpt_model()
        
        messages = [
            SystemMessage(content=scene_base_prompt),
            self._create_message_with_frames(base64_frames, f"이 동영상은 전체 동영상을 나눈 청크 중 {chunk_index + 1}번째 청크입니다. 먼저 비슷한 장면을 찾아 묶어주세요.")
        ]
        
        print(f"청크 {chunk_index} GPT 분석 중...")
        response = model.invoke([*messages])
        
        # Debug save
        with open(f"response_{chunk_index}.txt", "w", encoding="utf-8") as f:
            f.write(str(response.content))

        messages.append(AIMessage(content=response.content))
        response = model.invoke([*messages, SystemMessage(content=scene_describer_prompt), HumanMessage(content="이제 씬을 분석해주세요. 이전 단계에서 나눈 씬을 주의깊게 참고해서 분석해주세요.")])
        
        response_text = str(response.content)
        print(f"청크 {chunk_index} 분석 완료")
        
        try:
            if "```json" in response_text:
                json_part = response_text.split("```json")[1].split("```")[0].strip()
                analysis_result = json.loads(json_part)
            elif "```" in response_text:
                json_part = response_text.split("```")[1].strip()
                analysis_result = json.loads(json_part)
            else:
                analysis_result_json = json.loads(response_text)
                analysis_result = analysis_result_json # Assume it's already list or dict

            if isinstance(analysis_result, dict):
                 analysis_result = [analysis_result]
                 
            return [Scene(**item) for item in analysis_result]
                
        except json.JSONDecodeError:
            raise ValueError(f"GPT API 응답을 JSON으로 파싱할 수 없습니다: {response_text}")

    def _merge_scenes_logic(self, analysis_results: List[List[Scene]], responses: List[Dict[str, Any]]) -> List[Scene]:
        if not analysis_results:
            return []
        
        merged_scenes = []
        for scene in analysis_results[0][:-1]:
            merged_scenes.append(scene.model_copy())
        
        for i in range(len(analysis_results) - 1):
            current_chunk_last_scene = analysis_results[i][-1]
            response = responses[i] if i < len(responses) else {}
            is_same_scene = response.get('is_same_scene', False)
            
            if is_same_scene:
                merged_scene = response.get("merged_scene")
                if merged_scene:
                    merged_scenes.append(Scene(**merged_scene))
                    next_scenes = analysis_results[i + 1][1:]
                else: 
                     # Fallback if merged_scene missing but is_same_scene true
                     merged_scenes.append(current_chunk_last_scene.model_copy())
                     next_scenes = analysis_results[i + 1] # Treat as not merged
                     
                if i == len(analysis_results) - 2:
                    for scene in next_scenes:
                        merged_scenes.append(scene.model_copy())
                else:
                    for scene in next_scenes[:-1]:
                         merged_scenes.append(scene.model_copy())
            else:
                merged_scenes.append(current_chunk_last_scene.model_copy())
                next_scenes = analysis_results[i + 1]
                if i == len(analysis_results) - 2:
                    for scene in next_scenes:
                         merged_scenes.append(scene.model_copy())
                else:
                    for scene in next_scenes[:-1]:
                        merged_scenes.append(scene.model_copy())
                        
        if len(analysis_results) == 1:
            for scene in analysis_results[0]:
                merged_scenes.append(scene.model_copy())
                
        return merged_scenes

    def _merge_scenes(self, video_path: str, analysis_results: List[List[Scene]], chunk_duration: int, fps: int = 1) -> List[Scene]:
        model = self._init_gpt_model()
        responses = []

        for i, this_analysis in enumerate(analysis_results):
            if i == len(analysis_results) - 1:
                continue
            
            boundary_time = chunk_duration * (i + 1)
            next_analysis = analysis_results[i + 1]

            this_base64_frames = extract_video_chunk_frames(video_path, boundary_time - 3, boundary_time, fps)
            next_base64_frames = extract_video_chunk_frames(video_path, boundary_time, boundary_time + 3, fps)
            
            this_prompt = f"다음은 AI가 분석한 씬의 내용입니다:\n{json.dumps(this_analysis[-1].model_dump(), ensure_ascii=False)}\n\n또한 적절한 판단을 위해 영상의 일부분을 첨부합니다."
            next_prompt = f"다음은 AI가 분석한 씬의 내용입니다:\n{json.dumps(next_analysis[0].model_dump(), ensure_ascii=False)}\n\n또한 적절한 판단을 위해 영상의 일부분을 첨부합니다."

            messages = [
                SystemMessage(content=scene_merger_prompt),
                self._create_message_with_frames(this_base64_frames, this_prompt),
                self._create_message_with_frames(next_base64_frames, next_prompt),
                HumanMessage(content="이제 이 두 씬이 같은 씬인지 판단해주세요.")
            ]

            response = model.invoke(messages)
            content = str(response.content)
            if "```json" in content:
                json_part = content.split("```json")[1].split("```")[0].strip()
            else:
                json_part = content
            response_json = json.loads(json_part)
            responses.append(response_json)

        return self._merge_scenes_logic(analysis_results, responses)

    def analyze_scenes(self, video_path: str, chunk_duration: int = 300) -> List[Scene]:
        print(f"Forced OpenAI Analysis on: {video_path}")
        chunk_paths = split_video_into_chunks(video_path, chunk_duration=chunk_duration)
        temp_chunks_dir = os.path.dirname(chunk_paths[0]) if chunk_paths else None
        
        try:
            analysis_results = []
            for i, chunk_path in enumerate(chunk_paths):
                # FPS Sampling
                sampled_path = sample_video_some_fps(chunk_path, fps=1)
                
                # Caching Logic for Analysis
                cache_args = {
                    "operation": "openai_analysis", 
                    "chunk_index": i, 
                    "model": self.model_name, 
                    "video_path": sampled_path # cache by chunk content not original video path? 
                    # Original logic used full video_path + chunk_index. 
                    # sampled_path is unique per chunk content if temp name is unique.
                    # But get_cache_path expects video_path.
                }
                # Let's use the original video path and chunk index for consistency with existing cache logic
                # But here we are iterating chunks.
                # Actually, in SceneAnalyzer, it used `video_path` (original) and `chunk_index`.
                # I don't have original video_path easily unless passed or derived.
                # `video_path` argument IS the original video path.
                
                cache_key_path = video_path # Key for cache
                
                cache_args = {
                    "operation": "openai_analysis_v2", # v2 to avoid conflicts with old cache if format changed
                    "chunk_index": i,
                    "model": self.model_name,
                    "video_path": cache_key_path
                }
                
                cache_exists, cache_path = get_cache_path(cache_key_path, cache_args)
                json_cache_path = cache_path + ".json"
                
                if os.path.exists(json_cache_path):
                    with open(json_cache_path, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    analysis = [Scene(**item) for item in cached_data]
                    print(f"Chunk {i} loaded from cache")
                else:
                    analysis = self._analyze_chunk(sampled_path, i)
                    # Cache it
                    with open(json_cache_path, 'w', encoding='utf-8') as f:
                        json.dump([s.model_dump(mode='json') for s in analysis], f, ensure_ascii=False, indent=2)
                
                if len(analysis) == 0:
                    continue
                analysis_results.append(analysis)
            
            # Merge
            return self._merge_scenes(video_path, analysis_results, chunk_duration, 1)
            
        finally:
            if temp_chunks_dir and os.path.exists(temp_chunks_dir):
                shutil.rmtree(temp_chunks_dir)
