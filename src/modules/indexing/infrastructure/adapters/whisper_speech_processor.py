import os
import json
import openai
from typing import List, Dict, Any

from src.modules.indexing.application.ports.speech_processor_port import SpeechProcessorPort
from src.modules.indexing.infrastructure.utils.audio_utils import split_audio_chunks, format_timestamp, get_audio_duration

class WhisperSpeechProcessor(SpeechProcessorPort):
    def __init__(self, transcription_model: str = "whisper-1", enhancement_model: str = "gpt-4o-mini"):
        self.transcription_model = transcription_model
        self.enhancement_model = enhancement_model
        if "OPENAI_API_KEY" not in os.environ:
             raise EnvironmentError("OPENAI_API_KEY missing - required for WhisperSpeechProcessor")

    def _transcribe_chunk(self, chunk_path: str) -> str:
        with open(chunk_path, 'rb') as audio_file:
            response = openai.audio.transcriptions.create(
                model=self.transcription_model,
                file=audio_file,
                response_format="text",
                language="ko"
            )
            return str(response)

    def _create_semantic_chunks(self, segments: List[Dict[str, Any]], semantic_chunk_size: int = 30) -> List[Dict[str, Any]]:
        if not segments:
            return []
        
        semantic_chunks = []
        current_chunk = {
            'segments': [],
            'start_time': 0,
            'end_time': 0,
            'text_parts': []
        }
        
        for segment in segments:
            segment_start = segment['start_seconds']
            segment_end = segment['end_seconds']
            
            if (current_chunk['segments'] and 
                segment_start - current_chunk['start_time'] > semantic_chunk_size):
                
                if current_chunk['segments']:
                    chunk_data = self._finalize_semantic_chunk(current_chunk, len(semantic_chunks))
                    semantic_chunks.append(chunk_data)
                
                current_chunk = {
                    'segments': [segment],
                    'start_time': segment_start,
                    'end_time': segment_end,
                    'text_parts': [segment['text']]
                }
            else:
                if not current_chunk['segments']:
                    current_chunk['start_time'] = segment_start
                
                current_chunk['segments'].append(segment)
                current_chunk['end_time'] = segment_end
                current_chunk['text_parts'].append(segment['text'])
        
        if current_chunk['segments']:
            chunk_data = self._finalize_semantic_chunk(current_chunk, len(semantic_chunks))
            semantic_chunks.append(chunk_data)
        
        return semantic_chunks

    def _finalize_semantic_chunk(self, chunk_data: Dict[str, Any], chunk_id: int) -> Dict[str, Any]:
        combined_text = ' '.join(chunk_data['text_parts'])
        start_formatted = format_timestamp(chunk_data['start_time'])
        end_formatted = format_timestamp(chunk_data['end_time'])
        
        return {
            'chunk_id': chunk_id,
            'text': combined_text,
            'start_time': start_formatted,
            'end_time': end_formatted,
            'start_seconds': chunk_data['start_time'],
            'end_seconds': chunk_data['end_time'],
            'duration': chunk_data['end_time'] - chunk_data['start_time'],
            'duration_range': f"{start_formatted} - {end_formatted}",
            'segment_count': len(chunk_data['segments']),
            'language': 'ko'  # metadata not passed deeply, assuming ko
        }

    def process_audio(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Process audio file (split, transcribe) and return SEMANTIC chunks.
        Note: Original logic returned 'transcription_data', but we return chunks for better flow.
        """
        print(f"Processing audio: {audio_path}")
        chunk_duration = 10
        chunks = split_audio_chunks(audio_path, chunk_duration)
        
        all_segments = []
        
        for i, (chunk_path, chunk_start, chunk_end) in enumerate(chunks):
            transcription = self._transcribe_chunk(chunk_path)
            
            if transcription and transcription.strip():
                segment_data = {
                    'text': transcription.strip(),
                    'start': format_timestamp(chunk_start),
                    'end': format_timestamp(chunk_end),
                    'start_seconds': chunk_start,
                    'end_seconds': chunk_end
                }
                all_segments.append(segment_data)
                
        # Cleanup chunks (temp dir logic was in split_audio_chunks or caller? 
        # split_audio_chunks created temp dir but didn't return it directly, only chunk paths.
        # we can verify check process_mp3_file in speech_processor for cleanup logic.
        # It did cleanup. We should too.)
        
        if chunks:
            temp_dir = os.path.dirname(chunks[0][0])
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        # Create Semantic Chunks
        return self._create_semantic_chunks(all_segments)

    def enhance_transcription(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print("Enhancing transcription with GPT...")
        prompt = f"""
다음은 10초 단위로 분할된 음성 전사 텍스트들의 리스트입니다. 이 리스트를 분석하여 **의미적으로 연속되거나 관련된 텍스트는 하나의 덩어리로 묶고**, 시멘틱 검색에 적합한 정보 구조로 요약 및 분류해주세요.

# 분석 조건
- 불확실하다고 판단되는 텍스트는 무시해주세요.
- 사용자가 질문할만한 최대한 많은 요약을 만들어주세요.

# 출력 형식
각 의미 덩어리에 대해 다음 JSON 형식으로 정리해주세요:

{{
"summary": "해당 발화 묶음의 핵심 내용을 1~2문장으로 요약",
"keywords": ["이 텍스트에서 중요한 단어 또는 개념"],
"topics": ["이 텍스트가 다루는 주제나 카테고리"],
"sentiment": "긍정적 / 부정적 / 중립적 중 하나로 판단",
"importance": "시멘틱 검색 관점에서 중요도: 높음 / 중간 / 낮음 중 하나로 평가",
"context": "대화의 흐름이나 발화자의 의도 등, 가능한 문맥 정보 설명",
"start_time": "시작 시간",
"end_time": "끝 시간",
"text": ["묶인 원문 텍스트들을 배열로 포함"]
}}

# 입력 음성 텍스트 청크 리스트:
{chunks}
"""
        response = openai.chat.completions.create(
            model=self.enhancement_model,
            messages=[
                {"role": "system", "content": "당신은 음성 전사 텍스트를 분석하고 시멘틱 서치를 위해 최적화하는 전문가입니다. 항상 JSON 형식으로 응답하세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        gpt_response = response.choices[0].message.content
        
        try:
            if "```json" in gpt_response:
                gpt_response = gpt_response.split("```json")[1].split("```")[0]
            
            return json.loads(gpt_response)
        except Exception as e:
            print(f"GPT Enhancement parsing failed: {e}")
            return chunks # Fallback
