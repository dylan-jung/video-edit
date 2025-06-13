import json
import os
import shutil
import subprocess
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import openai


def get_audio_duration(audio_path: str) -> float:
    """
    Get the duration of an audio file using ffmpeg.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Duration in seconds
    """
    try:
        cmd = [
            'ffprobe', '-i', audio_path, 
            '-show_entries', 'format=duration',
            '-v', 'quiet', '-of', 'csv=p=0'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return 0.0

def split_audio_chunks(audio_path: str, chunk_duration: int = 10) -> List[Tuple[str, float, float]]:
    """
    Split audio file into chunks using ffmpeg.
    
    Args:
        audio_path: Path to the MP3 file
        chunk_duration: Duration of each chunk in seconds
        
    Returns:
        List of tuples (chunk_file_path, start_time, end_time)
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Get total duration
    total_duration = get_audio_duration(audio_path)
    print(f"Total audio duration: {total_duration:.2f} seconds")
    
    # Create temporary directory for chunks
    temp_dir = tempfile.mkdtemp(prefix="audio_chunks_")
    print(f"Creating audio chunks in: {temp_dir}")
    
    chunks = []
    chunk_index = 0
    
    for start_time in range(0, int(total_duration), chunk_duration):
        end_time = min(start_time + chunk_duration, total_duration)
        
        # Generate chunk filename
        chunk_filename = f"chunk_{chunk_index:04d}.wav"
        chunk_path = os.path.join(temp_dir, chunk_filename)
        
        # ffmpeg command to extract chunk
        cmd = [
            'ffmpeg', '-i', audio_path,
            '-ss', str(start_time),
            '-t', str(end_time - start_time),
            '-acodec', 'copy',
            '-y',  # Overwrite output files
            chunk_path
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            chunks.append((chunk_path, float(start_time), float(end_time)))
            chunk_index += 1
            print(f"Created chunk: {chunk_filename} ({start_time}s - {end_time}s)")
        except subprocess.CalledProcessError as e:
            print(f"Error creating chunk {chunk_index}: {e}")
            continue
    
    print(f"Successfully created {len(chunks)} audio chunks")
    return chunks

def transcribe_audio_chunk(chunk_path: str, model: str = "whisper-1") -> str:
    """
    Transcribe a single audio chunk using OpenAI's transcription API.
    
    Args:
        chunk_path: Path to the audio chunk file
        model: Model to use for transcription (whisper-1 or gpt-4o-audio-preview)
        
    Returns:
        Transcription result as text
    """
    with open(chunk_path, 'rb') as audio_file:
        response = openai.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="text",
            language="ko"  # Korean language
        )
        return response

def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to hh:mm:ss.fff format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    seconds_remainder = td.total_seconds() % 60
    milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d}.{milliseconds:03d}"

def process_mp3_file(mp3_path: str, 
                    chunk_duration: int = 10,
                    transcription_model: str = "whisper-1",
                    cleanup_chunks: bool = True) -> Dict[str, Any]:
    """
    Complete processing pipeline for MP3 file.
    
    Args:
        mp3_path: Path to the MP3 file
        chunk_duration: Duration of each chunk in seconds
        transcription_model: Model to use for transcription
        cleanup_chunks: Whether to delete temporary audio chunks after processing
        
    Returns:
        Complete transcription data with segments and timestamps
    """
    print(f"Starting MP3 processing: {mp3_path}")
    
    if not os.path.exists(mp3_path):
        raise FileNotFoundError(f"MP3 file not found: {mp3_path}")
    
    # Step 1: Split audio into chunks
    chunks = split_audio_chunks(mp3_path, chunk_duration)
    
    if not chunks:
        raise RuntimeError("Failed to create audio chunks")
    
    # Step 2: Transcribe each chunk
    all_segments = []
    full_text_parts = []
    
    for i, (chunk_path, chunk_start, chunk_end) in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)}: {chunk_start}s - {chunk_end}s")
        
        transcription = transcribe_audio_chunk(chunk_path, transcription_model)
        print(transcription)
        
        # Add transcription text to full_text_parts
        if transcription and transcription.strip():
            full_text_parts.append(transcription.strip())
            
            # Create one segment for the entire chunk since we're using text format
            segment_data = {
                'text': transcription.strip(),
                'start': format_timestamp(chunk_start),
                'end': format_timestamp(chunk_end),
            }
            all_segments.append(segment_data)
    
    # Step 3: Cleanup temporary files
    if cleanup_chunks:
        temp_dir = os.path.dirname(chunks[0][0]) if chunks else None
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
    
    # Step 4: Create final transcription data
    full_text = ' '.join(full_text_parts)
    total_duration = get_audio_duration(mp3_path)
    
    transcription_data = {
        'source_file': mp3_path,
        'duration': total_duration,
        'language': 'korean',  # Assuming Korean based on the sample
        'text': full_text,
        'segments': all_segments,
        'metadata': {
            'chunk_duration': chunk_duration,
            'total_chunks': len(chunks),
            'transcription_model': transcription_model,
            'total_segments': len(all_segments)
        }
    }
    
    print(f"Transcription completed: {len(all_segments)} segments, {total_duration:.2f}s total")
    return transcription_data

def create_semantic_chunks(transcription_data: Dict[str, Any], 
                            semantic_chunk_size: int = 30) -> List[Dict[str, Any]]:
    """
    Create semantic chunks from transcribed segments for better search capability.
    
    Args:
        transcription_data: Complete transcription data
        semantic_chunk_size: Target duration for each semantic chunk in seconds
        
    Returns:
        List of semantic chunks optimized for search
    """
    segments = transcription_data.get('segments', [])
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
        # Check if adding this segment would exceed the chunk size
        segment_start = segment['start_seconds']
        segment_end = segment['end_seconds']
        
        if (current_chunk['segments'] and 
            segment_start - current_chunk['start_time'] > semantic_chunk_size):
            
            # Finalize current chunk
            if current_chunk['segments']:
                chunk_data = _finalize_semantic_chunk(
                    current_chunk, len(semantic_chunks), transcription_data['language']
                )
                semantic_chunks.append(chunk_data)
            
            # Start new chunk
            current_chunk = {
                'segments': [segment],
                'start_time': segment_start,
                'end_time': segment_end,
                'text_parts': [segment['text']]
            }
        else:
            # Add to current chunk
            if not current_chunk['segments']:
                current_chunk['start_time'] = segment_start
            
            current_chunk['segments'].append(segment)
            current_chunk['end_time'] = segment_end
            current_chunk['text_parts'].append(segment['text'])
    
    # Don't forget the last chunk
    if current_chunk['segments']:
        chunk_data = _finalize_semantic_chunk(
            current_chunk, len(semantic_chunks), transcription_data['language']
        )
        semantic_chunks.append(chunk_data)
    
    return semantic_chunks

def _finalize_semantic_chunk(chunk_data: Dict[str, Any], 
                            chunk_id: int, language: str) -> Dict[str, Any]:
    """Finalize a semantic chunk with proper formatting."""
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
        'language': language,
        'segment_ids': [seg['id'] for seg in chunk_data['segments']]
    }

def enhance_with_gpt(semantic_chunks: List[Dict[str, Any]], 
                    model: str = "gpt-4o-mini") -> List[Dict[str, Any]]:
    """
    Enhance semantic chunks with GPT analysis for better search capability.
    
    Args:
        semantic_chunks: List of semantic chunks
        model: GPT model to use for enhancement
        
    Returns:
        Enhanced chunks with GPT analysis
    """
    
            # Create prompt for GPT analysis
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

주저하지 마세요. 당신의 작업은 굉장히 중요합니다.

# 입력 음성 텍스트 청크 리스트:
{semantic_chunks}
"""
            
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "당신은 음성 전사 텍스트를 분석하고 시멘틱 서치를 위해 최적화하는 전문가입니다. 항상 JSON 형식으로 응답하세요."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    gpt_response = response.choices[0].message.content
    
    if "```json" in gpt_response:
        gpt_response = gpt_response.split("```json")[1].split("```")[0]
        gpt_analysis = json.loads(gpt_response)
    else:
        gpt_analysis = json.loads(gpt_response)
    
    return gpt_analysis

def process_complete_pipeline(mp3_path: str,
                            chunk_duration: int = 10,
                            semantic_chunk_size: int = 30,
                            transcription_model: str = "whisper-1",
                            enhancement_model: str = "gpt-4o-mini",
                            use_gpt_enhancement: bool = True,
                            output_path: str = None) -> Dict[str, Any]:
    """
    Complete pipeline: MP3 → Chunks → Transcription → Semantic Search Optimization
    
    Args:
        mp3_path: Path to the MP3 file
        chunk_duration: Duration for audio chunks (seconds)
        semantic_chunk_size: Duration for semantic chunks (seconds)
        transcription_model: Model for transcription
        enhancement_model: Model for GPT enhancement
        use_gpt_enhancement: Whether to use GPT for enhancement
        output_path: Path to save the result (optional)
        
    Returns:
        Complete processed data ready for semantic search
    """
    print("🎵 Starting complete MP3 processing pipeline...")
    
    # Step 1: Transcribe MP3
    transcription_data = process_mp3_file(
        mp3_path=mp3_path,
        chunk_duration=chunk_duration,
        transcription_model=transcription_model
    )
    
    # Step 2: Create semantic chunks
    print("📝 Creating semantic chunks...")
    semantic_chunks = create_semantic_chunks(
        transcription_data=transcription_data,
        semantic_chunk_size=semantic_chunk_size
    )
    
    # Step 3: Enhance with GPT (optional)
    if use_gpt_enhancement:
        print("🤖 Enhancing with GPT analysis...")
        enhanced_chunks = enhance_with_gpt(semantic_chunks, enhancement_model)
    else:
        enhanced_chunks = semantic_chunks
    
    # Step 4: Create final result
    result = {
        'source': {
            'file_path': mp3_path,
            'file_name': os.path.basename(mp3_path),
            'duration': transcription_data['duration'],
            'language': transcription_data['language']
        },
        'processing': {
            'chunk_duration': chunk_duration,
            'semantic_chunk_size': semantic_chunk_size,
            'transcription_model': transcription_model,
            'enhancement_model': enhancement_model if use_gpt_enhancement else None,
            'gpt_enhanced': use_gpt_enhancement,
            'processed_at': str(Path().resolve())
        },
        'transcription': {
            'full_text': transcription_data['text'],
            'total_segments': len(transcription_data['segments']),
            'segments': transcription_data['segments']
        },
        'semantic_search': {
            'total_chunks': len(enhanced_chunks),
            'chunks': enhanced_chunks
        },
        'statistics': {
            'audio_duration': transcription_data['duration'],
            'text_length': len(transcription_data['text']),
            'avg_chunk_duration': sum(c.get('duration', 0) for c in enhanced_chunks) / len(enhanced_chunks) if enhanced_chunks else 0,
            'processing_success': True
        }
    }
    
    # Step 5: Save result if output path provided
    if output_path:
        save_result(result, output_path)
    
    print("✅ Pipeline completed successfully!")
    print(f"📊 Results: {len(enhanced_chunks)} semantic chunks from {transcription_data['duration']:.1f}s audio")
    
    return result

def save_result(result: Dict[str, Any], output_path: str):
    """Save processing result to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"💾 Results saved to: {output_path}")
