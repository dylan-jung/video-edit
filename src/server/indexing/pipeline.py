import json
import os

from src.repository import CloudStorageRepository, Repository
from src.server.ai_adapter import AIRepository
from src.server.indexing.project import initialize_project
from src.server.indexing.scene_analyzer import analyze_video_scenes
from src.server.indexing.scene_index import index_scenes
from src.server.indexing.speech_analyzer.speech_analyzer import (
    enhance_with_gpt, process_mp3_file)
from src.server.indexing.speech_index import index_speech


def pipeline(project_id: str, video_id: str):
    """
    Indexing pipeline with new Google Multimodal Embeddings + FAISS clustering-based scene analysis.
    """
    repository: Repository = CloudStorageRepository()
    ai_repository = AIRepository()

    if not os.path.exists(f"projects/{project_id}/{video_id}"):
        os.makedirs(f"projects/{project_id}/{video_id}")
    
    audio_url = f"projects/{project_id}/{video_id}/audio.wav"
    video_url = f"projects/{project_id}/{video_id}/video.mp4"
    scenes_url = f"projects/{project_id}/{video_id}/scenes.json"
    metadata_url = f"projects/{project_id}/{video_id}/metadata.json"
    vision_vector_db_url = f"projects/{project_id}/vision_vector_db.faiss"
    speech_vector_db_url = f"projects/{project_id}/speech_vector_db.faiss"

    transcription_path = f"projects/{project_id}/{video_id}/transcription.json"
    scene_descriptions_path = f"projects/{project_id}/{video_id}/scene_descriptions.json"
    speech_analysis_path = f"projects/{project_id}/{video_id}/speech_analysis.json"
    project_path = f"projects/{project_id}/{video_id}/project.json"

    # Download video and audio files, metadata from repository
    if not os.path.exists(audio_url):
        audio_bytes = repository.get_video_audio(project_id, video_id)
        with open(audio_url, "wb") as f:
            f.write(audio_bytes)
    if not os.path.exists(video_url):
        video_bytes = repository.get_video_video(project_id, video_id)
        with open(video_url, "wb") as f:
            f.write(video_bytes)
    if not os.path.exists(metadata_url):
        metadata_bytes = repository.get_video_metadata(project_id, video_id)
        with open(metadata_url, "wb") as f:
            f.write(metadata_bytes)

    # Generate scenes
    if not os.path.exists(scene_descriptions_path):
        try:
            scenes = analyze_video_scenes(video_url, model="gpt-4.1-mini", chunk_duration=300)
            with open(scene_descriptions_path, "w", encoding="utf-8") as f:
                json.dump(scenes, f, ensure_ascii=False, indent=2)
            print("장면 분석 완료")
        except Exception as e:
            print(f"장면 분석 실패: {str(e)}")
            raise e

    # scene indexing
    index_scenes(project_id, video_id, scene_descriptions_path, vision_vector_db_url)
    
    # speech analysis
    if not os.path.exists(speech_analysis_path):
        chunks = process_mp3_file(audio_url)
        enhanced_chunks = enhance_with_gpt(chunks)
        with open(speech_analysis_path, "w", encoding="utf-8") as f:
            json.dump(enhanced_chunks, f, ensure_ascii=False, indent=2)
    
    # speech indexing
    index_speech(project_id, video_id, speech_analysis_path, speech_vector_db_url)

    # whisper
    if not os.path.exists(transcription_path):
        remote_link = repository.get_temp_download_link(project_id, video_id, "audio.wav")
        transcription = ai_repository.whisper(remote_link)
        with open(transcription_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(transcription, ensure_ascii=False))

    # project
    if not os.path.exists(project_path):
        initialize_project(project_path)

    # push to storage
    if not repository.file_exists(project_id, video_id, "transcription.json"):
        repository.push_file(project_id, video_id, transcription_path, "transcription.json")
    # if not repository.file_exists(project_id, video_id, "scene_descriptions.json"):
    #     repository.push_file(project_id, video_id, scene_descriptions_path, "scene_descriptions.json")
    if not repository.file_exists(project_id, video_id, "project.json"):
        repository.push_file(project_id, video_id, project_path, "project.json")