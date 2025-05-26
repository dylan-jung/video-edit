import json
import os

from src.repository import CloudStorageRepository, Repository
from src.server.ai_adapter import AIRepository
from src.server.indexing import scene_processor
from src.server.indexing.project import initialize_project
from src.server.indexing.scene_analyzer import analyze_video_scenes_clustering


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

    transcription_path = f"projects/{project_id}/{video_id}/transcription.json"
    scene_descriptions_path = f"projects/{project_id}/{video_id}/scene_descriptions.json"
    project_path = f"projects/{project_id}/{video_id}/project.json"

    # Download video and audio files
    if not os.path.exists(audio_url):
        audio_bytes = repository.get_video_audio(project_id, video_id)
        with open(audio_url, "wb") as f:
            f.write(audio_bytes)
    if not os.path.exists(video_url):
        video_bytes = repository.get_video_video(project_id, video_id)
        with open(video_url, "wb") as f:
            f.write(video_bytes)
    
    # Download metadata
    if not os.path.exists(metadata_url):
        metadata_bytes = repository.get_video_metadata(project_id, video_id)
        with open(metadata_url, "wb") as f:
            f.write(metadata_bytes)

    # Generate scenes using new Google Multimodal + FAISS clustering-based method
    if not os.path.exists(scenes_url):
        print("🎬 Generating scenes using Google Multimodal + FAISS clustering analysis...")
        
        # Get Google AI API key from environment
        google_api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_AI_API_KEY environment variable is required for scene analysis")
        
        # Set FAISS vector DB path for this project
        vector_db_path = f"projects/{project_id}/faiss_vector_db"
        
        scenes = analyze_video_scenes_clustering(
            video_path=video_url,
            use_cache=True,
            vector_db_path=vector_db_path,
            google_api_key=google_api_key
        )
        
        with open(scenes_url, "w", encoding="utf-8") as f:
            json.dump(scenes, f, ensure_ascii=False, indent=2)
        print(f"✅ Generated {len(scenes)} scenes and saved to {scenes_url}")
    else:
        # Load existing scenes
        with open(scenes_url, "r", encoding="utf-8") as f:
            scenes = json.load(f)

    # whisper
    if not os.path.exists(transcription_path):
        remote_link = repository.get_temp_download_link(project_id, video_id, "audio.wav")
        transcription = ai_repository.whisper(remote_link)
        with open(transcription_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(transcription, ensure_ascii=False))

    # scene descriptions
    if not os.path.exists(scene_descriptions_path):
        # Use the new clustering-based scenes for description
        scene_paths = scene_processor.divide_scenes(
            video_url, scenes, use_cache=False)
        
        signed_urls = []
        for scene_path in scene_paths:
            file_name = os.path.basename(scene_path)

            if not repository.file_exists(project_id, video_id, file_name, sub_path="scene"):
                repository.push_file(project_id, video_id, scene_path, file_name, sub_path="scene")
            signed_url = repository.get_temp_download_link(project_id, video_id, file_name, sub_path="scene")
            signed_urls.append(signed_url)

        scene_descriptions = scene_processor.describe_scenes(
            signed_urls, scenes, use_cache=False)
        
        with open(scene_descriptions_path, "w", encoding="utf-8") as f:
            json.dump(scene_descriptions, f, ensure_ascii=False)

    # project
    if not os.path.exists(project_path):
        initialize_project(project_path)

    # push to storage
    if not repository.file_exists(project_id, video_id, "scenes.json"):
        repository.push_file(project_id, video_id, scenes_url, "scenes.json")
    if not repository.file_exists(project_id, video_id, "transcription.json"):
        repository.push_file(project_id, video_id, transcription_path, "transcription.json")
    if not repository.file_exists(project_id, video_id, "scene_descriptions.json"):
        repository.push_file(project_id, video_id, scene_descriptions_path, "scene_descriptions.json")
    if not repository.file_exists(project_id, video_id, "project.json"):
        repository.push_file(project_id, video_id, project_path, "project.json")