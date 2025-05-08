import json
import os

from src.repository import CloudStorageRepository, Repository
from src.server.ai_adapter import AIRepository
from src.server.indexing import scene_processor


def pipeline(project_id: str, video_id: str):
    """
    Indexing pipeline.
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

    if not os.path.exists(audio_url):
        audio_bytes = repository.get_video_audio(project_id, video_id)
        with open(audio_url, "wb") as f:
            f.write(audio_bytes)
    if not os.path.exists(video_url):
        video_bytes = repository.get_video_video(project_id, video_id)
        with open(video_url, "wb") as f:
            f.write(video_bytes)
    scenes_bytes = repository.get_video_scenes(project_id, video_id)
    if not os.path.exists(scenes_url):
        with open(scenes_url, "wb") as f:
            f.write(scenes_bytes)
    if not os.path.exists(metadata_url):
        metadata_bytes = repository.get_video_metadata(project_id, video_id)
        with open(metadata_url, "wb") as f:
            f.write(metadata_bytes)

    # whisper
    if not os.path.exists(transcription_path):
        remote_link = repository.get_temp_download_link(project_id, video_id, "audio.wav")
        transcription = ai_repository.whisper(remote_link)
        with open(transcription_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(transcription, ensure_ascii=False))

    # scene descriptions
    if not os.path.exists(scene_descriptions_path):
        scenes = json.loads(scenes_bytes)
        scene_descriptions = scene_processor.process_video_scenes(
            video_url, scenes, use_cache=False)
        with open(scene_descriptions_path, "w", encoding="utf-8") as f:
            json.dump(scene_descriptions, f, ensure_ascii=False)

    # push to storage
    if not repository.file_exists(project_id, video_id, "transcription.json"):
        repository.push_file(project_id, video_id, transcription_path, "transcription.json")
    if not repository.file_exists(project_id, video_id, "scene_descriptions.json"):
        repository.push_file(project_id, video_id, scene_descriptions_path, "scene_descriptions.json")