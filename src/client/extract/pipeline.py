import os
import shutil

from src.repository import CloudStorageRepository, Repository

from . import preprocessing
from .cache_manager import get_video_id
from .scene_divide import detect_scene_boundaries, save_scene_info


def pipeline(project_id: str, video_path: str):
    """
    Extract video pipeline.
    """
    video_id = get_video_id(video_path)
    video_name = os.path.basename(video_path)
    debug = os.environ.get("DEBUG", "false").lower() == "true"

    repository: Repository = CloudStorageRepository()
    
    processed_video_path, processed_audio_path, metadata_path = preprocessing.process_video_and_audio(
        video_path,
        noise_reduction=True,
        target_size=480,
        extract_metadata=True,
        use_cache=not debug
    )
    print("✅ Preprocessing done")

    # 디버그용 파일 복사
    if debug:
        if not os.path.exists(f"./projects/{project_id}/{video_id}"):
            os.makedirs(f"./projects/{project_id}/{video_id}")
        shutil.copy(processed_video_path, f"./projects/{project_id}/{video_id}/video.mp4")
        shutil.copy(processed_audio_path, f"./projects/{project_id}/{video_id}/audio.wav")
        shutil.copy(metadata_path, f"./projects/{project_id}/{video_id}/metadata.json")
        print("✅ Debug files copied")

    # Scene division
    scenes = detect_scene_boundaries(
        processed_video_path, visualize=False, auto_threshold=True, min_scene_length=2.0, max_threshold=0.95)
    scene_info_path = save_scene_info(processed_video_path, scenes, cache=True)
    print("✅ Scene division done")

    # Upload to repository
    if not repository.file_exists(project_id, video_id, 'metadata.json'):
        repository.push_file(
            project_id, video_id, metadata_path, 'metadata.json')
    if not repository.file_exists(project_id, video_id, 'scenes.json'):
        repository.push_file(
            project_id, video_id, scene_info_path, 'scenes.json')
    if not repository.file_exists(project_id, video_id, 'video.mp4'):
        repository.push_file(
            project_id, video_id, processed_video_path, 'video.mp4')
    if not repository.file_exists(project_id, video_id, 'audio.wav'):
        repository.push_file(
            project_id, video_id, processed_audio_path, 'audio.wav')
    print("✅ Upload to repository done")

    return video_id
