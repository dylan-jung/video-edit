import asyncio
import json
import logging
import os
import shutil
from dataclasses import asdict


# Ports
from src.shared.application.ports.video_repository import VideoRepositoryPort
from src.modules.indexing.application.ports.scene_analyzer_port import SceneAnalyzerPort
from src.modules.indexing.application.ports.speech_processor_port import SpeechProcessorPort
from src.modules.indexing.application.ports.embedding_port import EmbeddingPort
from src.modules.indexing.application.ports.scene_indexer_port import SceneIndexerPort
from src.modules.indexing.application.ports.speech_indexer_port import SpeechIndexerPort
from src.modules.indexing.application.ports.media_processing_port import MediaProcessingPort
from src.shared.application.ports.file_storage import FileStoragePort
from src.shared.application.ports.ai_service import AIServicePort
from src.modules.indexing.domain.project import Project

logger = logging.getLogger(__name__)

class PipelineConstants:
    VIDEO_FILE = "video.mp4"
    AUDIO_FILE = "audio.wav"
    TRANSCRIPTION_FILE = "transcription.json"
    SCENE_DESCRIPTIONS_FILE = "scene_descriptions.json"
    SPEECH_ANALYSIS_FILE = "speech_analysis.json"
    PROJECT_FILE = "project.json"
    VISION_VECTOR_DB = "vision_vector_db.faiss"
    SPEECH_VECTOR_DB = "speech_vector_db.faiss"


class PipelineOrchestrator:
    """
    Orchestrates the video indexing pipeline.
    Manages resources, delegates tasks to Analyzers and Indexers, and persists results.
    """
    
    def __init__(self, 
                 file_storage: FileStoragePort,
                 video_repository: VideoRepositoryPort,
                 ai_service: AIServicePort,
                 media_processor: MediaProcessingPort,
                 scene_analyzer: SceneAnalyzerPort,
                 speech_processor: SpeechProcessorPort,
                 embedding_service: EmbeddingPort,
                 scene_indexer: SceneIndexerPort,
                 speech_indexer: SpeechIndexerPort):

        self.file_storage = file_storage
        self.video_repository = video_repository
        self.ai_service = ai_service
        self.media_processor = media_processor
        self.scene_analyzer = scene_analyzer
        self.speech_processor = speech_processor
        self.embedding_service = embedding_service
        self.scene_indexer = scene_indexer
        self.speech_indexer = speech_indexer

    async def run_scene_processing(self, project_id: str, video_id: str, video_path: str, scene_descriptions_path: str, vision_vector_db_url: str):
        """Async task for Scene Analysis and Indexing"""
        logger.info(f"[{project_id}/{video_id}] Starting Scene Task")
        
        # 1. Analyze Scenes
        if not os.path.exists(scene_descriptions_path):
            try:
                logger.info("Analyzing Video Scenes...")
                # Run blocking analysis in thread
                scenes = await asyncio.to_thread(
                    self.scene_analyzer.run, 
                    video_path=video_path, 
                    chunk_duration=300
                )
                
                with open(scene_descriptions_path, "w", encoding="utf-8") as f:
                    json.dump([asdict(scene) for scene in scenes], f, ensure_ascii=False, indent=2)
                logger.info("Scene Analysis Complete")
            except Exception as e:
                logger.error(f"Scene Analysis Failed: {str(e)}")
                raise e
        
        # 2. Index Scenes
        logger.info("Indexing Scenes...")
        await asyncio.to_thread(
            self.scene_indexer.run, 
            project_id, 
            video_id, 
            scene_descriptions_path, 
            vision_vector_db_url
        )
        logger.info("Scene Task Completed")

    async def run_speech_processing(self, project_id: str, video_id: str, audio_path: str, speech_analysis_path: str, speech_vector_db_url: str, transcription_path: str):
        """Async task for Speech Analysis and Indexing"""
        logger.info(f"[{project_id}/{video_id}] Starting Speech Task")
        
        async def analyze_and_index():
            # Speech Analysis (Chunks)
            if not os.path.exists(speech_analysis_path):
                logger.info("Analyzing Speech (Chunks)...")
                chunks = await asyncio.to_thread(self.speech_processor.run, audio_path)
                enhanced_chunks = await asyncio.to_thread(self.speech_processor.enhance_transcription, chunks)
                with open(speech_analysis_path, "w", encoding="utf-8") as f:
                    json.dump(enhanced_chunks, f, ensure_ascii=False, indent=2)
            
            # Speech Indexing
            logger.info("Indexing Speech...")
            await asyncio.to_thread(
                self.speech_indexer.run, 
                project_id, 
                video_id, 
                speech_analysis_path, 
                speech_vector_db_url
            )

        async def run_transcription():
            # Full Transcription
            if not os.path.exists(transcription_path):
                logger.info("Running Whisper Transcription...")
                # Use local file with Replicate via AIServiceAdapter
                with open(audio_path, "rb") as audio_file:
                    transcription = await asyncio.to_thread(self.ai_service.transcribe_audio, audio_file)
                
                with open(transcription_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(transcription, ensure_ascii=False))
                logger.info("Whisper Transcription Completed")

        # Run both speech sub-tasks concurrently
        await asyncio.gather(analyze_and_index(), run_transcription())
        logger.info("Speech Task Completed")

    async def run_pipeline(self, project_id: str, video_id: str):
        """
        Main pipeline execution method.
        """
        base_dir = f"projects/{project_id}/{video_id}"
        os.makedirs(base_dir, exist_ok=True)
        
        # Define paths using constants
        video_path = os.path.join(base_dir, PipelineConstants.VIDEO_FILE)
        audio_path = os.path.join(base_dir, PipelineConstants.AUDIO_FILE)
        transcription_path = os.path.join(base_dir, PipelineConstants.TRANSCRIPTION_FILE)
        scene_descriptions_path = os.path.join(base_dir, PipelineConstants.SCENE_DESCRIPTIONS_FILE)
        speech_analysis_path = os.path.join(base_dir, PipelineConstants.SPEECH_ANALYSIS_FILE)
        project_file_path = os.path.join(base_dir, PipelineConstants.PROJECT_FILE)
        
        # Local Vector DB paths (Indexers modify these)
        vision_vector_db_local_path = os.path.join(base_dir, PipelineConstants.VISION_VECTOR_DB)
        speech_vector_db_local_path = os.path.join(base_dir, PipelineConstants.SPEECH_VECTOR_DB)
        
        # 1. Download Video
        if not os.path.exists(video_path):
             logger.info(f"Downloading video for {project_id}/{video_id} to {video_path}...")
             try:
                 # Use VideoRepository for download
                 await asyncio.to_thread(
                     self.video_repository.download_video, 
                     project_id, video_id, video_path
                 )
             except Exception as e:
                 logger.error(f"Failed to download video: {e}")
                 raise e

        # 2. Extract Audio
        if not os.path.exists(audio_path):
             logger.info(f"Extracting audio to {audio_path}...")
             try:
                 await asyncio.to_thread(self.media_processor.extract_audio, video_path, audio_path)
             except Exception as e:
                 logger.error(f"Audio extraction failed: {e}")
                 raise e
        
        # Verify Inputs Exist
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file missing: {video_path}")
        if not os.path.exists(audio_path):
             raise FileNotFoundError(f"Audio file missing: {audio_path}")

        try:
            logger.info(f"Starting Pipeline for {project_id}/{video_id}")
            
            # 3. Preparation: Download existing Vector DBs and Project File if available
            logger.info("Checking for existing artifacts...")
            await asyncio.to_thread(self.video_repository.download_vision_vector_db, project_id, vision_vector_db_local_path)
            await asyncio.to_thread(self.video_repository.download_speech_vector_db, project_id, speech_vector_db_local_path)
            await asyncio.to_thread(self.video_repository.download_project_file, project_id, project_file_path)
            
            # 4. Concurrent Execution
            # Pass LOCAL paths to indexers
            await asyncio.gather(
                self.run_scene_processing(project_id, video_id, video_path, scene_descriptions_path, vision_vector_db_local_path),
                self.run_speech_processing(project_id, video_id, audio_path, speech_analysis_path, speech_vector_db_local_path, transcription_path)
            )

            # Initialize Project File using Domain Model
            if not os.path.exists(project_file_path):
                project_model = Project()
                with open(project_file_path, "w", encoding="utf-8") as f:
                    json.dump(project_model.to_dict(), f, ensure_ascii=False, indent=2)

            logger.info("Pushing results to storage...")
            
            # 5. Upload Vector DBs via VideoRepository
            if os.path.exists(vision_vector_db_local_path):
                await asyncio.to_thread(self.video_repository.save_vision_vector_db, project_id, vision_vector_db_local_path)
            if os.path.exists(speech_vector_db_local_path):
                await asyncio.to_thread(self.video_repository.save_speech_vector_db, project_id, speech_vector_db_local_path)
            
            # 6. Upload Project File via VideoRepository
            if os.path.exists(project_file_path):
                await asyncio.to_thread(self.video_repository.save_project_file, project_id, project_file_path)
   
            logger.info("Pipeline Completed Successfully")

        finally:
            # Cleanup
            logger.info("Cleaning up workspace...")
            if os.path.exists(base_dir):
                shutil.rmtree(base_dir)
                logger.info(f"Removed {base_dir}")
