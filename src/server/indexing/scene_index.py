import json
import os

from src.server.ai_adapter.vector_db import VectorDB
from src.server.indexing.embedding.google_embeddings import \
    GoogleEmbeddingGenerator
from src.server.indexing.embedding.openai_embeddings import \
    OpenAIEmbeddingGenerator


def index_scenes(project_id: str, video_id: str, scene_descriptions_path: str, vector_db_url: str):
    """
    Scene indexing function that creates embeddings and stores them in vector database.
    
    Args:
        project_id: Project identifier
        video_id: Video identifier
        scene_descriptions_path: Path to scene descriptions JSON file
        vector_db_url: Path to vector database file
    """
    embedding_generator = OpenAIEmbeddingGenerator()
    print("장면 벡터 인덱싱 시작...")
    try:
        # 1. 벡터 데이터베이스 로드
        vector_db = VectorDB.load(vector_db_url, dimension=3072)
        print(f"벡터 데이터베이스 로드 완료: {vector_db.get_stats()}")
        
        # 2. 해당 video_id가 이미 인덱싱되어 있는지 확인
        if not vector_db.is_video_indexed(video_id):
            print(f"비디오 {video_id}가 인덱싱되지 않았습니다. 인덱싱을 시작합니다...")
            
            # 장면 데이터 로드
            with open(scene_descriptions_path, "r", encoding="utf-8") as f:
                scenes = json.load(f)
            
            # 각 장면에 대한 임베딩 생성
            print("장면 임베딩 생성 중...")
            embeddings = embedding_generator.generate_scene_embeddings(scenes)
            
            # 벡터 데이터베이스에 추가
            vector_db.add_scenes(video_id, scenes, embeddings)
            
            # 3. 벡터 데이터베이스 저장
            vector_db.save(vector_db_url)
            print("장면 벡터 인덱싱 완료")
        else:
            print(f"비디오 {video_id}는 이미 인덱싱되어 있습니다")
            
    except Exception as e:
        print(f"장면 인덱싱 실패: {str(e)}")
        # 인덱싱 실패해도 파이프라인 계속 진행
        pass
