import json
import os
from src.shared.infrastructure.ai.speech_vector_db import SpeechVectorDB
from src.modules.indexing.application.ports.embedding_port import EmbeddingPort
from src.modules.indexing.infrastructure.utils.embedding_utils import speech_to_string

class SpeechIndexer:
    def __init__(self, embedding_service: EmbeddingPort):
        self.embedding_service = embedding_service

    def index_speech(self, project_id: str, video_id: str, speech_analysis_path: str, vector_db_url: str):
        print("음성 벡터 인덱싱 시작...")
        try:
            # 1. 벡터 데이터베이스 로드
            vector_db = SpeechVectorDB.load(vector_db_url, dimension=3072)
            print(f"벡터 데이터베이스 로드 완료: {vector_db.get_stats()}")
            
            # 2. 해당 video_id가 이미 인덱싱되어 있는지 확인
            if not vector_db.is_video_indexed(video_id):
                print(f"비디오 {video_id}가 인덱싱되지 않았습니다. 인덱싱을 시작합니다...")
                
                # 음성 분석 데이터 로드
                with open(speech_analysis_path, "r", encoding="utf-8") as f:
                    speech_chunks = json.load(f)
                
                # 3. 임베딩 생성
                print("음성 청크 임베딩 생성 중...")
                embeddings = []
                for i, chunk in enumerate(speech_chunks):
                    # 음성 청크의 텍스트, 요약, 키워드, 토픽 등을 종합한 검색용 텍스트 생성
                    search_text = speech_to_string(chunk)
                    print(f"청크 {i+1}/{len(speech_chunks)} 임베딩 생성 중...")
                    embedding = self.embedding_service.generate_text_embedding(search_text)
                    embeddings.append(embedding)
                
                # 4. 벡터 데이터베이스에 추가
                vector_db.add_speech_chunks(video_id, speech_chunks, embeddings)
                
                # 벡터 데이터베이스 저장
                vector_db.save(vector_db_url)
                print("음성 벡터 인덱싱 완료")
            else:
                print(f"비디오 {video_id}는 이미 인덱싱되어 있습니다")
                
        except Exception as e:
             print(f"음성 인덱싱 실패: {str(e)}")
             pass
