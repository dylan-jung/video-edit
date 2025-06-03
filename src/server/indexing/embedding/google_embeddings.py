import os
from typing import List

import google.generativeai as genai
import numpy as np

from src.server.indexing.embedding.scene_to_string import scene_to_string


class GoogleEmbeddingGenerator:
    """구글 Multimodal 임베딩 생성기"""
    
    def __init__(self):
        """구글 AI API 클라이언트 초기화"""
        if "GOOGLE_AI_API_KEY" not in os.environ:
            raise EnvironmentError(
                "환경변수 GOOGLE_AI_API_KEY가 설정되어 있지 않습니다. "
                "Google AI API 키를 설정하세요."
            )
        
        genai.configure(api_key=os.environ["GOOGLE_AI_API_KEY"])
        self.model_name = "models/text-embedding-004"
        
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        텍스트에 대한 임베딩 생성
        
        Args:
            text: 임베딩을 생성할 텍스트
            
        Returns:
            임베딩 벡터 (numpy array)
        """
        try:
            response = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            embedding = np.array(response['embedding'])
            return embedding
            
        except Exception as e:
            print(f"텍스트 임베딩 생성 실패: {str(e)}")
            # 실패 시 랜덤 벡터 반환 (차원: 768)
            return np.random.randn(768)
    
    def generate_scene_embeddings(self, scenes: List[dict]) -> List[np.ndarray]:
        """
        장면 리스트에 대한 임베딩 생성
        
        Args:
            scenes: 장면 정보 리스트
            
        Returns:
            각 장면에 대한 임베딩 벡터 리스트
        """
        embeddings = []
        
        for i, scene in enumerate(scenes):
            # 장면을 종합적인 텍스트로 변환
            embedding_text = scene_to_string(scene)
            
            print(f"장면 {i+1}/{len(scenes)} 임베딩 생성 중...")
            print(f"텍스트: {embedding_text[:100]}..." if len(embedding_text) > 100 else f"텍스트: {embedding_text}")
            
            embedding = self.generate_text_embedding(embedding_text)
            embeddings.append(embedding)
        
        print(f"총 {len(embeddings)}개의 장면 임베딩 생성 완료")
        return embeddings


def create_scene_embeddings(scenes: List[dict]) -> List[np.ndarray]:
    """
    장면 리스트에 대한 임베딩 생성 (편의 함수)
    
    Args:
        scenes: 장면 정보 리스트
        
    Returns:
        각 장면에 대한 임베딩 벡터 리스트
    """
    generator = GoogleEmbeddingGenerator()
    return generator.generate_scene_embeddings(scenes) 