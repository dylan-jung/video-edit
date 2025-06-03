import os
from typing import List

import numpy as np
from openai import OpenAI

from src.server.indexing.embedding.scene_to_string import scene_to_string


class OpenAIEmbeddingGenerator:
    """OpenAI 임베딩 생성기"""
    
    def __init__(self):
        """OpenAI API 클라이언트 초기화"""
        if "OPENAI_API_KEY" not in os.environ:
            raise EnvironmentError(
                "환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다. "
                "OpenAI API 키를 설정하세요."
            )
        
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_name = "text-embedding-3-large"
        
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        텍스트에 대한 임베딩 생성
        
        Args:
            text: 임베딩을 생성할 텍스트
            
        Returns:
            임베딩 벡터 (numpy array)
        """
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text,
                encoding_format="float"
            )
            embedding = np.array(response.data[0].embedding)
            return embedding
            
        except Exception as e:
            print(f"텍스트 임베딩 생성 실패: {str(e)}")
            # 실패 시 랜덤 벡터 반환 (text-embedding-3-large 차원: 3072)
            return np.random.randn(3072)
    
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
            embedding_text = scene_to_string(scene)
            
            print(f"장면 {i+1}/{len(scenes)} 임베딩 생성 중...")
            embedding = self.generate_text_embedding(embedding_text)
            embeddings.append(embedding)
        
        print(f"총 {len(embeddings)}개의 장면 임베딩 생성 완료")
        return embeddings