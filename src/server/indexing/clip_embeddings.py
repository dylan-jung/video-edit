import base64
import os
from typing import List, Optional

import numpy as np
import open_clip
import torch
from PIL import Image


class CLIPMultimodalEmbeddings:
    """
    Google Multimodal Embeddings를 사용하여 이미지 임베딩을 생성하는 클래스
    """
    
    def __init__(self, model_name: str = "ViT-L-14-336", device: str | None = None):
        """
        Initialize CLIP Multimodal Embeddings

        Args:
            model_name: CLIP model variant to load (e.g. "ViT-L-14-336", "ViT-L-14", "ViT-B-32").
            device: torch device string. Defaults to "cuda" if available else "cpu".
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="openai"
        )
        self.model.to(self.device).eval()
        self.model_name = model_name
        self._embed_dim = self.model.visual.output_dim
        print(f"✅ Initialized CLIP embeddings with model: {model_name} on {self.device}")
    
    def generate_embedding_from_path(self, image_path: str, task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        """
        이미지 파일 경로로부터 임베딩 생성
        
        Args:
            image_path: 이미지 파일 경로
            task_type: 임베딩 태스크 타입
            
        Returns:
            numpy array of embedding
        """
        try:
            image = Image.open(image_path).convert("RGB")
            return self.generate_embedding_from_image(image, task_type)
        except Exception as e:
            print(f"Error generating embedding from {image_path}: {e}")
            return np.zeros(self._embed_dim)
    
    def generate_embedding_from_image(self, image: Image.Image, task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        """
        PIL Image로부터 임베딩 생성
        
        Args:
            image: PIL Image 객체
            task_type: 임베딩 태스크 타입
            
        Returns:
            numpy array of embedding
        """
        try:
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor).cpu().numpy()[0]
            return embedding / np.linalg.norm(embedding)
        except Exception as e:
            print(f"Error generating CLIP embedding from image: {e}")
            return np.zeros(self._embed_dim)
    
    def generate_embeddings_batch(self, image_paths: List[str], 
                                 task_type: str = "RETRIEVAL_DOCUMENT",
                                 batch_size: int = 10) -> List[np.ndarray]:
        """
        여러 이미지에 대해 배치로 임베딩 생성
        
        Args:
            image_paths: 이미지 파일 경로 리스트
            task_type: 임베딩 태스크 타입
            batch_size: 배치 크기 (API 제한 고려)
            
        Returns:
            List of numpy arrays (embeddings)
        """
        embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_embeddings = []
            
            for j, image_path in enumerate(batch_paths):
                try:
                    embedding = self.generate_embedding_from_path(image_path, task_type)
                    batch_embeddings.append(embedding)
                    
                    if (i + j + 1) % 10 == 0:
                        print(f"  Processed {i + j + 1}/{len(image_paths)} images")
                        
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    batch_embeddings.append(np.zeros(self._embed_dim))
            
            embeddings.extend(batch_embeddings)
            
            # API 제한을 고려한 간단한 지연
            # if i + batch_size < len(image_paths):
            #     import time
            #     time.sleep(0.1)  # 100ms 지연
        
        print(f"✅ Generated {len(embeddings)} CLIP embeddings")
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """임베딩 차원 반환"""
        return self._embed_dim


def create_embeddings_generator(**kwargs) -> CLIPMultimodalEmbeddings:
    """
    임베딩 생성기 팩토리 함수 (CLIP 기반)
    
    Args:
        **kwargs: 생성자 인자들
    
    Returns:
        CLIP 임베딩 생성기 인스턴스
    """
    return CLIPMultimodalEmbeddings(**kwargs)