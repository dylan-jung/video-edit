import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np


class VectorDB:
    """FAISS 기반 벡터 데이터베이스 관리 클래스"""
    
    def __init__(self, dimension: int = 768):
        """
        벡터 데이터베이스 초기화
        
        Args:
            dimension: 벡터 차원 (기본값: 768, 구글 임베딩 차원)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # 코사인 유사도 기반 인덱스 (정규화된 벡터와 Inner Product 사용)
        self.video_metadata = {}  # video_id별 메타데이터 저장
        self.scene_metadata = []  # 각 벡터에 대응하는 장면 메타데이터
        
    def add_scenes(self, video_id: str, scenes: List[Dict], embeddings: List[np.ndarray]):
        """
        특정 비디오의 장면들을 벡터 데이터베이스에 추가
        
        Args:
            video_id: 비디오 ID
            scenes: 장면 정보 리스트
            embeddings: 각 장면에 대응하는 임베딩 벡터 리스트
        """
        if len(scenes) != len(embeddings):
            raise ValueError("scenes와 embeddings의 길이가 일치하지 않습니다")
        
        # 임베딩을 numpy 배열로 변환 및 정규화 (코사인 유사도를 위해)
        embedding_matrix = np.array(embeddings).astype('float32')
        # 벡터 정규화 (L2 norm = 1)
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        embedding_matrix = embedding_matrix / (norms + 1e-8)  # 0으로 나누는 것 방지
        
        # FAISS 인덱스에 추가
        start_idx = self.index.ntotal
        self.index.add(embedding_matrix)
        
        # 메타데이터 저장
        for i, scene in enumerate(scenes):
            scene_meta = {
                'video_id': video_id,
                'start_time': scene.get('start_time'),
                'end_time': scene.get('end_time'),
                'background': scene.get('background'),
                'objects': scene.get('objects', []),
                'ocr_text': scene.get('ocr_text', []),
                'actions': scene.get('actions', []),
                'emotions': scene.get('emotions', []),
                'context': scene.get('context'),
                'highlight': scene.get('highlight', []),
            }
            self.scene_metadata.append(scene_meta)
        
        # 비디오별 메타데이터 업데이트
        if video_id not in self.video_metadata:
            self.video_metadata[video_id] = {
                'scene_count': 0,
                'start_index': start_idx,
                'end_index': start_idx + len(scenes) - 1
            }
        else:
            self.video_metadata[video_id]['end_index'] = start_idx + len(scenes) - 1
        
        self.video_metadata[video_id]['scene_count'] += len(scenes)
        
        print(f"비디오 {video_id}의 {len(scenes)}개 장면이 벡터 데이터베이스에 추가되었습니다")
    
    def is_video_indexed(self, video_id: str) -> bool:
        """
        특정 비디오가 이미 인덱싱되어 있는지 확인
        
        Args:
            video_id: 확인할 비디오 ID
            
        Returns:
            인덱싱 여부
        """
        return video_id in self.video_metadata
    
    def search_similar_scenes(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        유사한 장면 검색
        
        Args:
            query_embedding: 검색할 쿼리 벡터
            k: 반환할 유사 장면 수
            
        Returns:
            유사한 장면들의 메타데이터 리스트
        """
        if self.index.ntotal == 0:
            return []
        
        # 쿼리 벡터 정규화 (코사인 유사도를 위해)
        query_vector = np.array([query_embedding]).astype('float32')
        query_norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
        query_vector = query_vector / (query_norm + 1e-8)  # 0으로 나누는 것 방지
        
        # Inner Product 검색 (정규화된 벡터에서는 코사인 유사도와 동일)
        similarities, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(self.scene_metadata):
                scene_meta = self.scene_metadata[idx].copy()
                scene_meta['similarity_score'] = float(similarity)  # Inner Product 값 (높을수록 유사)
                results.append(scene_meta)
        
        return results
    
    def get_video_scenes(self, video_id: str) -> List[Dict]:
        """
        특정 비디오의 모든 장면 메타데이터 조회
        
        Args:
            video_id: 비디오 ID
            
        Returns:
            해당 비디오의 장면 메타데이터 리스트
        """
        return [meta for meta in self.scene_metadata if meta['video_id'] == video_id]
    
    def save(self, filepath: str):
        """
        벡터 데이터베이스를 파일로 저장 (Index + Metadata Sidecar)
        
        Args:
            filepath: 저장할 파일 경로 (.faiss)
        """
        # 1. Main File: FAISS Index
        # 이 파일이 존재해야 Orchestrator의 exist 체크를 통과함
        # 또한 표준 FAISS 도구로 읽을 수 있음
        faiss.write_index(self.index, filepath)
        
        # 2. Sidecar File: Metadata
        metadata = {
            'video_metadata': self.video_metadata,
            'scene_metadata': self.scene_metadata,
            'dimension': self.dimension
        }
        
        metadata_path = f"{filepath}.metadata"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"벡터 데이터베이스가 저장되었습니다:\n- Index: {filepath}\n- Metadata: {metadata_path}")
    
    @classmethod
    def load(cls, filepath: str, dimension: int = 768) -> 'VectorDB':
        """
        파일에서 벡터 데이터베이스 로드
        
        Args:
            filepath: 로드할 파일 경로 (.faiss)
            
        Returns:
            로드된 VectorDB 인스턴스
        """
        metadata_path = f"{filepath}.metadata"
        
        # 1. Load Metadata
        if not os.path.exists(metadata_path):
             # Try legacy fallback (pre-fix naming)
             legacy_meta = filepath.replace('.faiss', '_metadata.pkl')
             if os.path.exists(legacy_meta):
                 metadata_path = legacy_meta
             else:
                 print(f"메타데이터 파일을 찾을 수 없어 새로 생성합니다: {filepath}")
                 return cls(dimension=dimension)

        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
        except Exception as e:
            print(f"메타데이터 로드 실패: {e}")
            return cls(dimension=dimension)
            
        # 2. Load Index
        # filepath points to the main .faiss file now
        if not os.path.exists(filepath):
            # Try legacy fallback
            legacy_index = filepath.replace('.faiss', '_index.faiss')
            if os.path.exists(legacy_index):
                filepath = legacy_index
            else:
                # If metadata exists but index doesn't, create empty index with correct dim
                print("인덱스 파일 없음, 빈 인덱스로 초기화")
                db = cls(metadata.get('dimension', dimension))
                db.video_metadata = metadata['video_metadata']
                db.scene_metadata = metadata['scene_metadata']
                return db

        try:
            index = faiss.read_index(filepath)
        except Exception as e:
             print(f"FAISS 인덱스 읽기 실패 ({filepath}): {e}")
             return cls(dimension=dimension)
        
        # 3. Reconstruct
        db = cls(metadata.get('dimension', dimension))
        db.index = index
        db.video_metadata = metadata['video_metadata']
        db.scene_metadata = metadata['scene_metadata']
        
        print(f"벡터 데이터베이스 로드 완료 ({db.index.ntotal} vectors)")
        return db
    
    def get_stats(self) -> Dict:
        """
        데이터베이스 통계 정보 반환
        
        Returns:
            통계 정보 딕셔너리
        """
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'total_videos': len(self.video_metadata),
            'total_scenes': len(self.scene_metadata)
        } 