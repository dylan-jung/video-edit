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
        self.index = faiss.IndexFlatL2(dimension)  # L2 거리 기반 인덱스
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
        
        # 임베딩을 numpy 배열로 변환
        embedding_matrix = np.array(embeddings).astype('float32')
        
        # FAISS 인덱스에 추가
        start_idx = self.index.ntotal
        self.index.add(embedding_matrix)
        
        # 메타데이터 저장
        for i, scene in enumerate(scenes):
            scene_meta = {
                'video_id': video_id,
                'scene_index': i,
                'start_time': scene.get('start_time'),
                'end_time': scene.get('end_time'),
                'background': scene.get('background'),
                'objects': scene.get('objects', []),
                'ocr_text': scene.get('ocr_text', []),
                'actions': scene.get('actions', []),
                'emotions': scene.get('emotions', []),
                'context': scene.get('context'),
                'highlight': scene.get('highlight', []),
                'scene_id': f"{video_id}_scene_{i+1}",  # video_id와 scene 번호로 ID 생성
                'faiss_index': start_idx + i
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
        
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.scene_metadata):
                scene_meta = self.scene_metadata[idx].copy()
                scene_meta['similarity_score'] = float(distance)
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
        벡터 데이터베이스를 파일로 저장
        
        Args:
            filepath: 저장할 파일 경로 (.faiss 확장자)
        """
        # FAISS 인덱스 저장
        index_path = filepath.replace('.faiss', '_index.faiss')
        faiss.write_index(self.index, index_path)
        
        # 메타데이터 저장
        metadata = {
            'video_metadata': self.video_metadata,
            'scene_metadata': self.scene_metadata,
            'dimension': self.dimension
        }
        
        metadata_path = filepath.replace('.faiss', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"벡터 데이터베이스가 {filepath}에 저장되었습니다")
    
    @classmethod
    def load(cls, filepath: str, dimension: int = 768) -> 'VectorDB':
        """
        파일에서 벡터 데이터베이스 로드
        
        Args:
            filepath: 로드할 파일 경로 (.faiss 확장자)
            
        Returns:
            로드된 VectorDB 인스턴스
        """
        # 메타데이터 로드
        metadata_path = filepath.replace('.faiss', '_metadata.pkl')
        print(f"metadata_path: {metadata_path}")
        if not os.path.exists(metadata_path):
            # 파일이 없으면 새 인스턴스 생성
            return cls(dimension=dimension)
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # FAISS 인덱스 로드
        index_path = filepath.replace('.faiss', '_index.faiss')
        print(f"index_path: {index_path}")
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
        else:
            index = faiss.IndexFlatL2(metadata['dimension'])
        
        # 인스턴스 복원
        db = cls(metadata['dimension'])
        db.index = index
        db.video_metadata = metadata['video_metadata']
        db.scene_metadata = metadata['scene_metadata']
        
        print(f"벡터 데이터베이스가 {filepath}에서 로드되었습니다 (총 {db.index.ntotal}개 벡터)")
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