import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np


class FaissVectorDB:
    """
    FAISS 기반 벡터 데이터베이스 구현
    비디오별 파티션을 지원하여 임베딩을 저장하고 유사도 검색을 수행합니다.
    """
    
    def __init__(self, db_path: str, embedding_dim: int = 1408):  # Google Multimodal Embeddings 차원
        """
        Initialize FAISS vector database
        
        Args:
            db_path: 데이터베이스 디렉토리 경로
            embedding_dim: 임베딩 차원 (Google Multimodal Embeddings: 1408)
        """
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        self.video_partitions = {}  # video_path -> partition_info
        self.load_db()
    
    def _get_partition_path(self, video_path: str) -> Path:
        """비디오별 파티션 경로 생성"""
        video_name = Path(video_path).stem
        return self.db_path / f"partition_{video_name}"
    
    def _create_index(self) -> faiss.Index:
        """새로운 FAISS 인덱스 생성"""
        # L2 거리 기반 인덱스 사용 (코사인 유사도를 위해 정규화된 벡터 사용)
        index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product (정규화된 벡터에서 코사인 유사도)
        return index
    
    def load_db(self):
        """데이터베이스 파일에서 파티션 정보 로드"""
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # 파티션 메타데이터 로드
        metadata_path = self.db_path / "partitions_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.video_partitions = json.load(f)
                print(f"✅ Loaded {len(self.video_partitions)} video partitions from {self.db_path}")
            except Exception as e:
                print(f"Error loading partition metadata: {e}")
                self.video_partitions = {}
        else:
            print(f"Creating new FAISS vector database at {self.db_path}")
            self.video_partitions = {}
    
    def save_db(self):
        """파티션 메타데이터를 파일에 저장"""
        metadata_path = self.db_path / "partitions_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.video_partitions, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved partition metadata for {len(self.video_partitions)} videos")
    
    def _load_partition(self, video_path: str) -> Tuple[faiss.Index, List[Dict]]:
        """특정 비디오 파티션 로드"""
        partition_path = self._get_partition_path(video_path)
        
        index_path = partition_path / "index.faiss"
        metadata_path = partition_path / "metadata.pkl"
        
        if index_path.exists() and metadata_path.exists():
            try:
                # FAISS 인덱스 로드
                index = faiss.read_index(str(index_path))
                
                # 메타데이터 로드
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                return index, metadata
            except Exception as e:
                print(f"Error loading partition for {video_path}: {e}")
                return self._create_index(), []
        else:
            return self._create_index(), []
    
    def _save_partition(self, video_path: str, index: faiss.Index, metadata: List[Dict]):
        """특정 비디오 파티션 저장"""
        partition_path = self._get_partition_path(video_path)
        partition_path.mkdir(parents=True, exist_ok=True)
        
        index_path = partition_path / "index.faiss"
        metadata_path = partition_path / "metadata.pkl"
        
        # FAISS 인덱스 저장
        faiss.write_index(index, str(index_path))
        
        # 메타데이터 저장
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # 파티션 정보 업데이트
        self.video_partitions[video_path] = {
            'count': len(metadata),
            'last_updated': str(Path().cwd())  # 간단한 타임스탬프 대용
        }
    
    def add_embedding(self, embedding: np.ndarray, metadata: Dict):
        """
        임베딩과 메타데이터 추가
        
        Args:
            embedding: 임베딩 벡터 (정규화된 상태여야 함)
            metadata: 메타데이터 (video_path 필수)
        """
        video_path = metadata.get('video_path')
        if not video_path:
            raise ValueError("video_path is required in metadata")
        
        # 해당 비디오 파티션 로드
        index, partition_metadata = self._load_partition(video_path)
        
        # 임베딩 정규화 (코사인 유사도를 위해)
        embedding = embedding.flatten()
        embedding = embedding / np.linalg.norm(embedding)
        
        # 인덱스에 추가
        index.add(embedding.reshape(1, -1).astype(np.float32))
        partition_metadata.append(metadata)
        
        # 파티션 저장
        self._save_partition(video_path, index, partition_metadata)
    
    def add_embeddings_batch(self, embeddings: List[np.ndarray], metadata_list: List[Dict]):
        """
        여러 임베딩을 배치로 추가 (비디오별로 그룹화하여 효율적으로 처리)
        
        Args:
            embeddings: 임베딩 벡터 리스트
            metadata_list: 메타데이터 리스트
        """
        # 비디오별로 그룹화
        video_groups = {}
        for embedding, metadata in zip(embeddings, metadata_list):
            video_path = metadata.get('video_path')
            if not video_path:
                continue
            
            if video_path not in video_groups:
                video_groups[video_path] = {'embeddings': [], 'metadata': []}
            
            video_groups[video_path]['embeddings'].append(embedding)
            video_groups[video_path]['metadata'].append(metadata)
        
        # 각 비디오별로 배치 추가
        for video_path, group in video_groups.items():
            index, partition_metadata = self._load_partition(video_path)
            
            # 임베딩들 정규화
            embeddings_array = np.array([emb.flatten() for emb in group['embeddings']])
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / norms
            
            # 배치로 인덱스에 추가
            index.add(embeddings_array.astype(np.float32))
            partition_metadata.extend(group['metadata'])
            
            # 파티션 저장
            self._save_partition(video_path, index, partition_metadata)
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5, 
                      threshold: float = 0.5, video_path: str = None) -> List[Tuple[Dict, float]]:
        """
        유사한 임베딩 검색
        
        Args:
            query_embedding: 쿼리 임베딩
            top_k: 반환할 상위 결과 수
            threshold: 유사도 임계값
            video_path: 특정 비디오에서만 검색 (None이면 모든 비디오)
            
        Returns:
            List of (metadata, similarity_score) tuples
        """
        # 쿼리 임베딩 정규화
        query_embedding = query_embedding.flatten()
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        all_results = []
        
        # 검색할 비디오 목록 결정
        search_videos = [video_path] if video_path else list(self.video_partitions.keys())
        
        for vid_path in search_videos:
            try:
                index, metadata = self._load_partition(vid_path)
                
                if index.ntotal == 0:  # 빈 인덱스
                    continue
                
                # 검색 수행
                search_k = min(top_k * 2, index.ntotal)  # 여유있게 검색
                scores, indices = index.search(query_embedding, search_k)
                
                # 결과 처리
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0 and score >= threshold and idx < len(metadata):
                        all_results.append((metadata[idx], float(score)))
                        
            except Exception as e:
                print(f"Error searching in partition {vid_path}: {e}")
                continue
        
        # 점수순으로 정렬하고 상위 k개 반환
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]
    
    def get_embeddings_by_metadata(self, filter_func, video_path: str = None) -> List[Tuple[np.ndarray, Dict]]:
        """
        메타데이터 조건에 맞는 임베딩들 반환
        
        Args:
            filter_func: 메타데이터를 받아서 boolean을 반환하는 함수
            video_path: 특정 비디오에서만 검색 (None이면 모든 비디오)
            
        Returns:
            List of (embedding, metadata) tuples
        """
        results = []
        
        # 검색할 비디오 목록 결정
        search_videos = [video_path] if video_path else list(self.video_partitions.keys())
        
        for vid_path in search_videos:
            try:
                index, metadata = self._load_partition(vid_path)
                
                for i, meta in enumerate(metadata):
                    if filter_func(meta):
                        # 인덱스에서 임베딩 복원
                        embedding = index.reconstruct(i)
                        results.append((embedding, meta))
                        
            except Exception as e:
                print(f"Error accessing partition {vid_path}: {e}")
                continue
        
        return results
    
    def clear_video(self, video_path: str):
        """특정 비디오의 파티션 삭제"""
        partition_path = self._get_partition_path(video_path)
        
        # 파티션 파일들 삭제
        if partition_path.exists():
            import shutil
            shutil.rmtree(partition_path)
        
        # 메타데이터에서 제거
        if video_path in self.video_partitions:
            del self.video_partitions[video_path]
    
    def clear_all(self):
        """모든 파티션 삭제"""
        if self.db_path.exists():
            import shutil
            shutil.rmtree(self.db_path)
        
        self.video_partitions = {}
        self.db_path.mkdir(parents=True, exist_ok=True)
    
    def get_video_count(self, video_path: str) -> int:
        """특정 비디오의 임베딩 수 반환"""
        return self.video_partitions.get(video_path, {}).get('count', 0)
    
    def get_total_count(self) -> int:
        """전체 임베딩 수 반환"""
        return sum(info.get('count', 0) for info in self.video_partitions.values())
    
    def list_videos(self) -> List[str]:
        """저장된 비디오 목록 반환"""
        return list(self.video_partitions.keys())

    def get_video_embeddings_ordered(self, video_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        특정 비디오의 모든 임베딩을 프레임 순서대로 반환
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            Tuple of (embeddings_array, metadata_list) - 프레임 인덱스 순으로 정렬됨
        """
        try:
            index, metadata = self._load_partition(video_path)
            
            if index.ntotal == 0:
                return np.array([]), []
            
            # 프레임 인덱스 순으로 정렬
            sorted_indices = sorted(range(len(metadata)), key=lambda i: metadata[i].get('frame_index', i))
            
            # 정렬된 순서로 임베딩과 메타데이터 추출
            embeddings_list = []
            sorted_metadata = []
            
            for i in sorted_indices:
                embedding = index.reconstruct(i)
                embeddings_list.append(embedding)
                sorted_metadata.append(metadata[i])
            
            embeddings_array = np.array(embeddings_list)
            return embeddings_array, sorted_metadata
            
        except Exception as e:
            print(f"Error loading embeddings for {video_path}: {e}")
            return np.array([]), []


class FrameVectorDB(FaissVectorDB):
    """
    비디오 프레임 임베딩을 위한 특화된 FAISS 벡터 DB
    """
    
    def add_frame_embedding(self, embedding: np.ndarray, video_path: str, 
                           frame_index: int, timestamp: float):
        """
        프레임 임베딩 추가
        
        Args:
            embedding: 프레임 임베딩
            video_path: 비디오 파일 경로
            frame_index: 프레임 인덱스
            timestamp: 타임스탬프 (초)
        """
        metadata = {
            'video_path': video_path,
            'frame_index': frame_index,
            'timestamp': timestamp
        }
        self.add_embedding(embedding, metadata)
    
    def search_frames_by_time_range(self, start_time: float, end_time: float, 
                                   video_path: str = None) -> List[Tuple[np.ndarray, Dict]]:
        """
        시간 범위로 프레임 검색
        
        Args:
            start_time: 시작 시간 (초)
            end_time: 종료 시간 (초)
            video_path: 비디오 경로 (None이면 모든 비디오)
            
        Returns:
            List of (embedding, metadata) tuples
        """
        def filter_func(metadata):
            return start_time <= metadata['timestamp'] <= end_time
        
        return self.get_embeddings_by_metadata(filter_func, video_path) 