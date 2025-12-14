import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np


class SpeechVectorDB:
    """FAISS 기반 음성 벡터 데이터베이스 관리 클래스"""
    
    def __init__(self, dimension: int = 768):
        """
        음성 벡터 데이터베이스 초기화
        
        Args:
            dimension: 벡터 차원 (기본값: 768, 구글 임베딩 차원)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # 코사인 유사도 기반 인덱스 (정규화된 벡터와 Inner Product 사용)
        self.video_metadata = {}  # video_id별 메타데이터 저장
        self.speech_metadata = []  # 각 벡터에 대응하는 음성 청크 메타데이터
        
    def add_speech_chunks(self, video_id: str, speech_chunks: List[Dict], embeddings: List[np.ndarray]):
        """
        특정 비디오의 음성 청크들을 벡터 데이터베이스에 추가
        
        Args:
            video_id: 비디오 ID
            speech_chunks: 음성 청크 정보 리스트
            embeddings: 각 음성 청크에 대응하는 임베딩 벡터 리스트
        """
        if len(speech_chunks) != len(embeddings):
            raise ValueError("speech_chunks와 embeddings의 길이가 일치하지 않습니다")
        
        # 임베딩을 numpy 배열로 변환 및 정규화 (코사인 유사도를 위해)
        embedding_matrix = np.array(embeddings).astype('float32')
        # 벡터 정규화 (L2 norm = 1)
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        embedding_matrix = embedding_matrix / (norms + 1e-8)  # 0으로 나누는 것 방지
        
        # FAISS 인덱스에 추가
        start_idx = self.index.ntotal
        self.index.add(embedding_matrix)
        
        # 메타데이터 저장
        for i, chunk in enumerate(speech_chunks):
            speech_meta = {
                'video_id': video_id,
                'start_time': chunk.get('start_time'),
                'end_time': chunk.get('end_time'),
                'summary': chunk.get('summary'),
                'keywords': chunk.get('keywords', []),
                'topics': chunk.get('topics', []),
                'sentiment': chunk.get('sentiment'),
                'importance': chunk.get('importance'),
                'context': chunk.get('context'),
                'text': chunk.get('text', []),
            }
            self.speech_metadata.append(speech_meta)
        
        # 비디오별 메타데이터 업데이트
        if video_id not in self.video_metadata:
            self.video_metadata[video_id] = {
                'chunk_count': 0,
                'start_index': start_idx,
                'end_index': start_idx + len(speech_chunks) - 1
            }
        else:
            self.video_metadata[video_id]['end_index'] = start_idx + len(speech_chunks) - 1
        
        self.video_metadata[video_id]['chunk_count'] += len(speech_chunks)
        
        print(f"비디오 {video_id}의 {len(speech_chunks)}개 음성 청크가 벡터 데이터베이스에 추가되었습니다")
    
    def is_video_indexed(self, video_id: str) -> bool:
        """
        특정 비디오가 이미 인덱싱되어 있는지 확인
        
        Args:
            video_id: 확인할 비디오 ID
            
        Returns:
            인덱싱 여부
        """
        return video_id in self.video_metadata
    
    def search_similar_speech(self, query_embedding: np.ndarray, k: int = 5, 
                            video_id: Optional[str] = None,
                            sentiment_filter: Optional[str] = None,
                            importance_filter: Optional[str] = None) -> List[Dict]:
        """
        유사한 음성 청크 검색
        
        Args:
            query_embedding: 검색할 쿼리 벡터
            k: 반환할 유사 음성 청크 수
            video_id: 특정 비디오로 검색 범위 제한 (선택사항)
            sentiment_filter: 감정으로 필터링 (선택사항)
            importance_filter: 중요도로 필터링 (선택사항)
            
        Returns:
            유사한 음성 청크들의 메타데이터 리스트
        """
        if self.index.ntotal == 0:
            return []
        
        # 쿼리 벡터 정규화 (코사인 유사도를 위해)
        query_vector = np.array([query_embedding]).astype('float32')
        query_norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
        query_vector = query_vector / (query_norm + 1e-8)  # 0으로 나누는 것 방지
        
        # Inner Product 검색 (정규화된 벡터에서는 코사인 유사도와 동일)
        similarities, indices = self.index.search(query_vector, min(k * 2, self.index.ntotal))  # 필터링을 위해 더 많이 가져옴
        
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(self.speech_metadata):
                speech_meta = self.speech_metadata[idx].copy()
                
                # 필터링 적용
                if video_id and speech_meta['video_id'] != video_id:
                    continue
                if sentiment_filter and speech_meta['sentiment'] != sentiment_filter:
                    continue
                if importance_filter and speech_meta['importance'] != importance_filter:
                    continue
                
                speech_meta['similarity_score'] = float(similarity)  # Inner Product 값 (높을수록 유사)
                results.append(speech_meta)
                
                if len(results) >= k:  # 원하는 개수만큼 채우면 종료
                    break
        
        return results
    
    def search_by_keywords(self, keywords: List[str], k: int = 5, 
                          video_id: Optional[str] = None) -> List[Dict]:
        """
        키워드로 음성 청크 검색
        
        Args:
            keywords: 검색할 키워드 리스트
            k: 반환할 음성 청크 수
            video_id: 특정 비디오로 검색 범위 제한 (선택사항)
            
        Returns:
            키워드가 포함된 음성 청크들의 메타데이터 리스트
        """
        results = []
        keywords_lower = [kw.lower() for kw in keywords]
        
        for speech_meta in self.speech_metadata:
            if video_id and speech_meta['video_id'] != video_id:
                continue
                
            # 키워드 매칭 점수 계산
            score = 0
            speech_keywords = [kw.lower() for kw in speech_meta.get('keywords', [])]
            
            for keyword in keywords_lower:
                if keyword in speech_keywords:
                    score += 1
                # 텍스트에서도 검색
                for text in speech_meta.get('text', []):
                    if keyword in text.lower():
                        score += 0.5
                        break
            
            if score > 0:
                speech_meta_copy = speech_meta.copy()
                speech_meta_copy['keyword_score'] = score
                results.append(speech_meta_copy)
        
        # 점수 순으로 정렬
        results.sort(key=lambda x: x['keyword_score'], reverse=True)
        return results[:k]
    
    def search_by_time_range(self, start_time: str, end_time: str, 
                           video_id: Optional[str] = None) -> List[Dict]:
        """
        시간 범위로 음성 청크 검색
        
        Args:
            start_time: 검색 시작 시간 (HH:MM:SS.mmm 형식)
            end_time: 검색 종료 시간 (HH:MM:SS.mmm 형식)
            video_id: 특정 비디오로 검색 범위 제한 (선택사항)
            
        Returns:
            시간 범위에 포함된 음성 청크들의 메타데이터 리스트
        """
        results = []
        
        for speech_meta in self.speech_metadata:
            if video_id and speech_meta['video_id'] != video_id:
                continue
                
            chunk_start = speech_meta.get('start_time')
            chunk_end = speech_meta.get('end_time')
            
            if chunk_start and chunk_end:
                # 시간 범위 겹침 확인
                if (chunk_start <= end_time and chunk_end >= start_time):
                    results.append(speech_meta.copy())
        
        # 시작 시간 순으로 정렬
        results.sort(key=lambda x: x.get('start_time', ''))
        return results
    
    def get_video_speech_chunks(self, video_id: str) -> List[Dict]:
        """
        특정 비디오의 모든 음성 청크 메타데이터 조회
        
        Args:
            video_id: 비디오 ID
            
        Returns:
            해당 비디오의 음성 청크 메타데이터 리스트
        """
        chunks = [meta for meta in self.speech_metadata if meta['video_id'] == video_id]
        # 시작 시간 순으로 정렬
        chunks.sort(key=lambda x: x.get('start_time', ''))
        return chunks
    
    def save(self, filepath: str):
        """
        음성 벡터 데이터베이스를 파일로 저장
        
        Args:
            filepath: 저장할 파일 경로 (.faiss 확장자)
        """
        # 1. Main File: FAISS Index
        faiss.write_index(self.index, filepath)
        
        # 2. Sidecar File: Metadata
        metadata = {
            'video_metadata': self.video_metadata,
            'speech_metadata': self.speech_metadata,
            'dimension': self.dimension
        }
        
        metadata_path = f"{filepath}.metadata"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"음성 벡터 데이터베이스가 저장되었습니다:\n- Index: {filepath}\n- Metadata: {metadata_path}")
    
    @classmethod
    def load(cls, filepath: str, dimension: int = 768) -> 'SpeechVectorDB':
        """
        파일에서 음성 벡터 데이터베이스 로드
        
        Args:
            filepath: 로드할 파일 경로 (.faiss 확장자)
            dimension: 벡터 차원 (기본값: 768)
            
        Returns:
            로드된 SpeechVectorDB 인스턴스
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
        if not os.path.exists(filepath):
            # Try legacy fallback
            legacy_index = filepath.replace('.faiss', '_index.faiss')
            if os.path.exists(legacy_index):
                filepath = legacy_index
            else:
                print("인덱스 파일 없음, 빈 인덱스로 초기화")
                db = cls(metadata.get('dimension', dimension))
                db.video_metadata = metadata['video_metadata']
                db.speech_metadata = metadata['speech_metadata']
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
        db.speech_metadata = metadata['speech_metadata']
        
        print(f"음성 벡터 데이터베이스가 {filepath}에서 로드되었습니다 (총 {db.index.ntotal}개 벡터)")
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
            'total_speech_chunks': len(self.speech_metadata)
        }
    
    def get_sentiment_distribution(self, video_id: Optional[str] = None) -> Dict[str, int]:
        """
        감정 분포 통계
        
        Args:
            video_id: 특정 비디오로 제한 (선택사항)
            
        Returns:
            감정별 개수 딕셔너리
        """
        sentiments = {}
        for meta in self.speech_metadata:
            if video_id and meta['video_id'] != video_id:
                continue
            sentiment = meta.get('sentiment', 'unknown')
            sentiments[sentiment] = sentiments.get(sentiment, 0) + 1
        return sentiments
    
    def get_importance_distribution(self, video_id: Optional[str] = None) -> Dict[str, int]:
        """
        중요도 분포 통계
        
        Args:
            video_id: 특정 비디오로 제한 (선택사항)
            
        Returns:
            중요도별 개수 딕셔너리
        """
        importance_levels = {}
        for meta in self.speech_metadata:
            if video_id and meta['video_id'] != video_id:
                continue
            importance = meta.get('importance', 'unknown')
            importance_levels[importance] = importance_levels.get(importance, 0) + 1
        return importance_levels
