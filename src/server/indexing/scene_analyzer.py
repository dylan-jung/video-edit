import base64
import glob
import json
import os
import subprocess
import tempfile
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from st_dbscan import ST_DBSCAN

try:
    from kneed import KneeLocator
except ImportError:
    KneeLocator = None  # kneed not installed

from src.server.indexing.clip_embeddings import CLIPMultimodalEmbeddings
from src.server.indexing.vector_db import FrameVectorDB
from src.server.utils.cache_manager import get_cache_path


class SceneAnalyzer:
    """
    ST-DBSCAN을 사용한 시공간 scene 분석 클래스
    1. 비디오를 1초마다 샘플링
    2. 각 프레임을 CLIP Embeddings로 임베딩하여 FAISS 벡터DB에 저장
    3. ST-DBSCAN 클러스터링 (공간적 유사성 + 시간적 연속성)
    4. 클러스터 기반 scene 분할
    """
    
    def __init__(self, vector_db_path: str = None, model_name: str = "ViT-L-14-336", device: str = None,
                 spatial_eps: float = 35.191, temporal_eps: float = 3.0, min_samples: int = 3):
        """
        Initialize the scene analyzer with CLIP Embeddings and FAISS vector DB
        
        Args:
            vector_db_path: Path to FAISS vector database directory
            model_name: CLIP model name (e.g., "ViT-L-14", "ViT-B-32")
            device: torch device string
            spatial_eps: 공간적 거리 임계값 (CLIP embedding space)
            temporal_eps: 시간적 거리 임계값 (초 단위)
            min_samples: 클러스터를 형성하기 위한 최소 샘플 수
        """
        # CLIP Embeddings 초기화
        self.embeddings_generator = CLIPMultimodalEmbeddings(model_name=model_name, device=device)
        self.scaler = StandardScaler()
        
        # ST-DBSCAN 파라미터
        self.spatial_eps = spatial_eps
        self.temporal_eps = temporal_eps
        self.min_samples = min_samples
        
        # FAISS 벡터 데이터베이스 초기화
        if vector_db_path is None:
            vector_db_path = "projects/faiss_vector_db"
        
        embedding_dim = self.embeddings_generator.get_embedding_dimension()
        self.vector_db = FrameVectorDB(vector_db_path, embedding_dim=embedding_dim)
        
    def extract_frames_per_second(self, video_path: str, output_dir: str = None) -> List[str]:
        """
        비디오를 1초마다 1개의 jpg로 샘플링
        
        Args:
            video_path: 입력 비디오 파일 경로
            output_dir: 출력 디렉토리 (None이면 임시 디렉토리 사용)
            
        Returns:
            List of frame file paths
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # ffmpeg를 사용하여 1초마다 프레임 추출
        frame_pattern = os.path.join(output_dir, "frame_%04d.jpg")
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf', 'fps=1',  # 1 frame per second
            '-q:v', '2',     # High quality
            frame_pattern
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            raise Exception(f"Error extracting frames from {video_path}")
        
        # 추출된 프레임 파일들을 정렬하여 반환
        frame_files = sorted(glob.glob(os.path.join(output_dir, "frame_*.jpg")))
        print(f"✅ Extracted {len(frame_files)} frames from video")
        
        return frame_files
    
    def generate_embeddings(self, frame_paths: List[str], video_path: str = None, 
                          save_to_db: bool = True) -> np.ndarray:
        """
        각 프레임에 대해 CLIP Embeddings 생성하고 FAISS 벡터DB에 저장
        벡터DB에 이미 존재하는 임베딩이 있으면 불러와서 재사용
        
        Args:
            frame_paths: 프레임 이미지 파일 경로 리스트
            video_path: 비디오 파일 경로 (벡터DB 저장용)
            save_to_db: 벡터DB에 저장할지 여부
            
        Returns:
            numpy array of embeddings (n_frames, embedding_dim)
        """
        expected_frame_count = len(frame_paths)
        
        # 벡터DB에서 기존 임베딩 확인
        if video_path and save_to_db:
            existing_embeddings, existing_metadata = self.vector_db.get_video_embeddings_ordered(video_path)
            
            # 기존 임베딩이 있고 프레임 수가 일치하면 재사용
            if len(existing_embeddings) == expected_frame_count:
                print(f"✅ Found existing embeddings in vector DB for {expected_frame_count} frames")
                print(f"✅ Reusing embeddings with shape: {existing_embeddings.shape}")
                return existing_embeddings
            elif len(existing_embeddings) > 0:
                print(f"⚠️  Found {len(existing_embeddings)} existing embeddings, but expected {expected_frame_count}")
                print(f"🔄 Clearing existing embeddings and regenerating...")
                self.vector_db.clear_video(video_path)
        
        print(f"🔄 Generating CLIP embeddings for {len(frame_paths)} frames...")
        
        # CLIP Embeddings 배치 생성
        embeddings = self.embeddings_generator.generate_embeddings_batch(
            frame_paths, 
            task_type="RETRIEVAL_DOCUMENT",
            batch_size=5  # API 제한 고려
        )
        
        # 벡터DB에 저장
        if save_to_db and video_path:
            print("💾 Saving embeddings to FAISS vector database...")
            
            # 메타데이터 준비
            metadata_list = []
            for i, frame_path in enumerate(frame_paths):
                metadata = {
                    'video_path': video_path,
                    'frame_index': i,
                    'timestamp': float(i),  # 1초마다 샘플링이므로 인덱스가 곧 초
                    'frame_path': frame_path
                }
                metadata_list.append(metadata)
            
            # 배치로 벡터DB에 추가
            self.vector_db.add_embeddings_batch(embeddings, metadata_list)
            self.vector_db.save_db()
        
        embeddings_array = np.array(embeddings)
        print(f"✅ Generated embeddings with shape: {embeddings_array.shape}")
        
        return embeddings_array
    
    def perform_clustering(self, embeddings: np.ndarray, timestamps: np.ndarray, 
                          n_clusters: int = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        ST-DBSCAN 클러스터링 수행 (시공간 차원)
        
        Args:
            embeddings: 임베딩 배열
            timestamps: 타임스탬프 배열 (초 단위)
            n_clusters: 사용되지 않음 (ST-DBSCAN에서는 자동 결정)
            
        Returns:
            Tuple of (cluster_labels, None) - ST-DBSCAN에서는 cluster_centers가 없음
        """
        print("🔄 Performing ST-DBSCAN clustering...")
        
        # ST-DBSCAN 클러스터링 수행
        cluster_labels = self.st_dbscan_clustering(embeddings, timestamps)
        
        # 최종 클러스터링 결과 출력
        unique_labels = np.unique(cluster_labels)
        n_clusters_final = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(cluster_labels == -1)
        
        print(f"✅ Final clustering completed with {n_clusters_final} clusters and {n_noise} noise points")
        
        # 클러스터별 프레임 수 출력
        for cluster_id in unique_labels:
            if cluster_id == -1:
                print(f"  Noise points: {n_noise} frames")
                continue
            count = np.sum(cluster_labels == cluster_id)
            cluster_timestamps = timestamps[cluster_labels == cluster_id]
            time_span = cluster_timestamps.max() - cluster_timestamps.min() if count > 1 else 0
            print(f"  Cluster {cluster_id}: {count} frames, time span: {time_span:.1f}s")
        
        return cluster_labels, None
    
    def generate_scenes_from_clusters(self, cluster_labels: np.ndarray, 
                                    frame_indices: np.ndarray = None) -> List[Dict]:
        """
        클러스터를 기반으로 Scene의 start_time, end_time 설계
        노이즈 포인트(-1)는 제외하고 처리
        
        Args:
            cluster_labels: 클러스터 라벨 배열
            frame_indices: 프레임 인덱스 (outlier 제거 후)
            
        Returns:
            List of scene dictionaries with start_time and end_time
        """
        if frame_indices is None:
            frame_indices = np.arange(len(cluster_labels))
        
        print("🔄 Generating scenes from clusters (excluding noise points)...")
        
        # 노이즈 포인트(-1) 제외
        valid_indices = cluster_labels != -1
        valid_cluster_labels = cluster_labels[valid_indices]
        valid_frame_indices = frame_indices[valid_indices]
        
        if len(valid_cluster_labels) == 0:
            print("⚠️  All points are noise, no scenes generated")
            return []
        
        scenes = []
        
        # 연속된 같은 클러스터를 하나의 scene으로 그룹화
        current_cluster = valid_cluster_labels[0]
        scene_start = valid_frame_indices[0]
        
        for i in range(1, len(valid_cluster_labels)):
            if valid_cluster_labels[i] != current_cluster:
                # 이전 scene 종료
                scene_end = valid_frame_indices[i - 1]
                scenes.append({
                    "start_time": float(scene_start),  # 초 단위
                    "end_time": float(scene_end + 1),  # 다음 초까지 포함
                    "cluster_id": int(current_cluster)
                })
                
                # 새로운 scene 시작
                current_cluster = valid_cluster_labels[i]
                scene_start = valid_frame_indices[i]
        
        # 마지막 scene 추가
        scene_end = valid_frame_indices[-1]
        scenes.append({
            "start_time": float(scene_start),
            "end_time": float(scene_end + 1),
            "cluster_id": int(current_cluster)
        })
        
        print(f"✅ Generated {len(scenes)} scenes from clusters (noise points excluded)")
        
        # 생성된 scene 정보 출력
        for i, scene in enumerate(scenes, 1):
            duration = scene['end_time'] - scene['start_time']
            print(f"  Scene {i}: {int(scene['start_time'] / 60)}m {int(scene['start_time'] % 60)}s - {int(scene['end_time'] / 60)}m {int(scene['end_time'] % 60)}s "
                  f"(duration: {duration:.1f}s, cluster: {scene['cluster_id']})")
        
        return scenes
    
    def search_similar_frames(self, query_embedding: np.ndarray, top_k: int = 5, 
                             threshold: float = 0.7, video_path: str = None) -> List[Tuple[Dict, float]]:
        """
        유사한 프레임 검색
        
        Args:
            query_embedding: 쿼리 임베딩
            top_k: 반환할 상위 결과 수
            threshold: 유사도 임계값
            video_path: 특정 비디오에서만 검색
            
        Returns:
            List of (metadata, similarity_score) tuples
        """
        return self.vector_db.search_similar(query_embedding, top_k, threshold, video_path)
    
    def analyze_video_scenes(self, video_path: str, use_cache: bool = True) -> List[Dict]:
        """
        전체 비디오 scene 분석 파이프라인 (ST-DBSCAN 사용)
        
        Args:
            video_path: 비디오 파일 경로
            use_cache: 캐시 사용 여부
            
        Returns:
            List of scene dictionaries
        """
        # 캐시 확인
        cache_key = {"method": "st_dbscan_clustering", "video_path": video_path, 
                    "spatial_eps": self.spatial_eps, "temporal_eps": self.temporal_eps, 
                    "min_samples": self.min_samples}
        hit, cache_path = get_cache_path(video_path, cache_key)
        
        if hit and use_cache:
            print(f"🔍 Using cached scene analysis: {cache_path}")
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        print(f"🎬 Starting ST-DBSCAN scene analysis for: {video_path}")
        print(f"  Parameters: spatial_eps={self.spatial_eps}, temporal_eps={self.temporal_eps}, min_samples={self.min_samples}")
        
        # 1. 비디오에서 1초마다 프레임 추출
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_paths = self.extract_frames_per_second(video_path, temp_dir)
            
            if len(frame_paths) == 0:
                raise Exception("No frames extracted from video")
            
            # 2. 각 프레임에 대해 CLIP Embeddings 생성하고 FAISS 벡터DB에 저장
            embeddings = self.generate_embeddings(frame_paths, video_path, save_to_db=True)
            
            # 3. 타임스탬프 생성 (1초마다 샘플링이므로 인덱스가 곧 초)
            timestamps = np.arange(len(embeddings), dtype=float)
            
            # 4. ST-DBSCAN 클러스터링 수행
            cluster_labels, _ = self.perform_clustering(embeddings, timestamps)
            
            # 5. 클러스터를 기반으로 scene 생성
            scenes = self.generate_scenes_from_clusters(cluster_labels, timestamps.astype(int))
        
        # 캐시에 저장
        if use_cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(scenes, f, indent=2)
        
        print(f"✅ ST-DBSCAN scene analysis completed: {len(scenes)} scenes generated")
        
        return scenes
    
    def st_dbscan_clustering(self, embeddings: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """
        ST-DBSCAN 클러스터링 수행 (공간적 + 시간적 차원)
        
        Args:
            embeddings: CLIP 임베딩 배열 (n_frames, embedding_dim)
            timestamps: 프레임 타임스탬프 배열 (n_frames,) - 초 단위
            
        Returns:
            cluster_labels: 클러스터 라벨 배열 (-1은 노이즈)
        """
        print("🔄 Performing ST-DBSCAN clustering...")
        
        # 임베딩 정규화
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        print(f"  Spatial features shape: {embeddings_scaled.shape}")
        print(f"  Temporal features shape: {timestamps.shape}")
        print(f"  Parameters: spatial_eps={self.spatial_eps}, temporal_eps={self.temporal_eps}, min_samples={self.min_samples}")

        # --- k‑distance elbow → knee for spatial εₛ suggestion ---
        try:
            k_val = max(self.min_samples, 2)  # at least 2‑NN
            nbrs = NearestNeighbors(n_neighbors=k_val, metric='euclidean').fit(embeddings_scaled)
            dists, _ = nbrs.kneighbors(embeddings_scaled)
            k_dists = np.sort(dists[:, -1])

            if KneeLocator is not None:
                knee = KneeLocator(range(len(k_dists)), k_dists,
                                   curve='convex', direction='increasing')
                if knee.knee is not None:
                    knee_eps = float(k_dists[knee.knee])
                    print(f"🔍 k‑distance elbow knee detected → εₛ≈{knee_eps:.3f} "
                          f"(k={k_val}, suggest try ~{knee_eps*1.2:.3f})")
                else:
                    print("⚠️  KneeLocator could not find an elbow; inspect k‑distance plot manually.")
            else:
                print("⚠️  `kneed` not installed – skipping automatic knee detection.")
        except Exception as e:
            print(f"⚠️  Error during k‑distance knee calculation: {e}")
        # ----------------------------------------------------------
        
        # ST-DBSCAN 클러스터링 수행
        # st_dbscan 라이브러리는 공간적, 시간적 파라미터를 분리하여 처리
        st_dbscan = ST_DBSCAN(
            eps1=self.spatial_eps,      # 공간적 거리 임계값
            eps2=self.temporal_eps,     # 시간적 거리 임계값
            min_samples=self.min_samples
        )
        
        # 데이터 준비: 공간 데이터와 시간 데이터를 결합
        # st_dbscan은 [spatial_features + temporal_features] 형태의 데이터를 기대
        temporal_data = timestamps.reshape(-1, 1)
        combined_data = np.hstack([embeddings_scaled, temporal_data])
        
        with open("combined_data.json", "w") as f:
            f.write(json.dumps(combined_data.tolist()))
            print("✅ Saved combined data to combined_data.json")
        
        # ST-DBSCAN 실행
        st_dbscan.fit(combined_data)
        cluster_labels = st_dbscan.labels
        
        # 클러스터링 결과 분석
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(cluster_labels == -1)
        
        print(f"✅ ST-DBSCAN completed:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Number of noise points: {n_noise}")
        print(f"  Noise ratio: {n_noise/len(cluster_labels)*100:.1f}%")
        
        # 클러스터별 프레임 수 출력
        for label in unique_labels:
            if label == -1:
                continue
            count = np.sum(cluster_labels == label)
            cluster_timestamps = timestamps[cluster_labels == label]
            time_span = cluster_timestamps.max() - cluster_timestamps.min()
            print(f"  Cluster {label}: {count} frames, time span: {time_span:.1f}s")

        # --- Silhouette score for scale sanity check ---
        try:
            if n_clusters > 1:
                sil = silhouette_score(combined_data, cluster_labels, metric='euclidean')
                print(f"📈 Silhouette score (joint space): {sil:.3f}")
            else:
                print("ℹ️  Silhouette score not computed (only one cluster).")
        except Exception as e:
            print(f"⚠️  Error computing silhouette score: {e}")
        # ------------------------------------------------
        
        return cluster_labels


def analyze_video_scenes_clustering(video_path: str, use_cache: bool = True, 
                                   vector_db_path: str = None, model_name: str = "ViT-H-14", device: str = None,
                                   spatial_eps: float = 31.992, temporal_eps: float = 3.0, min_samples: int = 3) -> List[Dict]:
    """
    ST-DBSCAN 기반 비디오 scene 분석 함수 (시공간 클러스터링)
    
    Args:
        video_path: 비디오 파일 경로
        use_cache: 캐시 사용 여부
        vector_db_path: FAISS 벡터 DB 디렉토리 경로
        model_name: CLIP model name (e.g., "ViT-L-14", "ViT-B-32")
        device: torch device string
        spatial_eps: 공간적 거리 임계값 (CLIP embedding space)
        temporal_eps: 시간적 거리 임계값 (초 단위)
        min_samples: 클러스터를 형성하기 위한 최소 샘플 수
        
    Returns:
        List of scene dictionaries with start_time, end_time, cluster_id
    """
    analyzer = SceneAnalyzer(vector_db_path=vector_db_path, model_name=model_name, device=device,
                           spatial_eps=35.3904, temporal_eps=temporal_eps, min_samples=min_samples)
    return analyzer.analyze_video_scenes(video_path, use_cache) 