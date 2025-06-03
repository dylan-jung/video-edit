import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

# run test_vector_search.py projects/test/ea48283a31baa560/vector_db.faiss -i

# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.server.ai_adapter.vector_db import VectorDB
from src.server.indexing.embedding.openai_embeddings import \
    OpenAIEmbeddingGenerator


def generate_random_query_embedding(dimension: int = 3072) -> np.ndarray:
    """
    테스트용 랜덤 쿼리 임베딩 생성
    
    Args:
        dimension: 임베딩 차원 (OpenAI text-embedding-3-large: 3072)
        
    Returns:
        랜덤 임베딩 벡터
    """
    return np.random.randn(dimension).astype('float32')


def generate_text_embedding_with_openai(text: str) -> np.ndarray:
    """
    OpenAI를 사용한 실제 텍스트 임베딩 생성
    
    Args:
        text: 입력 텍스트
        
    Returns:
        OpenAI 임베딩 벡터
    """
    try:
        embedding_generator = OpenAIEmbeddingGenerator()
        embedding = embedding_generator.generate_text_embedding(text)
        return embedding.astype('float32')
    except Exception as e:
        print(f"⚠️  OpenAI 임베딩 생성 실패: {e}")
        print("   랜덤 벡터를 사용합니다...")
        return generate_random_query_embedding()


def generate_text_embedding_fallback(text: str, dimension: int = 3072) -> np.ndarray:
    """
    텍스트 기반 임베딩 생성 (OpenAI 사용 불가시 폴백)
    해시 기반 벡터 생성
    
    Args:
        text: 입력 텍스트
        dimension: 임베딩 차원
        
    Returns:
        텍스트 기반 임베딩 벡터
    """
    # 간단한 해시 기반 벡터 생성 (폴백용)
    np.random.seed(hash(text) % (2**32))
    embedding = np.random.randn(dimension).astype('float32')
    np.random.seed()  # 시드 리셋
    return embedding


def print_search_results(results: List[Dict], query_text: str = None):
    """
    검색 결과를 보기 좋게 출력
    
    Args:
        results: 검색 결과 리스트
        query_text: 쿼리 텍스트 (있는 경우)
    """
    if query_text:
        print(f"\n🔍 Query: '{query_text}'")
    print(f"\n📊 Found {len(results)} similar scenes:")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(result)
        print(f"\n🎬 Rank {i} - Video: {result['video_id']}")
        print(f"   📝 Scene ID: {result['scene_id']}")
        print(f"   ⏱️  Time: {result['start_time']}s - {result['end_time']}s")
        print(f"   📍 Similarity Score: {result['similarity_score']:.4f}")
        
        if result.get('background'):
            print(f"   🏞️  Background: {result['background']}")
        
        # if result.get('objects'):
        #     print(f"   🎯 Objects: {', '.join(result['objects'])}")
        
        if result.get('ocr_text'):
            print(f"   📄 OCR Text: {', '.join(result['ocr_text'])}")
        
        if result.get('actions'):
            print(f"   🎭 Actions: {', '.join(result['actions'])}")
        
        if result.get('emotions'):
            print(f"   😊 Emotions: {', '.join(result['emotions'])}")
        
        if result.get('context'):
            print(f"   📖 Context: {result['context']}")
        
        # if result.get('highlight'):
        #     print(f"   ⭐ Highlights: {', '.join(result['highlight'])}")
        
        print("-" * 40)


def interactive_search(vector_db: VectorDB, use_openai: bool = True):
    """
    대화형 검색 모드
    
    Args:
        vector_db: 로드된 벡터 데이터베이스
        use_openai: OpenAI 임베딩 사용 여부
    """
    print("\n🎯 Interactive Search Mode")
    if use_openai:
        print("🤖 Using OpenAI embeddings (text-embedding-3-large)")
    else:
        print("⚡ Using fallback hash-based embeddings")
    
    print("Commands:")
    print("  - Type text query for semantic search")
    print("  - Type 'random' for random query")
    print("  - Type 'stats' for database statistics")
    print("  - Type 'videos' to list all videos")
    print("  - Type 'toggle' to switch embedding mode")
    print("  - Type 'quit' to exit")
    
    while True:
        try:
            query = input("\n🔍 Enter query (or command): ").strip()
            
            if query.lower() == 'quit':
                break
            elif query.lower() == 'toggle':
                use_openai = not use_openai
                mode = "OpenAI embeddings" if use_openai else "Hash-based embeddings"
                print(f"🔄 Switched to: {mode}")
                continue
            elif query.lower() == 'stats':
                stats = vector_db.get_stats()
                print("\n📊 Database Statistics:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                continue
            elif query.lower() == 'videos':
                print("\n🎬 Videos in database:")
                for video_id in vector_db.video_metadata.keys():
                    meta = vector_db.video_metadata[video_id]
                    print(f"   - {video_id}: {meta['scene_count']} scenes")
                continue
            elif query.lower() == 'random':
                query_embedding = generate_random_query_embedding(vector_db.dimension)
                query_text = "Random Query"
            else:
                if use_openai:
                    print("🤖 Generating OpenAI embedding...")
                    query_embedding = generate_text_embedding_with_openai(query)
                else:
                    query_embedding = generate_text_embedding_fallback(query, vector_db.dimension)
                query_text = query
            
            # Top-k 개수 입력
            k = 5
            
            # 검색 수행
            results = vector_db.search_similar_scenes(query_embedding, k=k)
            
            if results:
                print_search_results(results, query_text)
            else:
                print("❌ No results found.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        # except Exception as e:
        #     print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Vector Database Search Tester")
    parser.add_argument("faiss_path", help="Path to .faiss file (without extension)")
    parser.add_argument("--query", "-q", type=str, help="Query text for search")
    parser.add_argument("--k", "-k", type=int, default=5, help="Number of top results to return")
    parser.add_argument("--random", "-r", action="store_true", help="Use random query vector")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive search mode")
    parser.add_argument("--dimension", "-d", type=int, default=3072, help="Vector dimension (default: 3072 for OpenAI)")
    parser.add_argument("--no-openai", action="store_true", help="Use fallback embeddings instead of OpenAI")
    
    args = parser.parse_args()
    
    # OpenAI 사용 여부 결정
    use_openai = not args.no_openai
    
    # .faiss 확장자 처리
    faiss_path = args.faiss_path
    if not faiss_path.endswith('.faiss'):
        faiss_path += '.faiss'
    
    # 파일 존재 확인
    index_path = faiss_path.replace('.faiss', '_index.faiss')
    metadata_path = faiss_path.replace('.faiss', '_metadata.pkl')
    
    if not os.path.exists(index_path):
        print(f"❌ Index file not found: {index_path}")
        return
    
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata file not found: {metadata_path}")
        return
    
    print(f"📂 Loading vector database from: {faiss_path}")
    
    try:
        # 벡터 데이터베이스 로드
        vector_db = VectorDB.load(faiss_path, dimension=args.dimension)
        
        # 통계 출력
        stats = vector_db.get_stats()
        print(f"✅ Database loaded successfully!")
        print(f"   📊 Total vectors: {stats['total_vectors']}")
        print(f"   🎬 Total videos: {stats['total_videos']}")
        print(f"   🎭 Total scenes: {stats['total_scenes']}")
        print(f"   📐 Vector dimension: {stats['dimension']}")
        
        if use_openai:
            print(f"   🤖 Using OpenAI embeddings")
        else:
            print(f"   ⚡ Using fallback embeddings")
        
        if stats['total_vectors'] == 0:
            print("⚠️  Database is empty!")
            return
        
        # 대화형 모드
        if args.interactive:
            interactive_search(vector_db, use_openai)
            return
        
        # 단일 쿼리 모드
        if args.random:
            query_embedding = generate_random_query_embedding(args.dimension)
            query_text = "Random Query"
        elif args.query:
            if use_openai:
                print("🤖 Generating OpenAI embedding...")
                query_embedding = generate_text_embedding_with_openai(args.query)
            else:
                query_embedding = generate_text_embedding_fallback(args.query, args.dimension)
            query_text = args.query
        else:
            print("❌ Please provide --query, --random, or --interactive option")
            return
        
        # 검색 수행
        print(f"\n🔍 Searching for top-{args.k} similar scenes...")
        results = vector_db.search_similar_scenes(query_embedding, k=args.k)
        
        if results:
            print_search_results(results, query_text)
        else:
            print("❌ No results found.")
    
    except Exception as e:
        print(f"❌ Error loading or searching database: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 