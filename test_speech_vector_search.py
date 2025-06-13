import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

# run test_speech_vector_search.py projects/test/ea48283a31baa560/speech_vector_db.faiss -i

# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.server.ai_adapter.speech_vector_db import SpeechVectorDB
from src.server.indexing.embedding.openai_embeddings import \
    OpenAIEmbeddingGenerator


def generate_random_query_embedding(dimension: int = 768) -> np.ndarray:
    """
    테스트용 랜덤 쿼리 임베딩 생성
    
    Args:
        dimension: 임베딩 차원 (기본값: 768, 구글 임베딩 차원)
        
    Returns:
        랜덤 임베딩 벡터
    """
    return np.random.randn(dimension).astype('float32')


def generate_text_embedding_with_openai(text: str, dimension: int = 768) -> np.ndarray:
    """
    OpenAI를 사용한 실제 텍스트 임베딩 생성 (차원 축소)
    
    Args:
        text: 입력 텍스트
        dimension: 목표 차원 (기본값: 768)
        
    Returns:
        축소된 차원의 임베딩 벡터
    """
    try:
        embedding_generator = OpenAIEmbeddingGenerator()
        embedding = embedding_generator.generate_text_embedding(text)
        
        # OpenAI 임베딩(3072)을 목표 차원(768)으로 축소
        if embedding.shape[0] > dimension:
            # 단순 슬라이싱으로 차원 축소
            embedding = embedding[:dimension]
        elif embedding.shape[0] < dimension:
            # 패딩으로 차원 확장
            padding = np.zeros(dimension - embedding.shape[0])
            embedding = np.concatenate([embedding, padding])
            
        return embedding.astype('float32')
    except Exception as e:
        print(f"⚠️  OpenAI 임베딩 생성 실패: {e}")
        print("   랜덤 벡터를 사용합니다...")
        return generate_random_query_embedding(dimension)


def generate_text_embedding_fallback(text: str, dimension: int = 768) -> np.ndarray:
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
    음성 검색 결과를 보기 좋게 출력
    
    Args:
        results: 검색 결과 리스트
        query_text: 쿼리 텍스트 (있는 경우)
    """
    if query_text:
        print(f"\n🔍 Query: '{query_text}'")
    print(f"\n📊 Found {len(results)} similar speech chunks:")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"\n🎙️  Rank {i} - Video: {result['video_id']}")
        print(f"   ⏱️  Time: {result.get('start_time', 'N/A')} - {result.get('end_time', 'N/A')}")
        
        if result.get('similarity_score') is not None:
            print(f"   📍 Similarity Score: {result['similarity_score']:.4f}")
        if result.get('keyword_score') is not None:
            print(f"   🔑 Keyword Score: {result['keyword_score']}")
        
        if result.get('summary'):
            print(f"   📝 Summary: {result['summary']}")
        
        if result.get('keywords'):
            print(f"   🏷️  Keywords: {', '.join(result['keywords'])}")
        
        if result.get('topics'):
            print(f"   🏆 Topics: {', '.join(result['topics'])}")
        
        if result.get('sentiment'):
            print(f"   😊 Sentiment: {result['sentiment']}")
        
        if result.get('importance'):
            print(f"   ⭐ Importance: {result['importance']}")
        
        if result.get('context'):
            print(f"   📖 Context: {result['context']}")
        
        if result.get('text'):
            # 텍스트가 너무 길면 일부만 표시
            text_preview = ', '.join(result['text'][:3])
            if len(result['text']) > 3:
                text_preview += f" ... (+{len(result['text'])-3} more)"
            print(f"   💬 Text: {text_preview}")
        
        print("-" * 40)


def interactive_search(speech_db: SpeechVectorDB, use_openai: bool = True):
    """
    대화형 검색 모드
    
    Args:
        speech_db: 로드된 음성 벡터 데이터베이스
        use_openai: OpenAI 임베딩 사용 여부
    """
    print("\n🎯 Interactive Speech Search Mode")
    if use_openai:
        print("🤖 Using OpenAI embeddings (dimension reduced to 768)")
    else:
        print("⚡ Using fallback hash-based embeddings")
    
    print("Commands:")
    print("  - Type text query for semantic search")
    print("  - Type 'keywords: word1,word2' for keyword search")
    print("  - Type 'time: HH:MM:SS.mmm-HH:MM:SS.mmm' for time range search")
    print("  - Type 'video: video_id' to set video filter")
    print("  - Type 'sentiment: positive/negative/neutral' to set sentiment filter")
    print("  - Type 'importance: high/medium/low' to set importance filter")
    print("  - Type 'random' for random query")
    print("  - Type 'stats' for database statistics")
    print("  - Type 'videos' to list all videos")
    print("  - Type 'sentiments' for sentiment distribution")
    print("  - Type 'importance' for importance distribution")
    print("  - Type 'chunks: video_id' to list all chunks of a video")
    print("  - Type 'reset' to reset all filters")
    print("  - Type 'toggle' to switch embedding mode")
    print("  - Type 'quit' to exit")
    
    # 현재 필터 설정
    current_video_filter = None
    current_sentiment_filter = None
    current_importance_filter = None
    
    while True:
        try:
            # 현재 필터 상태 표시
            filters = []
            if current_video_filter:
                filters.append(f"video={current_video_filter}")
            if current_sentiment_filter:
                filters.append(f"sentiment={current_sentiment_filter}")
            if current_importance_filter:
                filters.append(f"importance={current_importance_filter}")
            
            filter_str = f" [Filters: {', '.join(filters)}]" if filters else ""
            
            query = input(f"\n🔍 Enter query (or command){filter_str}: ").strip()
            
            if query.lower() == 'quit':
                break
            elif query.lower() == 'toggle':
                use_openai = not use_openai
                mode = "OpenAI embeddings" if use_openai else "Hash-based embeddings"
                print(f"🔄 Switched to: {mode}")
                continue
            elif query.lower() == 'reset':
                current_video_filter = None
                current_sentiment_filter = None
                current_importance_filter = None
                print("🔄 All filters reset")
                continue
            elif query.lower() == 'stats':
                stats = speech_db.get_stats()
                print("\n📊 Database Statistics:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                continue
            elif query.lower() == 'videos':
                print("\n🎬 Videos in database:")
                for video_id in speech_db.video_metadata.keys():
                    meta = speech_db.video_metadata[video_id]
                    print(f"   - {video_id}: {meta['chunk_count']} chunks")
                continue
            elif query.lower() == 'sentiments':
                sentiments = speech_db.get_sentiment_distribution(current_video_filter)
                print(f"\n😊 Sentiment Distribution{' for ' + current_video_filter if current_video_filter else ''}:")
                for sentiment, count in sentiments.items():
                    print(f"   {sentiment}: {count}")
                continue
            elif query.lower() == 'importance':
                importance = speech_db.get_importance_distribution(current_video_filter)
                print(f"\n⭐ Importance Distribution{' for ' + current_video_filter if current_video_filter else ''}:")
                for level, count in importance.items():
                    print(f"   {level}: {count}")
                continue
            elif query.startswith('video:'):
                video_id = query.split(':', 1)[1].strip()
                if video_id in speech_db.video_metadata:
                    current_video_filter = video_id
                    print(f"🎬 Video filter set to: {video_id}")
                else:
                    print(f"❌ Video '{video_id}' not found in database")
                continue
            elif query.startswith('sentiment:'):
                sentiment = query.split(':', 1)[1].strip().lower()
                current_sentiment_filter = sentiment
                print(f"😊 Sentiment filter set to: {sentiment}")
                continue
            elif query.startswith('importance:'):
                importance = query.split(':', 1)[1].strip().lower()
                current_importance_filter = importance
                print(f"⭐ Importance filter set to: {importance}")
                continue
            elif query.startswith('chunks:'):
                video_id = query.split(':', 1)[1].strip()
                chunks = speech_db.get_video_speech_chunks(video_id)
                if chunks:
                    print(f"\n🎙️  Speech chunks for video {video_id}:")
                    print_search_results(chunks)
                else:
                    print(f"❌ No chunks found for video '{video_id}'")
                continue
            elif query.startswith('keywords:'):
                keywords_str = query.split(':', 1)[1].strip()
                keywords = [kw.strip() for kw in keywords_str.split(',')]
                print(f"🔑 Searching by keywords: {keywords}")
                
                # Top-k 개수 입력
                k = 5
                
                # 키워드 검색 수행
                results = speech_db.search_by_keywords(keywords, k=k, video_id=current_video_filter)
                
                if results:
                    print_search_results(results, f"Keywords: {', '.join(keywords)}")
                else:
                    print("❌ No results found.")
                continue
            elif query.startswith('time:'):
                time_range = query.split(':', 1)[1].strip()
                try:
                    start_time, end_time = time_range.split('-')
                    start_time = start_time.strip()
                    end_time = end_time.strip()
                    print(f"⏰ Searching by time range: {start_time} - {end_time}")
                    
                    # 시간 범위 검색 수행
                    results = speech_db.search_by_time_range(start_time, end_time, video_id=current_video_filter)
                    
                    if results:
                        print_search_results(results, f"Time range: {start_time} - {end_time}")
                    else:
                        print("❌ No results found.")
                except ValueError:
                    print("❌ Invalid time format. Use: HH:MM:SS.mmm-HH:MM:SS.mmm")
                continue
            elif query.lower() == 'random':
                query_embedding = generate_random_query_embedding(speech_db.dimension)
                query_text = "Random Query"
            else:
                if use_openai:
                    print("🤖 Generating OpenAI embedding...")
                    query_embedding = generate_text_embedding_with_openai(query, speech_db.dimension)
                else:
                    query_embedding = generate_text_embedding_fallback(query, speech_db.dimension)
                query_text = query
            
            # Top-k 개수 입력
            k = 5
            
            # 유사도 검색 수행
            results = speech_db.search_similar_speech(
                query_embedding, 
                k=k, 
                video_id=current_video_filter,
                sentiment_filter=current_sentiment_filter,
                importance_filter=current_importance_filter
            )
            
            if results:
                print_search_results(results, query_text)
            else:
                print("❌ No results found.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Speech Vector Database Search Tester")
    parser.add_argument("faiss_path", help="Path to .faiss file (without extension)")
    parser.add_argument("--query", "-q", type=str, help="Query text for search")
    parser.add_argument("--keywords", type=str, help="Comma-separated keywords for search")
    parser.add_argument("--time-range", type=str, help="Time range for search (HH:MM:SS.mmm-HH:MM:SS.mmm)")
    parser.add_argument("--video-id", type=str, help="Filter by specific video ID")
    parser.add_argument("--sentiment", type=str, choices=['positive', 'negative', 'neutral'], help="Filter by sentiment")
    parser.add_argument("--importance", type=str, choices=['high', 'medium', 'low'], help="Filter by importance")
    parser.add_argument("--k", "-k", type=int, default=5, help="Number of top results to return")
    parser.add_argument("--random", "-r", action="store_true", help="Use random query vector")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive search mode")
    parser.add_argument("--dimension", "-d", type=int, default=768, help="Vector dimension (default: 768)")
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
    
    print(f"📂 Loading speech vector database from: {faiss_path}")
    
    try:
        # 음성 벡터 데이터베이스 로드
        speech_db = SpeechVectorDB.load(faiss_path, dimension=args.dimension)
        
        # 통계 출력
        stats = speech_db.get_stats()
        print(f"✅ Database loaded successfully!")
        print(f"   📊 Total vectors: {stats['total_vectors']}")
        print(f"   🎬 Total videos: {stats['total_videos']}")
        print(f"   🎙️  Total speech chunks: {stats['total_speech_chunks']}")
        print(f"   📐 Vector dimension: {stats['dimension']}")
        
        if use_openai:
            print(f"   🤖 Using OpenAI embeddings (dimension reduced to {args.dimension})")
        else:
            print(f"   ⚡ Using fallback embeddings")
        
        if stats['total_vectors'] == 0:
            print("⚠️  Database is empty!")
            return
        
        # 대화형 모드
        if args.interactive:
            interactive_search(speech_db, use_openai)
            return
        
        # 단일 쿼리 모드
        if args.random:
            query_embedding = generate_random_query_embedding(args.dimension)
            query_text = "Random Query"
            results = speech_db.search_similar_speech(
                query_embedding, 
                k=args.k, 
                video_id=args.video_id,
                sentiment_filter=args.sentiment,
                importance_filter=args.importance
            )
        elif args.keywords:
            keywords = [kw.strip() for kw in args.keywords.split(',')]
            query_text = f"Keywords: {', '.join(keywords)}"
            results = speech_db.search_by_keywords(keywords, k=args.k, video_id=args.video_id)
        elif args.time_range:
            try:
                start_time, end_time = args.time_range.split('-')
                start_time = start_time.strip()
                end_time = end_time.strip()
                query_text = f"Time range: {start_time} - {end_time}"
                results = speech_db.search_by_time_range(start_time, end_time, video_id=args.video_id)
            except ValueError:
                print("❌ Invalid time format. Use: HH:MM:SS.mmm-HH:MM:SS.mmm")
                return
        elif args.query:
            if use_openai:
                print("🤖 Generating OpenAI embedding...")
                query_embedding = generate_text_embedding_with_openai(args.query, args.dimension)
            else:
                query_embedding = generate_text_embedding_fallback(args.query, args.dimension)
            query_text = args.query
            results = speech_db.search_similar_speech(
                query_embedding, 
                k=args.k, 
                video_id=args.video_id,
                sentiment_filter=args.sentiment,
                importance_filter=args.importance
            )
        else:
            print("❌ Please provide --query, --keywords, --time-range, --random, or --interactive option")
            return
        
        # 검색 수행
        print(f"\n🔍 Searching for top-{args.k} results...")
        
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