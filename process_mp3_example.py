#!/usr/bin/env python3
"""
Simple example of using the MP3 Speech Analyzer
Usage: python process_mp3_example.py path/to/your/audio.mp3
"""

import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.server.indexing.speech_analyzer.speech_analyzer import \
    process_complete_pipeline


def main():
    parser = argparse.ArgumentParser(description='Process MP3 file for semantic search')
    parser.add_argument('mp3_file', help='Path to MP3 file to process')
    parser.add_argument('--output', '-o', default='processed_audio.json', 
                       help='Output JSON file path (default: processed_audio.json)')
    parser.add_argument('--chunk-duration', type=int, default=10,
                       help='Audio chunk duration in seconds (default: 10)')
    parser.add_argument('--semantic-chunk-size', type=int, default=30,
                       help='Semantic chunk size in seconds (default: 30)')
    parser.add_argument('--no-gpt', action='store_true',
                       help='Skip GPT enhancement (faster, less accurate)')
    parser.add_argument('--transcription-model', default='whisper-1',
                       choices=['whisper-1', 'gpt-4o-audio-preview'],
                       help='Transcription model to use')
    parser.add_argument('--enhancement-model', default='gpt-4o-mini',
                       choices=['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo'],
                       help='GPT model for enhancement')
    
    args = parser.parse_args()
    
    # Check if MP3 file exists
    if not os.path.exists(args.mp3_file):
        print(f"❌ MP3 file not found: {args.mp3_file}")
        return 1
    
    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OpenAI API key not found!")
        print("Please set the OPENAI_API_KEY environment variable.")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return 1
    
    print(f"🎵 Processing MP3 file: {args.mp3_file}")
    print(f"📁 Output will be saved to: {args.output}")
    print(f"⚙️  Settings:")
    print(f"   - Audio chunk duration: {args.chunk_duration}s")
    print(f"   - Semantic chunk size: {args.semantic_chunk_size}s")
    print(f"   - Transcription model: {args.transcription_model}")
    print(f"   - Enhancement model: {args.enhancement_model}")
    print(f"   - GPT enhancement: {'No' if args.no_gpt else 'Yes'}")
    print()
    
    try:
        # Process the complete pipeline
        result = process_complete_pipeline(
            mp3_path=args.mp3_file,
            chunk_duration=args.chunk_duration,
            semantic_chunk_size=args.semantic_chunk_size,
            transcription_model=args.transcription_model,
            enhancement_model=args.enhancement_model,
            use_gpt_enhancement=not args.no_gpt,
            output_path=args.output
        )
        
        # Print results summary
        print(f"\n✅ Processing completed successfully!")
        print(f"📊 Results Summary:")
        print(f"   - Source file: {result['source']['file_name']}")
        print(f"   - Duration: {result['source']['duration']:.1f} seconds")
        print(f"   - Language: {result['source']['language']}")
        print(f"   - Total segments: {result['transcription']['total_segments']}")
        print(f"   - Semantic chunks: {result['semantic_search']['total_chunks']}")
        print(f"   - Text length: {result['statistics']['text_length']} characters")
        print(f"   - Average chunk duration: {result['statistics']['avg_chunk_duration']:.1f}s")
        print(f"   - GPT enhanced: {result['processing']['gpt_enhanced']}")
        
        # Show sample content
        if result['semantic_search']['chunks']:
            print(f"\n📝 Sample processed content:")
            sample_chunk = result['semantic_search']['chunks'][0]
            print(f"   Time range: {sample_chunk['duration_range']}")
            print(f"   Text: {sample_chunk['text'][:200]}...")
            
            if sample_chunk.get('gpt_analysis'):
                analysis = sample_chunk['gpt_analysis']
                if analysis.get('summary'):
                    print(f"   Summary: {analysis['summary']}")
                if analysis.get('keywords'):
                    print(f"   Keywords: {', '.join(analysis['keywords'][:5])}")
                if analysis.get('topics'):
                    print(f"   Topics: {', '.join(analysis['topics'][:3])}")
        
        print(f"\n💾 Full results saved to: {args.output}")
        print(f"🔍 This file is now ready for semantic search indexing!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error processing MP3 file: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 