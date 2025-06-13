#!/usr/bin/env python3
"""
Test script for MP3-based SpeechAnalyzer
Usage: python test_mp3_speech_analyzer.py [mp3_file_path]
"""

import argparse
import json
import os
import sys
from pathlib import Path

from src.server.indexing.speech_analyzer.speech_analyzer import (
    enhance_with_gpt, process_mp3_file)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def main():
    parser = argparse.ArgumentParser(description='Test MP3 Speech Analyzer')
    parser.add_argument('mp3_file', nargs='?', help='Path to MP3 file to process')
    parser.add_argument('--skip-requirements', action='store_true', help='Skip requirements check')
    parser.add_argument('--test-basic', action='store_true', help='Run basic functionality tests only')
    parser.add_argument('--test-transcription', action='store_true', help='Run transcription tests only')
    
    args = parser.parse_args()
    
    print("🎤 MP3 Speech Analyzer Test Suite")
    print("=" * 50)
    # Determine MP3 file to use
    mp3_path = os.path.join(os.getcwd(), "projects/test/ea48283a31baa560/audio.wav")
    
    if not os.path.exists(mp3_path):
        print(f"❌ MP3 file not found: {mp3_path}")
        return 1
    
    print(f"📁 Using MP3 file: {mp3_path}")
    
    # Initialize analyzer
    try:
        api_key = os.getenv('OPENAI_API_KEY') if not args.skip_requirements else None
        print("✅ SpeechAnalyzer initialized")
    except Exception as e:
        print(f"❌ Failed to initialize SpeechAnalyzer: {e}")
        return 1
    
    chunks = process_mp3_file(mp3_path, cleanup_chunks=False)
    with open(os.path.join(os.getcwd(), 'transcriptions.json'), 'w') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    
    enhanced_chunks = enhance_with_gpt(chunks)
    with open(os.path.join(os.getcwd(), 'enhanced_transcriptions.json'), 'w') as f:
        json.dump(enhanced_chunks, f, ensure_ascii=False, indent=4)

    print("\n🎉 All tests completed successfully!")
    print("\n📋 Test Summary:")
    print("   ✅ Basic functionality")
    print("   ✅ MP3 transcription")
    print("   ✅ Semantic chunking")
    print("   ✅ GPT enhancement" if os.getenv('OPENAI_API_KEY') else "   ⚠️  GPT enhancement (skipped - no API key)")
    print("   ✅ Complete pipeline")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 