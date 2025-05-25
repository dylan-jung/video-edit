#!/usr/bin/env python3
"""
Test script for the new clustering-based scene analyzer
"""

import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.server.indexing.scene_analyzer import analyze_video_scenes_clustering


def test_scene_analyzer():
    """Test the new scene analyzer with a sample video"""
    
    # Check if sample video exists
    sample_videos = [
        "projects/test/ea48283a31baa560/video.mp4"
    ]
    
    video_path = None
    for video in sample_videos:
        if os.path.exists(video):
            video_path = video
            break
    
    if not video_path:
        print("❌ No sample video found. Please ensure one of these files exists:")
        for video in sample_videos:
            print(f"  - {video}")
        return
    
    print(f"🎬 Testing scene analyzer with: {video_path}")
    
    try:
        # Run scene analysis
        scenes = analyze_video_scenes_clustering(
            video_path=video_path,
            use_cache=False,
            vector_db_path="test_vector_db.pkl"
        )
        
        print(f"\n✅ Scene analysis completed!")
        print(f"📊 Generated {len(scenes)} scenes:")
        
        for i, scene in enumerate(scenes, 1):
            duration = scene['end_time'] - scene['start_time']
            print(f"  Scene {i}: {scene['start_time']:.1f}s - {scene['end_time']:.1f}s "
                  f"(duration: {duration:.1f}s, cluster: {scene['cluster_id']})")
        
        # Save results
        output_file = f"test_scenes_{Path(video_path).stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(scenes, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Print summary statistics
        total_duration = scenes[-1]['end_time'] if scenes else 0
        avg_scene_duration = sum(s['end_time'] - s['start_time'] for s in scenes) / len(scenes) if scenes else 0
        unique_clusters = len(set(s['cluster_id'] for s in scenes))
        
        print(f"\n📈 Summary:")
        print(f"  Total video duration: {total_duration:.1f}s")
        print(f"  Average scene duration: {avg_scene_duration:.1f}s")
        print(f"  Number of unique clusters: {unique_clusters}")
        
    except Exception as e:
        print(f"❌ Error during scene analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_scene_analyzer() 