#!/usr/bin/env python3
"""
Test the RealVideoAnalyzer directly
"""
import os
from pathlib import Path
from real_video_analyzer import RealVideoAnalyzer

def test_real_analyzer():
    """Test the real analyzer directly"""
    print("🎯 TESTING REAL VIDEO ANALYZER DIRECTLY")
    
    # Look for Taylor Swift video
    video_path = None
    possible_paths = [
        Path("../downloads/Taylor Swift - The Fate of Ophelia (Official Music Video).mp4"),
        Path("downloads/Taylor Swift - The Fate of Ophelia (Official Music Video).mp4"),
        Path("Taylor Swift - The Fate of Ophelia (Official Music Video).mp4")
    ]
    
    for path in possible_paths:
        if path.exists():
            video_path = str(path)
            break
    
    if not video_path:
        print("❌ Taylor Swift video not found!")
        return False
    
    print(f"✅ Found video: {video_path}")
    
    try:
        # Initialize the real analyzer
        print("🤖 Initializing RealVideoAnalyzer...")
        analyzer = RealVideoAnalyzer()
        
        # Run analysis
        print("🎬 Running REAL analysis...")
        result = analyzer.analyze_video(video_path, "test_session")
        
        print(f"\n🎵 REAL ANALYSIS RESULTS:")
        print(f"=" * 60)
        print(f"Status: {result['status']}")
        print(f"Content Type: {result['content_type']}")
        print(f"Duration: {result['total_duration']:.1f}s")
        print(f"Segments: {len(result['segments'])}")
        
        # Check if it correctly identified as music
        if result['content_type'] == 'music':
            print("✅ CORRECTLY IDENTIFIED AS MUSIC!")
        else:
            print(f"❌ WRONG CONTENT TYPE: {result['content_type']} (should be 'music')")
        
        # Show first segment
        if result['segments']:
            seg = result['segments'][0]
            print(f"\nFirst segment analysis:")
            print(f"Visual: {seg['visual_description']}")
            print(f"Audio: {seg['audio_description']}")
            print(f"Narrative: {seg['combined_narrative']}")
            print(f"Transcription: {seg['transcription'][:100]}...")
            print(f"Objects: {len(seg['detected_objects'])}")
            print(f"Confidence: {seg['confidence_score']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ REAL ANALYZER FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_analyzer()
    if success:
        print("\n🎉 REAL ANALYZER WORKING!")
    else:
        print("\n❌ REAL ANALYZER FAILED!")