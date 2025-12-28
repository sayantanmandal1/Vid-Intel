#!/usr/bin/env python3
"""
Test the REAL video analysis system with Taylor Swift music video
"""
import requests
import json
import time
import os
from pathlib import Path

def test_taylor_swift_analysis():
    """Test analysis with Taylor Swift music video"""
    print("🎵 TESTING TAYLOR SWIFT MUSIC VIDEO ANALYSIS")
    
    # Look for the Taylor Swift video
    video_path = Path("downloads/Taylor Swift - The Fate of Ophelia (Official Music Video).mp4")
    
    if not video_path.exists():
        # Try alternative locations
        alt_paths = [
            Path("../downloads/Taylor Swift - The Fate of Ophelia (Official Music Video).mp4"),
            Path("Taylor Swift - The Fate of Ophelia (Official Music Video).mp4"),
            Path("downloads").glob("*Taylor*Swift*.mp4"),
            Path("../downloads").glob("*Taylor*Swift*.mp4"),
            Path(".").glob("*Taylor*Swift*.mp4")
        ]
        
        for alt_path in alt_paths:
            if isinstance(alt_path, Path) and alt_path.exists():
                video_path = alt_path
                break
            elif hasattr(alt_path, '__iter__'):  # It's a glob result
                try:
                    video_path = next(alt_path)
                    break
                except StopIteration:
                    continue
    
    if not video_path.exists():
        print("❌ Taylor Swift music video not found!")
        print("Expected location: downloads/Taylor Swift - The Fate of Ophelia (Official Music Video).mp4")
        print("Please ensure the video file is in the correct location.")
        return False
    
    print(f"✅ Found video: {video_path}")
    print(f"📊 File size: {video_path.stat().st_size / (1024*1024):.1f} MB")
    
    try:
        # Upload the music video
        with open(video_path, 'rb') as f:
            files = {'file': (video_path.name, f, 'video/mp4')}
            data = {'analysis_options': '{}'}
            
            print("🚀 Uploading Taylor Swift music video...")
            response = requests.post('http://localhost:8000/analyze/comprehensive', 
                                   files=files, data=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                session_id = result['session_id']
                print(f"✅ Upload successful! Session ID: {session_id}")
                
                # Monitor progress with detailed logging
                print("\n📊 MONITORING ANALYSIS PROGRESS...")
                for i in range(30):  # Wait up to 60 seconds
                    time.sleep(2)
                    
                    try:
                        status_response = requests.get(f'http://localhost:8000/analyze/status/{session_id}')
                        if status_response.status_code == 200:
                            status = status_response.json()
                            current_segment = status.get('current_segment', 'Processing...')
                            progress = status['progress']
                            
                            print(f"   📈 {progress:.1f}% - {current_segment}")
                            
                            if status['status'] == 'completed':
                                print("🎉 ANALYSIS COMPLETED!")
                                break
                            elif status['status'] == 'failed':
                                print(f"❌ ANALYSIS FAILED: {status.get('error', 'Unknown error')}")
                                return False
                        else:
                            print(f"   ⚠️ Status check returned {status_response.status_code}")
                    except Exception as e:
                        print(f"   ⚠️ Status check error: {e}")
                
                # Get the analysis results
                print("\n📋 FETCHING ANALYSIS RESULTS...")
                try:
                    results_response = requests.get(f'http://localhost:8000/analyze/results/{session_id}')
                    if results_response.status_code == 200:
                        results = results_response.json()
                        
                        print(f"\n🎵 TAYLOR SWIFT MUSIC VIDEO ANALYSIS RESULTS")
                        print(f"=" * 60)
                        print(f"Status: {results['status']}")
                        print(f"Content Type: {results['content_type']}")
                        print(f"Duration: {results['total_duration']:.1f} seconds ({results['total_duration']/60:.1f} minutes)")
                        print(f"Processing Time: {results.get('processing_time', 0):.2f}s")
                        print(f"Total Segments: {len(results.get('segments', []))}")
                        
                        segments = results.get('segments', [])
                        if segments:
                            print(f"\n🎬 SEGMENT ANALYSIS:")
                            print(f"-" * 60)
                            
                            for i, segment in enumerate(segments):
                                start_min = int(segment['start_time'] // 60)
                                end_min = int(segment['end_time'] // 60)
                                start_sec = int(segment['start_time'] % 60)
                                end_sec = int(segment['end_time'] % 60)
                                
                                print(f"\n🎵 Segment {i+1}: {start_min}:{start_sec:02d} - {end_min}:{end_sec:02d}")
                                print(f"   Visual: {segment.get('visual_description', 'N/A')}")
                                print(f"   Audio: {segment.get('audio_description', 'N/A')}")
                                print(f"   Narrative: {segment.get('combined_narrative', 'N/A')}")
                                print(f"   Transcription: {segment.get('transcription', 'N/A')[:100]}{'...' if len(segment.get('transcription', '')) > 100 else ''}")
                                print(f"   Objects: {len(segment.get('detected_objects', []))}")
                                print(f"   Confidence: {segment.get('confidence_score', 0):.2f}")
                            
                            print(f"\n📊 SUMMARY:")
                            print(f"-" * 60)
                            print(results.get('summary', 'No summary available'))
                            
                            # Validate that this is actually analyzing the music video
                            music_keywords = ['music', 'song', 'taylor', 'swift', 'ophelia', 'video', 'artist', 'performance']
                            all_text = " ".join([
                                results.get('summary', ''),
                                " ".join([seg.get('combined_narrative', '') for seg in segments]),
                                " ".join([seg.get('transcription', '') for seg in segments])
                            ]).lower()
                            
                            music_matches = sum(1 for keyword in music_keywords if keyword in all_text)
                            
                            if music_matches >= 3:
                                print(f"\n✅ SUCCESS: Analysis correctly identified music video content!")
                                print(f"   Found {music_matches} music-related keywords in analysis")
                                return True
                            else:
                                print(f"\n⚠️ WARNING: Analysis may not be music-specific enough")
                                print(f"   Only found {music_matches} music-related keywords")
                                return False
                        else:
                            print("\n❌ No segments found in results")
                            return False
                            
                    else:
                        print(f"❌ Results fetch failed: {results_response.status_code}")
                        print(results_response.text)
                        return False
                        
                except Exception as e:
                    print(f"❌ Results fetch error: {e}")
                    return False
                    
            else:
                print(f"❌ Upload failed: {response.status_code}")
                print(response.text)
                return False
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎵 TAYLOR SWIFT MUSIC VIDEO ANALYSIS TEST")
    print("=" * 60)
    
    success = test_taylor_swift_analysis()
    
    if success:
        print(f"\n🎉 SUCCESS! The system correctly analyzed the Taylor Swift music video!")
        print("✅ No more fallback outputs - real analysis working!")
    else:
        print(f"\n❌ FAILED! The system still has issues or couldn't find the video.")
        print("🔧 Check the server logs and ensure the video file is in the correct location.")