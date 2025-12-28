#!/usr/bin/env python3
"""
Test that the server is using the NEW system
"""
import requests
import json
import time
from pathlib import Path

def create_test_video():
    """Create a small test video"""
    test_video = "test_music_video.mp4"
    
    # Create a minimal MP4 file
    mp4_header = bytes([
        0x00, 0x00, 0x00, 0x20,  # box size
        0x66, 0x74, 0x79, 0x70,  # 'ftyp'
        0x69, 0x73, 0x6F, 0x6D,  # major brand 'isom'
        0x00, 0x00, 0x02, 0x00,  # minor version
        0x69, 0x73, 0x6F, 0x6D,  # compatible brand 'isom'
        0x69, 0x73, 0x6F, 0x32,  # compatible brand 'iso2'
        0x61, 0x76, 0x63, 0x31,  # compatible brand 'avc1'
        0x6D, 0x70, 0x34, 0x31,  # compatible brand 'mp41'
    ])
    
    with open(test_video, "wb") as f:
        f.write(mp4_header)
        f.write(b'\x00' * 5000)  # 5KB of data
    
    return test_video

def test_new_system():
    """Test that server is using NEW system"""
    print("🧪 TESTING NEW SYSTEM")
    
    test_video = create_test_video()
    print(f"📁 Created test video: {test_video}")
    
    try:
        # Upload video
        with open(test_video, 'rb') as f:
            files = {'file': (test_video, f, 'video/mp4')}
            data = {'analysis_options': '{}'}
            
            print("🚀 Uploading test video...")
            response = requests.post('http://localhost:8000/analyze/comprehensive', 
                                   files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                session_id = result['session_id']
                message = result.get('message', '')
                
                print(f"✅ Upload successful! Session ID: {session_id}")
                print(f"📝 Message: {message}")
                
                # Check if message indicates new system
                if "REAL analysis" in message and "NO FALLBACKS" in message:
                    print("✅ SERVER IS USING NEW SYSTEM!")
                    return True
                else:
                    print(f"❌ Server still using old system. Message: {message}")
                    return False
                    
            else:
                print(f"❌ Upload failed: {response.status_code}")
                print(response.text)
                return False
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        # Cleanup
        if Path(test_video).exists():
            Path(test_video).unlink()
            print(f"🗑️ Cleaned up {test_video}")

if __name__ == "__main__":
    success = test_new_system()
    if success:
        print("\n🎉 NEW SYSTEM IS ACTIVE!")
        print("✅ Server is using RealVideoAnalyzer")
        print("✅ No more fallback outputs")
    else:
        print("\n❌ OLD SYSTEM STILL ACTIVE!")
        print("🔧 Server needs to be restarted")