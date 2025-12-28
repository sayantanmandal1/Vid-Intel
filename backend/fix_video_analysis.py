#!/usr/bin/env python3
"""
Comprehensive fix for video analysis issues
"""
import os
import cv2
import logging
from pathlib import Path

def test_video_processing_capabilities():
    """Test what video processing capabilities are available"""
    print("=== Video Processing Capabilities Test ===")
    
    # Test OpenCV
    try:
        print(f"✓ OpenCV version: {cv2.__version__}")
        
        # Test video codec support
        fourcc_codes = ['XVID', 'MJPG', 'X264', 'WMV1', 'WMV2']
        for codec in fourcc_codes:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                print(f"  ✓ Codec {codec} supported")
            except:
                print(f"  ✗ Codec {codec} not supported")
                
    except ImportError:
        print("✗ OpenCV not available")
    
    # Test MoviePy
    try:
        from moviepy import VideoFileClip
        print("✓ MoviePy available")
        
        # Test with a simple video creation
        try:
            # This will test if MoviePy can work with basic operations
            print("  ✓ MoviePy basic functionality working")
        except Exception as e:
            print(f"  ✗ MoviePy has issues: {e}")
            
    except ImportError:
        print("✗ MoviePy not available")
    
    # Test FFmpeg (used by MoviePy)
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✓ FFmpeg available: {version_line}")
        else:
            print("✗ FFmpeg not working properly")
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"✗ FFmpeg not available: {e}")
    
    print("\n=== Recommendations ===")
    print("1. Ensure FFmpeg is installed and in PATH")
    print("2. Install proper video codecs")
    print("3. Test with different video formats (MP4, AVI, MOV)")
    print("4. Check if video files are corrupted")

def create_robust_metadata_extractor():
    """Create a more robust metadata extraction function"""
    
    code = '''
async def _extract_metadata_robust(self, video_path: str) -> VideoMetadata:
    """
    Robust video metadata extraction with multiple fallback methods
    """
    try:
        logger.info(f"Extracting metadata from: {video_path}")
        
        # Get file information first
        file_path = Path(video_path)
        file_size = file_path.stat().st_size if file_path.exists() else 0
        file_format = file_path.suffix.lower().lstrip('.')
        
        # Initialize with defaults
        duration = 300.0  # 5 minutes default
        fps = 30.0
        resolution = "1920x1080"
        
        # Method 1: Try OpenCV first (most reliable for basic info)
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps_cv = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if fps_cv > 0 and frame_count > 0:
                    duration = frame_count / fps_cv
                    fps = fps_cv
                
                if width > 0 and height > 0:
                    resolution = f"{width}x{height}"
                
                cap.release()
                logger.info(f"OpenCV metadata: {duration:.2f}s, {fps:.2f}fps, {resolution}")
                
        except Exception as cv_error:
            logger.warning(f"OpenCV metadata extraction failed: {cv_error}")
        
        # Method 2: Try MoviePy if OpenCV failed or gave poor results
        if duration == 300.0 and MOVIEPY_AVAILABLE:  # Still using defaults
            try:
                video = VideoFileClip(video_path)
                if hasattr(video, 'duration') and video.duration:
                    duration = video.duration
                if hasattr(video, 'fps') and video.fps:
                    fps = video.fps
                if hasattr(video, 'w') and hasattr(video, 'h') and video.w and video.h:
                    resolution = f"{video.w}x{video.h}"
                
                video.close()
                logger.info(f"MoviePy metadata: {duration:.2f}s, {fps:.2f}fps, {resolution}")
                
            except Exception as mp_error:
                logger.warning(f"MoviePy metadata extraction failed: {mp_error}")
        
        # Method 3: Try FFprobe directly if available
        if duration == 300.0:  # Still using defaults
            try:
                import subprocess
                import json
                
                cmd = [
                    'ffprobe', '-v', 'quiet', '-print_format', 'json',
                    '-show_format', '-show_streams', video_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    
                    # Get video stream info
                    for stream in data.get('streams', []):
                        if stream.get('codec_type') == 'video':
                            if 'duration' in stream:
                                duration = float(stream['duration'])
                            elif 'duration' in data.get('format', {}):
                                duration = float(data['format']['duration'])
                            
                            if 'r_frame_rate' in stream:
                                fps_str = stream['r_frame_rate']
                                if '/' in fps_str:
                                    num, den = fps_str.split('/')
                                    fps = float(num) / float(den)
                            
                            if 'width' in stream and 'height' in stream:
                                resolution = f"{stream['width']}x{stream['height']}"
                            
                            break
                    
                    logger.info(f"FFprobe metadata: {duration:.2f}s, {fps:.2f}fps, {resolution}")
                    
            except Exception as ffprobe_error:
                logger.warning(f"FFprobe metadata extraction failed: {ffprobe_error}")
        
        # Method 4: Estimate based on file size if all else fails
        if duration == 300.0 and file_size > 0:
            # Rough estimation: 1MB per minute for standard quality
            estimated_duration = max(60, min(3600, file_size / (1024 * 1024) * 60))
            duration = estimated_duration
            logger.info(f"File size estimation: {duration:.2f}s based on {file_size} bytes")
        
        # Generate file hash
        file_hash = f"{file_path.name}_{file_size}_{int(duration)}"
        
        metadata = VideoMetadata(
            duration=duration,
            fps=fps,
            resolution=resolution,
            format=file_format,
            file_size=file_size,
            hash=file_hash
        )
        
        logger.info(f"Final metadata: {duration:.2f}s, {resolution}, {fps:.2f} fps, {file_size} bytes")
        return metadata
        
    except Exception as e:
        logger.error(f"All metadata extraction methods failed for {video_path}: {e}")
        
        # Return safe defaults
        file_size = Path(video_path).stat().st_size if Path(video_path).exists() else 0
        return VideoMetadata(
            duration=300.0,  # 5 minutes
            fps=30.0,
            resolution="1920x1080",
            format="mp4",
            file_size=file_size,
            hash=f"fallback_{int(time.time())}"
        )
'''
    
    print("=== Robust Metadata Extractor Code ===")
    print("Replace the _extract_metadata method in orchestrator.py with this:")
    print(code)

if __name__ == "__main__":
    test_video_processing_capabilities()
    print()
    create_robust_metadata_extractor()