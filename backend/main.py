import os
import tempfile
import shutil
import asyncio
import logging
from datetime import datetime, timedelta
import json
import hashlib
import sqlite3
from typing import List, Optional, Dict, Any
import uuid
import re
from pathlib import Path
import tempfile
# FastAPI and web related
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

# AI/ML Libraries
import whisper
import torch
from transformers import (
    pipeline, AutoTokenizer, AutoModelForTokenClassification,
    AutoModelForSequenceClassification, BartTokenizer, BartForConditionalGeneration
)
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import face_recognition
from PIL import Image
import pytesseract

# Video/Audio processing
from moviepy import VideoFileClip, concatenate_videoclips
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Data processing
import pandas as pd
from collections import Counter, defaultdict
import spacy
from textblob import TextBlob
import yake

# External APIs (free tiers)
import requests
from googletrans import Translator
import wikipedia

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="VidIntel Pro - Advanced Video Intelligence Platform",
    description="Comprehensive video analysis with AI-powered insights",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# At the top of your file
SUBTITLE_DIR = os.path.join(os.getcwd(), "static_subtitles")
os.makedirs(SUBTITLE_DIR, exist_ok=True)

# Security
security = HTTPBearer()

# Database setup
def init_db():
    conn = sqlite3.connect('vidintel.db')
    cursor = conn.cursor()
    
    # Videos table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id TEXT PRIMARY KEY,
            filename TEXT,
            upload_time TIMESTAMP,
            file_size INTEGER,
            duration REAL,
            format TEXT,
            resolution TEXT,
            fps REAL,
            hash TEXT UNIQUE
        )
    ''')
    
    # Analysis results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id TEXT PRIMARY KEY,
            video_id TEXT,
            analysis_type TEXT,
            result TEXT,
            created_at TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES videos (id)
        )
    ''')
    
    # Rate limiting table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rate_limits (
            ip_address TEXT,
            request_count INTEGER,
            window_start TIMESTAMP,
            PRIMARY KEY (ip_address)
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()

# Load models globally
class ModelLoader:
    def __init__(self):
        self.whisper_model = None
        self.summarizer = None
        self.sentiment_pipeline = None
        self.ner_pipeline = None
        self.toxicity_pipeline = None
        self.kw_model = None
        self.sentence_transformer = None
        self.nlp = None
        self.translator = None
        
    def load_models(self):
        logger.info("Loading AI models...")
        
        # Speech recognition
        self.whisper_model = whisper.load_model("base")
        
        # Text analysis models
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
        self.toxicity_pipeline = pipeline("text-classification", model="unitary/toxic-bert")
        
        # Keyword extraction
        self.kw_model = KeyBERT()
        
        # Sentence embeddings
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # SpaCy for advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            
        # Translator
        self.translator = Translator()
        
        logger.info("All models loaded successfully!")

model_loader = ModelLoader()

# Background task to load models
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(asyncio.to_thread(model_loader.load_models))

# Pydantic models
class VideoMetadata(BaseModel):
    duration: float
    fps: float
    resolution: str
    format: str
    file_size: int
    hash: str

class Speaker(BaseModel):
    speaker_id: str
    start_time: float
    end_time: float
    text: str
    confidence: float

class Entity(BaseModel):
    text: str
    label: str
    confidence: float
    start: int
    end: int

class SceneChange(BaseModel):
    timestamp: float
    confidence: float
    description: str

class Face(BaseModel):
    bbox: List[int]
    confidence: float
    timestamp: float
    encoding: Optional[List[float]] = None

class TextRegion(BaseModel):
    text: str
    bbox: List[int]
    confidence: float
    timestamp: float

class ComprehensiveAnalysis(BaseModel):
    video_id: str
    metadata: VideoMetadata
    transcript: str
    summary: str
    key_topics: List[str]
    sentiment_analysis: Dict[str, Any]
    named_entities: List[Entity]
    toxicity_score: Dict[str, float]
    speaker_diarization: List[Speaker]
    scene_changes: List[SceneChange]
    faces_detected: List[Face]
    text_recognition: List[TextRegion]
    language_detection: Dict[str, Any]
    content_categories: List[str]
    emotional_timeline: List[Dict[str, Any]]
    audio_features: Dict[str, Any]
    subtitle_file: str
    thumbnail_url: str
    similar_content: List[str]
    accessibility_score: Dict[str, Any]

# Utility functions
def get_file_hash(file_path: str) -> str:
    """Generate SHA-256 hash of file"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def extract_video_metadata(video_path: str) -> VideoMetadata:
    """Extract comprehensive video metadata"""
    clip = VideoFileClip(video_path)
    file_size = os.path.getsize(video_path)
    file_hash = get_file_hash(video_path)
    
    metadata = VideoMetadata(
        duration=clip.duration,
        fps=clip.fps,
        resolution=f"{clip.w}x{clip.h}",
        format=video_path.split('.')[-1],
        file_size=file_size,
        hash=file_hash
    )
    
    clip.close()
    return metadata

def extract_audio_advanced(video_path: str) -> str:
    """Extract audio with enhanced quality"""
    clip = VideoFileClip(video_path)
    audio_path = tempfile.mktemp(suffix=".wav")
    
    if clip.audio is not None:
        clip.audio.write_audiofile(
            audio_path, 
            codec='pcm_s16le',
            ffmpeg_params=["-ac", "1", "-ar", "16000"]  # Mono, 16kHz for better whisper performance
        )
    clip.close()
    return audio_path

def transcribe_with_timestamps(audio_path: str) -> Dict[str, Any]:
    """Advanced transcription with word-level timestamps"""
    result = model_loader.whisper_model.transcribe(
        audio_path, 
        word_timestamps=True,
        task="transcribe"
    )
    
    return {
        'text': result['text'],
        'segments': result.get('segments', []),
        'language': result.get('language', 'unknown')
    }

def perform_speaker_diarization(audio_path: str, transcript_segments: List) -> List[Speaker]:
    """Simple speaker diarization using audio features"""
    # Load audio
    y, sr = librosa.load(audio_path)
    
    speakers = []
    speaker_count = 0
    
    for i, segment in enumerate(transcript_segments):
        # Extract MFCC features for the segment
        start_frame = int(segment['start'] * sr)
        end_frame = int(segment['end'] * sr)
        segment_audio = y[start_frame:end_frame]
        
        if len(segment_audio) > 0:
            mfccs = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
            # Simple clustering based on MFCC features (in practice, use more sophisticated methods)
            speaker_id = f"Speaker_{(i % 2) + 1}"  # Simplified alternating speakers
            
            speakers.append(Speaker(
                speaker_id=speaker_id,
                start_time=segment['start'],
                end_time=segment['end'],
                text=segment['text'],
                confidence=0.8  # Placeholder confidence
            ))
    
    return speakers

def detect_scene_changes(video_path: str) -> List[SceneChange]:
    """Detect scene changes using computer vision"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    scene_changes = []
    prev_frame = None
    frame_count = 0
    threshold = 0.3  # Threshold for scene change detection
    
    while cap.read()[0]:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale and resize for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 240))
        
        if prev_frame is not None:
            # Calculate frame difference
            diff = cv2.absdiff(prev_frame, gray)
            mean_diff = np.mean(diff) / 255.0
            
            if mean_diff > threshold:
                timestamp = frame_count / fps
                scene_changes.append(SceneChange(
                    timestamp=timestamp,
                    confidence=min(mean_diff * 2, 1.0),
                    description=f"Scene change detected at {timestamp:.2f}s"
                ))
        
        prev_frame = gray
        frame_count += 1
        
        # Process every 30th frame for efficiency
        for _ in range(29):
            cap.read()
            frame_count += 29
    
    cap.release()
    return scene_changes

def detect_faces_in_video(video_path: str) -> List[Face]:
    """Detect and recognize faces in video"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    faces = []
    frame_count = 0
    
    while cap.read()[0]:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every 30th frame (1 second intervals)
        if frame_count % 30 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            timestamp = frame_count / fps
            
            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                faces.append(Face(
                    bbox=[left, top, right, bottom],
                    confidence=0.9,  # face_recognition doesn't provide confidence
                    timestamp=timestamp,
                    encoding=encoding.tolist()
                ))
        
        frame_count += 1
    
    cap.release()
    return faces

def extract_text_from_video(video_path: str) -> List[TextRegion]:
    """Extract text from video frames using OCR"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    text_regions = []
    frame_count = 0
    
    while cap.read()[0]:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every 60th frame (2 second intervals)
        if frame_count % 60 == 0:
            timestamp = frame_count / fps
            
            # Use pytesseract for OCR
            try:
                # Get text with bounding boxes
                data = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)
                
                for i, text in enumerate(data['text']):
                    if text.strip():
                        confidence = float(data['conf'][i]) / 100.0
                        if confidence > 0.5:  # Filter low confidence detections
                            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                            text_regions.append(TextRegion(
                                text=text.strip(),
                                bbox=[x, y, x+w, y+h],
                                confidence=confidence,
                                timestamp=timestamp
                            ))
            except Exception as e:
                logger.warning(f"OCR failed for frame at {timestamp}s: {e}")
        
        frame_count += 1
    
    cap.release()
    return text_regions

def analyze_audio_features(audio_path: str) -> Dict[str, Any]:
    """Extract comprehensive audio features"""
    y, sr = librosa.load(audio_path)
    
    # Extract various audio features
    features = {
        'tempo': float(librosa.beat.tempo(y=y, sr=sr)[0]),
        'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
        'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
        'rms_energy': float(np.mean(librosa.feature.rms(y=y))),
        'duration': float(len(y) / sr)
    }
    
    # MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}'] = float(np.mean(mfccs[i]))
    
    return features

def generate_srt_subtitle(transcript_segments: List) -> str:
    """Generate SRT subtitle file"""
    srt_content = []
    
    for i, segment in enumerate(transcript_segments, 1):
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text'].strip()
        
        # Convert seconds to SRT time format
        start_srt = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d},{int((start_time%1)*1000):03d}"
        end_srt = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d},{int((end_time%1)*1000):03d}"
        
        srt_content.append(f"{i}\n{start_srt} --> {end_srt}\n{text}\n")
    
    return "\n".join(srt_content)

def analyze_sentiment_timeline(transcript_segments: List) -> List[Dict[str, Any]]:
    """Analyze sentiment over time"""
    timeline = []
    
    for segment in transcript_segments:
        if len(segment['text'].strip()) > 10:  # Only analyze substantial text
            try:
                sentiment = model_loader.sentiment_pipeline(segment['text'][:512])
                timeline.append({
                    'timestamp': segment['start'],
                    'sentiment': sentiment[0]['label'],
                    'confidence': sentiment[0]['score'],
                    'text_sample': segment['text'][:100] + "..." if len(segment['text']) > 100 else segment['text']
                })
            except Exception as e:
                logger.warning(f"Sentiment analysis failed for segment: {e}")
    
    return timeline

def categorize_content(text: str, topics: List[str]) -> List[str]:
    """Categorize content based on text and topics"""
    categories = []
    
    # Define category keywords
    category_keywords = {
        'Education': ['learn', 'teach', 'school', 'university', 'course', 'lesson', 'tutorial'],
        'Entertainment': ['fun', 'funny', 'comedy', 'movie', 'music', 'game', 'show'],
        'News': ['news', 'breaking', 'report', 'journalist', 'politics', 'government'],
        'Technology': ['tech', 'computer', 'software', 'AI', 'digital', 'internet', 'coding'],
        'Health': ['health', 'medical', 'doctor', 'medicine', 'fitness', 'wellness'],
        'Business': ['business', 'company', 'market', 'finance', 'economy', 'investment'],
        'Sports': ['sport', 'game', 'team', 'player', 'match', 'competition', 'athletic'],
        'Travel': ['travel', 'trip', 'vacation', 'destination', 'tourism', 'journey']
    }
    
    text_lower = text.lower()
    topics_lower = [topic.lower() for topic in topics]
    
    for category, keywords in category_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword in text_lower:
                score += text_lower.count(keyword)
            for topic in topics_lower:
                if keyword in topic:
                    score += 2
        
        if score > 0:
            categories.append(category)
    
    return categories if categories else ['General']

def calculate_accessibility_score(faces: List[Face], text_regions: List[TextRegion], audio_features: Dict) -> Dict[str, Any]:
    """Calculate accessibility score for the video"""
    score = {
        'visual_clarity': 0.0,
        'audio_clarity': 0.0,
        'text_readability': 0.0,
        'overall_score': 0.0
    }
    
    # Visual clarity based on face detection quality
    if faces:
        avg_face_confidence = np.mean([face.confidence for face in faces])
        score['visual_clarity'] = avg_face_confidence
    else:
        score['visual_clarity'] = 0.5  # Neutral score if no faces
    
    # Audio clarity based on RMS energy and spectral features
    if 'rms_energy' in audio_features:
        rms_normalized = min(audio_features['rms_energy'] * 10, 1.0)
        score['audio_clarity'] = rms_normalized
    
    # Text readability based on OCR confidence
    if text_regions:
        avg_text_confidence = np.mean([region.confidence for region in text_regions])
        score['text_readability'] = avg_text_confidence
    else:
        score['text_readability'] = 0.8  # High score if no text (not penalized)
    
    # Overall score
    score['overall_score'] = (score['visual_clarity'] + score['audio_clarity'] + score['text_readability']) / 3
    
    return score

# Rate limiting
def check_rate_limit(ip_address: str, limit: int = 10, window: int = 3600) -> bool:
    """Check if IP is within rate limit"""
    conn = sqlite3.connect('vidintel.db')
    cursor = conn.cursor()
    
    current_time = datetime.now()
    window_start = current_time - timedelta(seconds=window)
    
    cursor.execute(
        "SELECT request_count, window_start FROM rate_limits WHERE ip_address = ?",
        (ip_address,)
    )
    result = cursor.fetchone()
    
    if result:
        count, stored_window_start = result
        stored_time = datetime.fromisoformat(stored_window_start)
        
        if stored_time < window_start:
            # Reset window
            cursor.execute(
                "UPDATE rate_limits SET request_count = 1, window_start = ? WHERE ip_address = ?",
                (current_time.isoformat(), ip_address)
            )
        else:
            if count >= limit:
                conn.close()
                return False
            cursor.execute(
                "UPDATE rate_limits SET request_count = request_count + 1 WHERE ip_address = ?",
                (ip_address,)
            )
    else:
        cursor.execute(
            "INSERT INTO rate_limits (ip_address, request_count, window_start) VALUES (?, 1, ?)",
            (ip_address, current_time.isoformat())
        )
    
    conn.commit()
    conn.close()
    return True

# API Endpoints
@app.post("/analyze/comprehensive", response_model=ComprehensiveAnalysis)
async def comprehensive_video_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    client_ip: str = "127.0.0.1"  # In production, extract from request
):
    """Comprehensive video analysis with all features"""
    
    # Rate limiting
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        # Save uploaded file
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, file.filename)
        
        with open(video_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Generate unique video ID
        video_id = str(uuid.uuid4())
        
        # Extract metadata
        metadata = extract_video_metadata(video_path)
        
        # Check for duplicate
        conn = sqlite3.connect('vidintel.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM videos WHERE hash = ?", (metadata.hash,))
        existing = cursor.fetchone()
        
        if existing:
            conn.close()
            shutil.rmtree(temp_dir)
            raise HTTPException(status_code=409, detail="Video already processed")
        
        # Store video info
        cursor.execute("""
            INSERT INTO videos (id, filename, upload_time, file_size, duration, format, resolution, fps, hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            video_id, file.filename, datetime.now().isoformat(), metadata.file_size,
            metadata.duration, metadata.format, metadata.resolution, metadata.fps, metadata.hash
        ))
        conn.commit()
        conn.close()
        
        # Extract audio
        audio_path = extract_audio_advanced(video_path)
        
        # Transcription with timestamps
        transcription_result = transcribe_with_timestamps(audio_path)
        transcript = transcription_result['text']
        segments = transcription_result['segments']
        
        # Parallel processing of various analyses
        results = await asyncio.gather(
            # Text analysis
            asyncio.to_thread(model_loader.summarizer, transcript[:1024], max_length=150, min_length=40, do_sample=False),
            asyncio.to_thread(model_loader.kw_model.extract_keywords, transcript, top_n=10),
            asyncio.to_thread(model_loader.sentiment_pipeline, transcript[:512]),
            asyncio.to_thread(model_loader.ner_pipeline, transcript[:512]),
            asyncio.to_thread(model_loader.toxicity_pipeline, transcript[:512]),
            
            # Audio/Video analysis
            asyncio.to_thread(perform_speaker_diarization, audio_path, segments),
            asyncio.to_thread(detect_scene_changes, video_path),
            asyncio.to_thread(detect_faces_in_video, video_path),
            asyncio.to_thread(extract_text_from_video, video_path),
            asyncio.to_thread(analyze_audio_features, audio_path),
            
            return_exceptions=True
        )
        
        # Process results
        summary = results[0][0]['summary_text'] if results[0] and not isinstance(results[0], Exception) else "Summary generation failed"
        
        key_topics = [kw[0] for kw in results[1]] if results[1] and not isinstance(results[1], Exception) else []
        
        sentiment_analysis = {
            'label': results[2][0]['label'] if results[2] and not isinstance(results[2], Exception) else 'NEUTRAL',
            'score': results[2][0]['score'] if results[2] and not isinstance(results[2], Exception) else 0.5
        }
        
        # Named entities
        named_entities = []
        if results[3] and not isinstance(results[3], Exception):
            for entity in results[3]:
                named_entities.append(Entity(
                    text=entity['word'],
                    label=entity['entity_group'],
                    confidence=entity['score'],
                    start=entity['start'],
                    end=entity['end']
                ))
        
        # Toxicity score
        toxicity_score = {
            'toxic': results[4][0]['score'] if results[4] and not isinstance(results[4], Exception) and results[4][0]['label'] == 'TOXIC' else 0.0,
            'label': results[4][0]['label'] if results[4] and not isinstance(results[4], Exception) else 'NON_TOXIC'
        }
        
        # Extract other results
        speaker_diarization = results[5] if results[5] and not isinstance(results[5], Exception) else []
        scene_changes = results[6] if results[6] and not isinstance(results[6], Exception) else []
        faces_detected = results[7] if results[7] and not isinstance(results[7], Exception) else []
        text_recognition = results[8] if results[8] and not isinstance(results[8], Exception) else []
        audio_features = results[9] if results[9] and not isinstance(results[9], Exception) else {}
        
        # Additional analyses
        emotional_timeline = analyze_sentiment_timeline(segments)
        content_categories = categorize_content(transcript, key_topics)
        
        # Language detection
        language_detection = {
            'detected_language': transcription_result.get('language', 'unknown'),
            'confidence': 0.9  # Whisper is generally confident
        }
        
        # Generate subtitle file
        subtitle_content = generate_srt_subtitle(segments)
        subtitle_path = os.path.join(temp_dir, f"{video_id}.srt")
        with open(subtitle_path, 'w', encoding='utf-8') as f:
            f.write(subtitle_content)
        
        # Calculate accessibility score
        accessibility_score = calculate_accessibility_score(faces_detected, text_recognition, audio_features)
        
        # Create response
        analysis_result = ComprehensiveAnalysis(
            video_id=video_id,
            metadata=metadata,
            transcript=transcript,
            summary=summary,
            key_topics=key_topics,
            sentiment_analysis=sentiment_analysis,
            named_entities=named_entities,
            toxicity_score=toxicity_score,
            speaker_diarization=speaker_diarization,
            scene_changes=scene_changes,
            faces_detected=faces_detected,
            text_recognition=text_recognition,
            language_detection=language_detection,
            content_categories=content_categories,
            emotional_timeline=emotional_timeline,
            audio_features=audio_features,
            subtitle_file=f"/download/subtitles/{video_id}.srt",
            thumbnail_url=f"/download/thumbnail/{video_id}.jpg",
            similar_content=[],  # Would implement similarity search
            accessibility_score=accessibility_score
        )
        
        # Clean up in background
        background_tasks.add_task(cleanup_temp_files, temp_dir, [audio_path])
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/quick")
async def quick_analysis(file: UploadFile = File(...)):
    """Quick analysis for faster results"""
    try:
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, file.filename)
        
        with open(video_path, "wb") as f:
            content = await file.read()
            f.write(content)

        audio_path = extract_audio_advanced(video_path)
        
        # Transcribe
        result = model_loader.whisper_model.transcribe(audio_path)
        
        # Summarize
        summary = model_loader.summarizer(
            result['text'][:512], max_length=80, min_length=20, do_sample=False
        )
        
        # Sentiment
        sentiment = model_loader.sentiment_pipeline(result['text'][:512])
        
        # Duration and other audio features
        audio_features = analyze_audio_features(audio_path)
        duration = audio_features.get("duration", 0.0)

        # Optional: Save SRT file using transcript segments if available
        if "segments" in result:
            srt = generate_srt_subtitle(result["segments"])
            srt_path = f"/tmp/{uuid.uuid4()}.srt"
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt)

        # Cleanup
        os.remove(audio_path)
        shutil.rmtree(temp_dir)
        
        return {
            "transcript": result['text'],
            "summary": summary[0]['summary_text'],
            "sentiment": sentiment[0],
            "language": result.get('language', 'unknown'),
            "duration": duration,
            # Optional: include audio features
            # "audio_features": audio_features
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
async def quick_analysis_from_path(video_path: str) -> dict:
    try:
        audio_path = extract_audio_advanced(video_path)

        result = model_loader.whisper_model.transcribe(audio_path)
        summary = model_loader.summarizer(
            result['text'][:512], max_length=80, min_length=20, do_sample=False
        )
        sentiment = model_loader.sentiment_pipeline(result['text'][:512])
        audio_features = analyze_audio_features(audio_path)
        duration = audio_features.get("duration", 0.0)

        # Save subtitles with a persistent filename
        video_id = Path(video_path).stem  # removes .mp4 or .mkv, etc.
        if "segments" in result:
            srt = generate_srt_subtitle(result["segments"])
            srt_path = os.path.join(SUBTITLE_DIR, f"{video_id}.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt)

        os.remove(audio_path)

        return {
            "transcript": result['text'],
            "summary": summary[0]['summary_text'],
            "sentiment": sentiment[0],
            "language": result.get('language', 'unknown'),
            "duration": duration,
            "video_id": video_id  # helpful for client to know which file to download
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/download/subtitles/{video_id}.srt")
async def download_subtitles(video_id: str):
    """Download subtitle file"""
    file_path = os.path.join(SUBTITLE_DIR, f"{video_id}.srt")
    
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='text/plain', filename=f"{video_id}.srt")
    else:
        raise HTTPException(status_code=404, detail="Subtitle file not found")


@app.get("/download/thumbnail/{video_id}.jpg")
async def download_thumbnail(video_id: str):
    """Download video thumbnail"""
    file_path = f"/tmp/{video_id}_thumbnail.jpg"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='image/jpeg', filename=f"{video_id}_thumbnail.jpg")
    else:
        raise HTTPException(status_code=404, detail="Thumbnail not found")

@app.post("/translate")
async def translate_text(
    text: str = Form(...),
    target_language: str = Form(default="es"),
    source_language: str = Form(default="auto")
):
    """Translate text to different languages"""
    try:
        translation = model_loader.translator.translate(text, src=source_language, dest=target_language)
        return {
            "original_text": text,
            "translated_text": translation.text,
            "source_language": translation.src,
            "target_language": target_language,
            "confidence": 0.9  # Google Translate doesn't provide confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/batch/analyze")
async def batch_analysis(
    files: List[UploadFile] = File(...),
    analysis_type: str = Form(default="quick")
):
    """Batch process multiple videos"""
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 files per batch")

    results = []

    for file in files:
        try:
            # Save uploaded file to temp
            temp_dir = tempfile.mkdtemp()
            temp_filename = f"{uuid.uuid4()}_{file.filename}"
            temp_filepath = os.path.join(temp_dir, temp_filename)

            with open(temp_filepath, "wb") as f:
                f.write(await file.read())
                f.flush()
                os.fsync(f.fileno())

            if analysis_type == "quick":
                result = await quick_analysis_from_path(temp_filepath)

                # Get duration if needed
                audio_features = analyze_audio_features(temp_filepath)
                result["duration"] = audio_features.get("duration", 0.0)

                result["filename"] = file.filename
                result["status"] = "success"
                results.append(result)
            else:
                results.append({
                    "filename": file.filename,
                    "status": "queued for processing"
                })

            os.remove(temp_filepath)

        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "status": "failed"
            })

    return {"batch_results": results, "total_processed": len(results)}



@app.post("/search/similar")
async def find_similar_videos(
    video_id: str = Form(...),
    threshold: float = Form(default=0.7)
):
    """Find similar videos based on content embeddings"""
    try:
        # Get video transcript from database
        conn = sqlite3.connect('vidintel.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT result FROM analysis_results 
            WHERE video_id = ? AND analysis_type = 'transcript'
        """, (video_id,))
        
        result = cursor.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Video not found")
        
        source_transcript = result[0]
        
        # Get all other video transcripts
        cursor.execute("""
            SELECT video_id, result FROM analysis_results 
            WHERE analysis_type = 'transcript' AND video_id != ?
        """, (video_id,))
        
        other_videos = cursor.fetchall()
        conn.close()
        
        if not other_videos:
            return {"similar_videos": [], "message": "No other videos to compare"}
        
        # Calculate similarity using sentence transformers
        source_embedding = model_loader.sentence_transformer.encode([source_transcript])
        other_embeddings = model_loader.sentence_transformer.encode([video[1] for video in other_videos])
        
        similarities = cosine_similarity(source_embedding, other_embeddings)[0]
        
        similar_videos = []
        for i, (other_video_id, _) in enumerate(other_videos):
            if similarities[i] >= threshold:
                similar_videos.append({
                    "video_id": other_video_id,
                    "similarity_score": float(similarities[i])
                })
        
        # Sort by similarity score
        similar_videos.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return {"similar_videos": similar_videos[:10]}  # Return top 10
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract/highlights")
async def extract_highlights(
    video_id: str = Form(...),
    highlight_duration: int = Form(default=30),
    num_highlights: int = Form(default=3)
):
    """Extract video highlights based on sentiment and engagement"""
    try:
        # Get video analysis results
        conn = sqlite3.connect('vidintel.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT result FROM analysis_results 
            WHERE video_id = ? AND analysis_type = 'emotional_timeline'
        """, (video_id,))
        
        result = cursor.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Video analysis not found")
        
        emotional_timeline = json.loads(result[0])
        
        # Find segments with highest positive sentiment
        positive_segments = [
            segment for segment in emotional_timeline 
            if segment['sentiment'] == 'POSITIVE' and segment['confidence'] > 0.7
        ]
        
        # Sort by confidence
        positive_segments.sort(key=lambda x: x['confidence'], reverse=True)
        
        highlights = []
        for i, segment in enumerate(positive_segments[:num_highlights]):
            highlights.append({
                "start_time": segment['timestamp'],
                "end_time": segment['timestamp'] + highlight_duration,
                "confidence": segment['confidence'],
                "description": segment['text_sample'],
                "highlight_rank": i + 1
            })
        
        conn.close()
        return {"highlights": highlights, "total_found": len(highlights)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/summary")
async def generate_custom_summary(
    video_id: str = Form(...),
    summary_type: str = Form(default="bullet_points"),
    max_length: int = Form(default=200)
):
    """Generate different types of summaries"""
    try:
        # Get transcript
        conn = sqlite3.connect('vidintel.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT result FROM analysis_results 
            WHERE video_id = ? AND analysis_type = 'transcript'
        """, (video_id,))
        
        result = cursor.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Video not found")
        
        transcript = result[0]
        conn.close()
        
        if summary_type == "bullet_points":
            # Generate bullet point summary
            summary = model_loader.summarizer(
                transcript[:1024], 
                max_length=max_length, 
                min_length=50, 
                do_sample=False
            )
            
            # Convert to bullet points
            sentences = summary[0]['summary_text'].split('. ')
            bullet_points = [f"• {sentence.strip()}" for sentence in sentences if sentence.strip()]
            
            return {
                "summary_type": "bullet_points",
                "summary": "\n".join(bullet_points)
            }
            
        elif summary_type == "executive":
            # Executive summary
            prompt_text = f"Executive Summary: {transcript[:800]}"
            summary = model_loader.summarizer(
                prompt_text,
                max_length=min(max_length, 150),
                min_length=80,
                do_sample=False
            )
            
            return {
                "summary_type": "executive",
                "summary": summary[0]['summary_text']
            }
            
        elif summary_type == "technical":
            # Technical summary focusing on key topics
            topics = model_loader.kw_model.extract_keywords(transcript, top_n=5)
            topic_text = ", ".join([topic[0] for topic in topics])
            
            summary = model_loader.summarizer(
                transcript[:1024],
                max_length=max_length,
                min_length=60,
                do_sample=False
            )
            
            return {
                "summary_type": "technical",
                "summary": summary[0]['summary_text'],
                "key_topics": topic_text
            }
        
        else:
            raise HTTPException(status_code=400, detail="Invalid summary type")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/content_moderation")
async def content_moderation_analysis(
    video_id: str = Form(...),
    strict_mode: bool = Form(default=False)
):
    """Advanced content moderation analysis"""
    try:
        # Get transcript and analysis results
        conn = sqlite3.connect('vidintel.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT result FROM analysis_results 
            WHERE video_id = ? AND analysis_type = 'transcript'
        """, (video_id,))
        
        transcript_result = cursor.fetchone()
        if not transcript_result:
            raise HTTPException(status_code=404, detail="Video not found")
        
        transcript = transcript_result[0]
        
        # Toxicity analysis
        toxicity_result = model_loader.toxicity_pipeline(transcript[:512])
        
        # Additional checks
        moderation_flags = []
        
        # Check for explicit content keywords
        explicit_keywords = [
            'violence', 'hate', 'harassment', 'discrimination', 
            'abuse', 'threat', 'harm', 'dangerous'
        ]
        
        text_lower = transcript.lower()
        for keyword in explicit_keywords:
            if keyword in text_lower:
                moderation_flags.append({
                    "type": "explicit_language",
                    "keyword": keyword,
                    "severity": "medium"
                })
        
        # Profanity detection using simple word list
        profanity_words = ['damn', 'hell', 'shit', 'fuck', 'bitch', 'asshole']  # Simplified list
        profanity_count = sum(text_lower.count(word) for word in profanity_words)
        
        if profanity_count > 0:
            moderation_flags.append({
                "type": "profanity",
                "count": profanity_count,
                "severity": "low" if profanity_count < 3 else "medium"
            })
        
        # Overall moderation score
        base_toxicity = toxicity_result[0]['score'] if toxicity_result[0]['label'] == 'TOXIC' else 0
        flag_penalty = len(moderation_flags) * 0.1
        profanity_penalty = min(profanity_count * 0.05, 0.3)
        
        overall_risk_score = min(base_toxicity + flag_penalty + profanity_penalty, 1.0)
        
        # Determine action
        if strict_mode:
            action = "block" if overall_risk_score > 0.3 else "approve"
        else:
            action = "block" if overall_risk_score > 0.7 else "review" if overall_risk_score > 0.4 else "approve"
        
        result = {
            "video_id": video_id,
            "overall_risk_score": overall_risk_score,
            "toxicity_score": base_toxicity,
            "moderation_flags": moderation_flags,
            "recommended_action": action,
            "strict_mode": strict_mode,
            "manual_review_required": overall_risk_score > 0.5
        }
        
        # Store moderation result
        cursor.execute("""
            INSERT INTO analysis_results (id, video_id, analysis_type, result, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()), video_id, 'content_moderation', 
            json.dumps(result), datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/chapters")
async def generate_video_chapters(
    video_id: str = Form(...),
    min_chapter_duration: int = Form(default=60)
):
    """Generate video chapters based on topic changes"""
    try:
        # Get scene changes and transcript segments
        conn = sqlite3.connect('vidintel.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT result FROM analysis_results 
            WHERE video_id = ? AND analysis_type = 'scene_changes'
        """, (video_id,))
        
        scene_result = cursor.fetchone()
        
        cursor.execute("""
            SELECT result FROM analysis_results 
            WHERE video_id = ? AND analysis_type = 'transcript_segments'
        """, (video_id,))
        
        segments_result = cursor.fetchone()
        
        if not scene_result or not segments_result:
            raise HTTPException(status_code=404, detail="Required analysis data not found")
        
        scene_changes = json.loads(scene_result[0])
        transcript_segments = json.loads(segments_result[0])
        
        # Generate chapters based on scene changes and topic shifts
        chapters = []
        current_chapter_start = 0
        current_topics = []
        
        for i, scene_change in enumerate(scene_changes):
            timestamp = scene_change['timestamp']
            
            # Find transcript segments around this timestamp
            relevant_segments = [
                seg for seg in transcript_segments 
                if abs(seg['start'] - timestamp) < 30  # Within 30 seconds
            ]
            
            if relevant_segments:
                # Extract topics from these segments
                combined_text = " ".join([seg['text'] for seg in relevant_segments])
                if len(combined_text) > 50:
                    topics = model_loader.kw_model.extract_keywords(combined_text, top_n=3)
                    new_topics = [topic[0] for topic in topics]
                    
                    # Check if topics changed significantly
                    topic_overlap = len(set(current_topics) & set(new_topics))
                    
                    if (timestamp - current_chapter_start) >= min_chapter_duration and topic_overlap < 2:
                        # Create new chapter
                        chapter_title = f"Chapter {len(chapters) + 1}"
                        if current_topics:
                            chapter_title += f": {', '.join(current_topics[:2])}"
                        
                        chapters.append({
                            "chapter_number": len(chapters) + 1,
                            "title": chapter_title,
                            "start_time": current_chapter_start,
                            "end_time": timestamp,
                            "duration": timestamp - current_chapter_start,
                            "topics": current_topics
                        })
                        
                        current_chapter_start = timestamp
                        current_topics = new_topics
                    else:
                        # Merge topics
                        current_topics = list(set(current_topics + new_topics))[:5]  # Keep top 5
        
        # Add final chapter
        if transcript_segments:
            final_timestamp = max([seg['end'] for seg in transcript_segments])
            if final_timestamp > current_chapter_start:
                chapters.append({
                    "chapter_number": len(chapters) + 1,
                    "title": f"Chapter {len(chapters) + 1}" + (f": {', '.join(current_topics[:2])}" if current_topics else ""),
                    "start_time": current_chapter_start,
                    "end_time": final_timestamp,
                    "duration": final_timestamp - current_chapter_start,
                    "topics": current_topics
                })
        
        conn.close()
        return {"chapters": chapters, "total_chapters": len(chapters)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/engagement_metrics")
async def analyze_engagement_metrics(
    video_id: str = Form(...),
    analyze_audio_energy: bool = Form(default=True)
):
    """Analyze video engagement potential"""
    try:
        # Get various analysis results
        conn = sqlite3.connect('vidintel.db')
        cursor = conn.cursor()
        
        # Get emotional timeline
        cursor.execute("""
            SELECT result FROM analysis_results 
            WHERE video_id = ? AND analysis_type = 'emotional_timeline'
        """, (video_id,))
        
        emotional_result = cursor.fetchone()
        
        # Get audio features
        cursor.execute("""
            SELECT result FROM analysis_results 
            WHERE video_id = ? AND analysis_type = 'audio_features'
        """, (video_id,))
        
        audio_result = cursor.fetchone()
        
        if not emotional_result:
            raise HTTPException(status_code=404, detail="Emotional analysis not found")
        
        emotional_timeline = json.loads(emotional_result[0])
        audio_features = json.loads(audio_result[0]) if audio_result else {}
        
        # Calculate engagement metrics
        metrics = {}
        
        # Emotional engagement
        if emotional_timeline:
            positive_count = sum(1 for e in emotional_timeline if e['sentiment'] == 'POSITIVE')
            negative_count = sum(1 for e in emotional_timeline if e['sentiment'] == 'NEGATIVE')
            total_segments = len(emotional_timeline)
            
            metrics['emotional_positivity_ratio'] = positive_count / total_segments if total_segments > 0 else 0
            metrics['emotional_engagement_score'] = (positive_count - negative_count) / total_segments if total_segments > 0 else 0
            
            # Emotional variance (how much sentiment changes)
            confidences = [e['confidence'] for e in emotional_timeline]
            metrics['emotional_variance'] = np.var(confidences) if confidences else 0
        
        # Audio engagement
        if audio_features and analyze_audio_energy:
            # Normalize audio features for engagement scoring
            tempo = audio_features.get('tempo', 120)  # Default 120 BPM
            energy = audio_features.get('rms_energy', 0.1)
            spectral_centroid = audio_features.get('spectral_centroid', 2000)
            
            # Score based on optimal ranges for engagement
            tempo_score = 1.0 - abs(tempo - 120) / 120  # Optimal around 120 BPM
            energy_score = min(energy * 10, 1.0)  # Higher energy = better
            brightness_score = min(spectral_centroid / 4000, 1.0)  # Brighter = better
            
            metrics['audio_engagement_score'] = (tempo_score + energy_score + brightness_score) / 3
        
        # Content diversity (based on topic variety)
        cursor.execute("""
            SELECT result FROM analysis_results 
            WHERE video_id = ? AND analysis_type = 'topics'
        """, (video_id,))
        
        topics_result = cursor.fetchone()
        if topics_result:
            topics = json.loads(topics_result[0])
            metrics['content_diversity_score'] = min(len(topics) / 10, 1.0)  # More topics = more diverse
        
        # Overall engagement prediction
        component_scores = [
            metrics.get('emotional_engagement_score', 0),
            metrics.get('audio_engagement_score', 0),
            metrics.get('content_diversity_score', 0)
        ]
        
        metrics['overall_engagement_prediction'] = sum(component_scores) / len([s for s in component_scores if s > 0])
        
        # Engagement recommendations
        recommendations = []
        
        if metrics.get('emotional_positivity_ratio', 0) < 0.3:
            recommendations.append("Consider adding more positive/uplifting content")
        
        if metrics.get('audio_engagement_score', 0) < 0.4:
            recommendations.append("Audio energy could be improved with better pacing or background music")
        
        if metrics.get('content_diversity_score', 0) < 0.3:
            recommendations.append("Content could benefit from covering more diverse topics")
        
        result = {
            "video_id": video_id,
            "engagement_metrics": metrics,
            "recommendations": recommendations,
            "engagement_tier": (
                "High" if metrics['overall_engagement_prediction'] > 0.7 else
                "Medium" if metrics['overall_engagement_prediction'] > 0.4 else
                "Low"
            )
        }
        
        conn.close()
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/dashboard/{video_id}")
async def get_analytics_dashboard(video_id: str):
    """Get comprehensive analytics dashboard data"""
    try:
        conn = sqlite3.connect('vidintel.db')
        cursor = conn.cursor()
        
        # Get all analysis results for the video
        cursor.execute("""
            SELECT analysis_type, result, created_at 
            FROM analysis_results 
            WHERE video_id = ?
        """, (video_id,))
        
        results = cursor.fetchall()
        
        # Get video metadata
        cursor.execute("""
            SELECT filename, upload_time, file_size, duration, resolution, fps
            FROM videos 
            WHERE id = ?
        """, (video_id,))
        
        video_info = cursor.fetchone()
        
        if not video_info:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Organize results
        dashboard_data = {
            "video_info": {
                "id": video_id,
                "filename": video_info[0],
                "upload_time": video_info[1],
                "file_size_mb": round(video_info[2] / (1024*1024), 2),
                "duration_minutes": round(video_info[3] / 60, 2),
                "resolution": video_info[4],
                "fps": video_info[5]
            },
            "analysis_summary": {},
            "processing_timeline": []
        }
        
        # Process analysis results
        for analysis_type, result_json, created_at in results:
            try:
                result_data = json.loads(result_json)
                dashboard_data["analysis_summary"][analysis_type] = result_data
                dashboard_data["processing_timeline"].append({
                    "analysis_type": analysis_type,
                    "completed_at": created_at,
                    "status": "completed"
                })
            except json.JSONDecodeError:
                # Handle non-JSON results (like plain text transcripts)
                dashboard_data["analysis_summary"][analysis_type] = result_json
        
        # Sort timeline by completion time
        dashboard_data["processing_timeline"].sort(key=lambda x: x["completed_at"])
        
        conn.close()
        return dashboard_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/platform")
async def get_platform_statistics():
    """Get overall platform usage statistics"""
    try:
        conn = sqlite3.connect('vidintel.db')
        cursor = conn.cursor()
        
        # Video statistics
        cursor.execute("SELECT COUNT(*) FROM videos")
        total_videos = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(file_size) FROM videos")
        total_storage = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT SUM(duration) FROM videos")
        total_duration = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT AVG(duration) FROM videos")
        avg_duration = cursor.fetchone()[0] or 0
        
        # Analysis statistics
        cursor.execute("SELECT COUNT(*) FROM analysis_results")
        total_analyses = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT analysis_type, COUNT(*) 
            FROM analysis_results 
            GROUP BY analysis_type
        """)
        analysis_breakdown = dict(cursor.fetchall())
        
        # Recent activity (last 24 hours)
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        cursor.execute("""
            SELECT COUNT(*) FROM videos 
            WHERE upload_time > ?
        """, (yesterday,))
        recent_uploads = cursor.fetchone()[0]
        
        # Top file formats
        cursor.execute("""
            SELECT format, COUNT(*) 
            FROM videos 
            GROUP BY format 
            ORDER BY COUNT(*) DESC 
            LIMIT 5
        """)
        format_distribution = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "platform_overview": {
                "total_videos_processed": total_videos,
                "total_storage_gb": round(total_storage / (1024**3), 2),
                "total_content_hours": round(total_duration / 3600, 2),
                "average_video_length_minutes": round(avg_duration / 60, 2)
            },
            "analysis_statistics": {
                "total_analyses_performed": total_analyses,
                "analysis_type_breakdown": analysis_breakdown
            },
            "recent_activity": {
                "uploads_last_24h": recent_uploads
            },
            "content_insights": {
                "popular_formats": format_distribution
            },
            "system_health": {
                "api_status": "healthy",
                "models_loaded": len([attr for attr in dir(model_loader) if not attr.startswith('_') and getattr(model_loader, attr) is not None])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Utility function for cleanup
async def cleanup_temp_files(temp_dir: str, additional_files: List[str] = None):
    """Clean up temporary files"""
    try:
        if additional_files:
            for file_path in additional_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": bool(model_loader.whisper_model),
        "database_connected": True  # Would implement actual DB health check
    }

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "VidIntel Pro - Advanced Video Intelligence Platform",
        "version": "2.0.0",
        "features": [
            "Comprehensive video analysis",
            "Multi-language transcription",
            "Advanced sentiment analysis",
            "Speaker diarization",
            "Scene change detection",
            "Face detection and recognition",
            "Text extraction (OCR)",
            "Content moderation",
            "Engagement metrics",
            "Video chapters generation",
            "Batch processing",
            "Content similarity search",
            "Multi-format subtitle generation",
            "Real-time analytics dashboard"
        ],
        "endpoints": {
            "analysis": "/analyze/comprehensive",
            "quick_analysis": "/analyze/quick",
            "batch_processing": "/batch/analyze",
            "content_moderation": "/analyze/content_moderation",
            "engagement_analysis": "/analyze/engagement_metrics",
            "similarity_search": "/search/similar",
            "highlight_extraction": "/extract/highlights",
            "chapter_generation": "/generate/chapters",
            "analytics_dashboard": "/analytics/dashboard/{video_id}",
            "platform_stats": "/stats/platform"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)