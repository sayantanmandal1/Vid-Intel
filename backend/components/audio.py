"""
Audio Processor - Handles audio extraction, transcription, and analysis
"""
import os
import tempfile
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("Whisper not available - audio transcription disabled")

try:
    from moviepy import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    try:
        from moviepy.editor import VideoFileClip
        MOVIEPY_AVAILABLE = True
    except ImportError:
        MOVIEPY_AVAILABLE = False
        logging.warning("MoviePy not available - audio extraction limited")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("Librosa not available - advanced audio analysis disabled")

from .base import (
    BaseProcessor, AnalysisContext, ProcessingResult, ProcessingStatus,
    ComponentInterface
)
from .error_handler import (
    handle_component_error, ErrorSeverity, ErrorCategory, get_error_handler
)

logger = logging.getLogger(__name__)


class AudioProcessor(BaseProcessor, ComponentInterface):
    """Handles all audio-related processing including extraction, transcription, and analysis"""
    
    def __init__(self):
        super().__init__("audio_processor")
        self.whisper_model = None
        self.temp_files = []
        
    async def initialize(self) -> bool:
        """Initialize audio processing models"""
        try:
            logger.info("Initializing Audio Processor")
            
            # Initialize Whisper model if available
            if WHISPER_AVAILABLE:
                try:
                    self.whisper_model = whisper.load_model("base")
                    logger.info("Whisper model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load Whisper model: {e}")
                    self.whisper_model = None
            
            # Register error handling strategies
            error_handler = get_error_handler()
            error_handler.register_fallback_strategy(
                'audio_processor', 
                self._audio_fallback
            )
            
            # Set retry policy for audio processing
            error_handler.set_retry_policy('audio_processor', {
                'max_retries': 2,
                'base_delay': 1.5,
                'max_delay': 8.0
            })
            
            self.is_initialized = True
            self.status = ProcessingStatus.COMPLETED
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Audio Processor: {e}")
            self.status = ProcessingStatus.FAILED
            
            # Handle initialization error
            await handle_component_error(
                component_name=self.name,
                error=e,
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.INITIALIZATION,
                allow_retry=False,
                allow_fallback=False
            )
            return False
    
    async def process(self, context: AnalysisContext) -> ProcessingResult:
        """Process audio content for analysis"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing audio content for segment {context.segment.segment_id if context.segment else 'full video'}")
            
            # Extract audio from video
            audio_path = await self.extract_audio(
                context.video_path,
                context.segment.start_time if context.segment else 0,
                context.segment.end_time if context.segment else context.metadata.duration
            )
            
            if not audio_path:
                return ProcessingResult(
                    component_name=self.name,
                    status=ProcessingStatus.FAILED,
                    data={},
                    error_message="Audio extraction failed"
                )
            
            # Analyze audio
            analysis_results = {
                'transcription': '',
                'speakers': [],
                'music_analysis': {},
                'sound_effects': [],
                'speech_patterns': {},
                'audio_features': {},
                'description': '',
                'confidence': 0.0
            }
            
            # Speech transcription
            if self.whisper_model:
                transcription_result = await self.transcribe_speech(audio_path)
                analysis_results['transcription'] = transcription_result.get('text', '')
                analysis_results['speakers'] = transcription_result.get('speakers', [])
            
            # Audio feature analysis
            if LIBROSA_AVAILABLE:
                features = await self.analyze_audio_features(audio_path)
                analysis_results['audio_features'] = features
                
                # Music analysis
                music_analysis = await self.analyze_music(audio_path, features)
                analysis_results['music_analysis'] = music_analysis
                
                # Sound effects detection
                sound_effects = await self.detect_sound_effects(audio_path, features)
                analysis_results['sound_effects'] = sound_effects
            
            # Speech pattern analysis
            if analysis_results['transcription']:
                speech_patterns = await self.analyze_speech_patterns(
                    analysis_results['transcription'],
                    analysis_results.get('speakers', [])
                )
                analysis_results['speech_patterns'] = speech_patterns
            
            # Generate description
            description = await self.generate_description(analysis_results, context)
            analysis_results['description'] = description
            
            # Calculate confidence
            analysis_results['confidence'] = self._calculate_confidence(analysis_results)
            
            # Cleanup temporary audio file
            if audio_path in self.temp_files:
                try:
                    os.remove(audio_path)
                    self.temp_files.remove(audio_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {audio_path}: {e}")
            
            processing_time = time.time() - start_time
            logger.info(f"Audio processing completed in {processing_time:.2f}s")
            
            return ProcessingResult(
                component_name=self.name,
                status=ProcessingStatus.COMPLETED,
                data=analysis_results,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            
            # Handle audio processing error with comprehensive error handling
            error_result = await handle_component_error(
                component_name=self.name,
                error=e,
                context={
                    'segment_id': context.segment.segment_id if context.segment else 'full_video',
                    'video_path': context.video_path,
                    'operation': 'audio_processing'
                },
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.PROCESSING,
                allow_retry=True,
                allow_fallback=True
            )
            
            # Return error result or fallback result
            if error_result.status == ProcessingStatus.COMPLETED and error_result.data.get('fallback_used'):
                return error_result
            else:
                return ProcessingResult(
                    component_name=self.name,
                    status=ProcessingStatus.FAILED,
                    data={},
                    error_message=str(e),
                    processing_time=time.time() - start_time
                )
    
    async def extract_audio(self, video_path: str, start_time: float = 0, end_time: float = None) -> Optional[str]:
        """Extract audio from video file"""
        try:
            if not MOVIEPY_AVAILABLE:
                logger.error("MoviePy not available for audio extraction")
                return None
            
            # Create temporary audio file
            temp_audio = tempfile.mktemp(suffix=".wav")
            self.temp_files.append(temp_audio)
            
            # Extract audio using MoviePy
            with VideoFileClip(video_path) as video:
                if video.audio is None:
                    logger.warning("No audio track found in video")
                    return None
                
                # Extract segment if specified
                audio_clip = video.audio
                if start_time > 0 or end_time is not None:
                    audio_clip = audio_clip.subclipped(start_time, end_time)
                
                # Write audio file
                audio_clip.write_audiofile(
                    temp_audio,
                    codec='pcm_s16le',
                    ffmpeg_params=["-ac", "1", "-ar", "16000"]  # Mono, 16kHz for Whisper
                )
                
                audio_clip.close()
            
            logger.info(f"Audio extracted to {temp_audio}")
            return temp_audio
            
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return None
    
    async def transcribe_speech(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe speech using Whisper"""
        try:
            if not self.whisper_model:
                return {'text': '', 'speakers': []}
            
            # Transcribe with word-level timestamps
            result = self.whisper_model.transcribe(
                audio_path,
                word_timestamps=True,
                task="transcribe"
            )
            
            transcription_data = {
                'text': result.get('text', ''),
                'language': result.get('language', 'unknown'),
                'segments': result.get('segments', []),
                'speakers': []
            }
            
            # Simple speaker diarization (placeholder - would use more sophisticated method)
            speakers = await self.perform_speaker_diarization(audio_path, transcription_data['segments'])
            transcription_data['speakers'] = speakers
            
            return transcription_data
            
        except Exception as e:
            logger.error(f"Speech transcription failed: {e}")
            return {'text': '', 'speakers': []}
    
    async def perform_speaker_diarization(self, audio_path: str, segments: List[Dict]) -> List[Dict[str, Any]]:
        """Simple speaker diarization using audio features"""
        try:
            if not LIBROSA_AVAILABLE:
                return []
            
            # Load audio
            y, sr = librosa.load(audio_path)
            
            speakers = []
            
            for i, segment in enumerate(segments):
                # Extract MFCC features for the segment
                start_frame = int(segment['start'] * sr)
                end_frame = int(segment['end'] * sr)
                segment_audio = y[start_frame:end_frame]
                
                if len(segment_audio) > 0:
                    # Simple clustering based on MFCC features
                    mfccs = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
                    
                    # Simplified speaker assignment (alternating speakers)
                    speaker_id = f"Speaker_{(i % 2) + 1}"
                    
                    speakers.append({
                        'speaker_id': speaker_id,
                        'start_time': segment['start'],
                        'end_time': segment['end'],
                        'text': segment.get('text', ''),
                        'confidence': 0.8  # Placeholder confidence
                    })
            
            return speakers
            
        except Exception as e:
            logger.error(f"Speaker diarization failed: {e}")
            return []
    
    async def analyze_audio_features(self, audio_path: str) -> Dict[str, Any]:
        """Extract comprehensive audio features"""
        try:
            if not LIBROSA_AVAILABLE:
                return {}
            
            # Load audio
            y, sr = librosa.load(audio_path)
            
            # Extract various audio features
            features = {
                'duration': float(len(y) / sr),
                'sample_rate': sr,
                'tempo': float(librosa.beat.tempo(y=y, sr=sr)[0]),
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
                'rms_energy': float(np.mean(librosa.feature.rms(y=y))),
                'spectral_bandwidth': float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
            }
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i+1}'] = float(np.mean(mfccs[i]))
            
            # Chroma features (for music analysis)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = float(np.mean(chroma))
            features['chroma_std'] = float(np.std(chroma))
            
            return features
            
        except Exception as e:
            logger.error(f"Audio feature analysis failed: {e}")
            return {}
    
    async def analyze_music(self, audio_path: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze musical elements in audio"""
        try:
            if not LIBROSA_AVAILABLE:
                return {}
            
            music_analysis = {
                'has_music': False,
                'tempo': features.get('tempo', 0),
                'key': 'unknown',
                'rhythm_strength': 0.0,
                'harmonic_content': 0.0
            }
            
            # Load audio for detailed analysis
            y, sr = librosa.load(audio_path)
            
            # Detect if music is present
            # High spectral centroid and chroma variance often indicate music
            spectral_centroid = features.get('spectral_centroid', 0)
            chroma_std = features.get('chroma_std', 0)
            
            if spectral_centroid > 2000 and chroma_std > 0.1:
                music_analysis['has_music'] = True
            
            # Rhythm strength analysis
            tempo_confidence = 0.0
            try:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                if len(beats) > 10:  # Sufficient beats detected
                    beat_intervals = np.diff(beats) / sr
                    rhythm_consistency = 1.0 - np.std(beat_intervals) / np.mean(beat_intervals)
                    music_analysis['rhythm_strength'] = max(0.0, min(1.0, rhythm_consistency))
            except Exception:
                pass
            
            # Harmonic content analysis
            try:
                harmonic, percussive = librosa.effects.hpss(y)
                harmonic_ratio = np.mean(np.abs(harmonic)) / (np.mean(np.abs(y)) + 1e-8)
                music_analysis['harmonic_content'] = float(harmonic_ratio)
            except Exception:
                pass
            
            return music_analysis
            
        except Exception as e:
            logger.error(f"Music analysis failed: {e}")
            return {}
    
    async def detect_sound_effects(self, audio_path: str, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect and classify sound effects"""
        try:
            if not LIBROSA_AVAILABLE:
                return []
            
            sound_effects = []
            
            # Load audio
            y, sr = librosa.load(audio_path)
            
            # Simple sound effect detection based on energy spikes
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            
            # Find energy spikes
            threshold = np.mean(rms) + 2 * np.std(rms)
            spikes = np.where(rms > threshold)[0]
            
            if len(spikes) > 0:
                # Group consecutive spikes
                spike_groups = []
                current_group = [spikes[0]]
                
                for spike in spikes[1:]:
                    if spike - current_group[-1] <= 5:  # Within 5 frames
                        current_group.append(spike)
                    else:
                        spike_groups.append(current_group)
                        current_group = [spike]
                
                spike_groups.append(current_group)
                
                # Analyze each spike group
                for group in spike_groups:
                    start_frame = group[0]
                    end_frame = group[-1]
                    
                    start_time = start_frame * 512 / sr
                    end_time = end_frame * 512 / sr
                    
                    # Simple classification based on spectral characteristics
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    segment = y[start_sample:end_sample]
                    
                    if len(segment) > 0:
                        # Analyze spectral characteristics
                        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
                        zcr = np.mean(librosa.feature.zero_crossing_rate(segment))
                        
                        # Simple classification
                        effect_type = "unknown"
                        if spectral_centroid > 4000 and zcr > 0.1:
                            effect_type = "high_frequency_noise"
                        elif spectral_centroid < 1000:
                            effect_type = "low_frequency_rumble"
                        else:
                            effect_type = "mid_frequency_sound"
                        
                        sound_effects.append({
                            'type': effect_type,
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': end_time - start_time,
                            'intensity': float(np.mean(rms[start_frame:end_frame+1])),
                            'confidence': 0.6  # Placeholder confidence
                        })
            
            return sound_effects
            
        except Exception as e:
            logger.error(f"Sound effect detection failed: {e}")
            return []
    
    async def analyze_speech_patterns(self, transcription: str, speakers: List[Dict]) -> Dict[str, Any]:
        """Analyze speech patterns and characteristics"""
        try:
            if not transcription:
                return {}
            
            patterns = {
                'word_count': len(transcription.split()),
                'sentence_count': len([s for s in transcription.split('.') if s.strip()]),
                'average_words_per_sentence': 0.0,
                'speaking_rate': 0.0,
                'emotional_indicators': [],
                'speaker_count': len(set(speaker['speaker_id'] for speaker in speakers))
            }
            
            # Calculate average words per sentence
            sentences = [s.strip() for s in transcription.split('.') if s.strip()]
            if sentences:
                word_counts = [len(sentence.split()) for sentence in sentences]
                patterns['average_words_per_sentence'] = np.mean(word_counts)
            
            # Calculate speaking rate (words per minute)
            if speakers:
                total_duration = max(speaker['end_time'] for speaker in speakers) - min(speaker['start_time'] for speaker in speakers)
                if total_duration > 0:
                    patterns['speaking_rate'] = (patterns['word_count'] / total_duration) * 60
            
            # Simple emotional indicators
            emotional_words = {
                'positive': ['good', 'great', 'excellent', 'amazing', 'wonderful', 'happy', 'love'],
                'negative': ['bad', 'terrible', 'awful', 'hate', 'angry', 'sad', 'disappointed'],
                'excitement': ['wow', 'incredible', 'fantastic', 'awesome', 'exciting'],
                'uncertainty': ['maybe', 'perhaps', 'possibly', 'might', 'could', 'uncertain']
            }
            
            text_lower = transcription.lower()
            for emotion, words in emotional_words.items():
                count = sum(text_lower.count(word) for word in words)
                if count > 0:
                    patterns['emotional_indicators'].append({
                        'emotion': emotion,
                        'count': count,
                        'intensity': min(count / 10.0, 1.0)  # Normalize to 0-1
                    })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Speech pattern analysis failed: {e}")
            return {}
    
    async def generate_description(self, analysis_results: Dict[str, Any], context: AnalysisContext) -> str:
        """Generate textual description of audio content"""
        try:
            description_parts = []
            
            # Describe transcription
            transcription = analysis_results.get('transcription', '')
            if transcription:
                word_count = len(transcription.split())
                description_parts.append(f"Speech transcribed ({word_count} words)")
                
                # Add language if detected
                # Note: This would come from Whisper results in full implementation
            
            # Describe speakers
            speakers = analysis_results.get('speakers', [])
            if speakers:
                speaker_count = len(set(speaker['speaker_id'] for speaker in speakers))
                description_parts.append(f"{speaker_count} speaker{'s' if speaker_count > 1 else ''} detected")
            
            # Describe music
            music_analysis = analysis_results.get('music_analysis', {})
            if music_analysis.get('has_music', False):
                tempo = music_analysis.get('tempo', 0)
                if tempo > 0:
                    description_parts.append(f"Music detected (tempo: {tempo:.0f} BPM)")
                else:
                    description_parts.append("Music detected")
            
            # Describe sound effects
            sound_effects = analysis_results.get('sound_effects', [])
            if sound_effects:
                effect_count = len(sound_effects)
                description_parts.append(f"{effect_count} sound effect{'s' if effect_count > 1 else ''} detected")
            
            # Describe speech patterns
            speech_patterns = analysis_results.get('speech_patterns', {})
            speaking_rate = speech_patterns.get('speaking_rate', 0)
            if speaking_rate > 0:
                if speaking_rate > 180:
                    description_parts.append("Fast-paced speech")
                elif speaking_rate < 120:
                    description_parts.append("Slow-paced speech")
                else:
                    description_parts.append("Normal-paced speech")
            
            # Describe emotional content
            emotional_indicators = speech_patterns.get('emotional_indicators', [])
            if emotional_indicators:
                dominant_emotion = max(emotional_indicators, key=lambda x: x['intensity'])
                description_parts.append(f"Emotional tone: {dominant_emotion['emotion']}")
            
            return ". ".join(description_parts) if description_parts else "Audio analysis completed"
            
        except Exception as e:
            logger.error(f"Description generation failed: {e}")
            return "Audio content processed"
    
    def _calculate_confidence(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate confidence score for audio analysis"""
        try:
            confidence_factors = []
            
            # Transcription confidence (based on word count and clarity)
            transcription = analysis_results.get('transcription', '')
            if transcription:
                word_count = len(transcription.split())
                # More words generally indicate better transcription
                transcription_confidence = min(word_count / 100.0, 1.0)
                confidence_factors.append(transcription_confidence)
            
            # Speaker detection confidence
            speakers = analysis_results.get('speakers', [])
            if speakers:
                # Average speaker confidence
                speaker_confidences = [speaker.get('confidence', 0.5) for speaker in speakers]
                confidence_factors.append(np.mean(speaker_confidences))
            
            # Audio feature analysis confidence
            audio_features = analysis_results.get('audio_features', {})
            if audio_features:
                # Presence of audio features indicates successful analysis
                confidence_factors.append(0.8)
            
            # Music analysis confidence
            music_analysis = analysis_results.get('music_analysis', {})
            if music_analysis.get('has_music', False):
                rhythm_strength = music_analysis.get('rhythm_strength', 0)
                confidence_factors.append(rhythm_strength)
            
            return np.mean(confidence_factors) if confidence_factors else 0.5
            
        except Exception:
            return 0.5
    
    def get_supported_operations(self) -> List[str]:
        """Return list of supported operations"""
        operations = ['audio_extraction']
        
        if WHISPER_AVAILABLE:
            operations.extend(['speech_transcription', 'speaker_diarization'])
        
        if LIBROSA_AVAILABLE:
            operations.extend(['audio_feature_analysis', 'music_analysis', 'sound_effect_detection'])
        
        return operations
    
    async def health_check(self) -> bool:
        """Check if component is healthy and ready"""
        return self.is_initialized and self.status != ProcessingStatus.FAILED
    
    async def cleanup(self) -> None:
        """Clean up audio processing resources"""
        logger.info("Cleaning up Audio Processor")
        
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
        
        self.temp_files.clear()
        
        # Clean up Whisper model
        if self.whisper_model:
            del self.whisper_model
            self.whisper_model = None
    
    async def _audio_fallback(self, context: Dict[str, Any]) -> ProcessingResult:
        """Fallback strategy for audio processing failures"""
        try:
            logger.info("Executing audio processing fallback")
            
            # Create minimal audio analysis result
            fallback_data = {
                'transcription': '',
                'speakers': [],
                'music_analysis': {'has_music': False, 'tempo': 0},
                'sound_effects': [],
                'speech_patterns': {'word_count': 0, 'speaker_count': 0},
                'audio_features': {},
                'description': 'Audio processing unavailable - visual analysis continued',
                'confidence': 0.1,
                'fallback_used': True
            }
            
            return ProcessingResult(
                component_name=self.name,
                status=ProcessingStatus.COMPLETED,
                data=fallback_data
            )
            
        except Exception as e:
            logger.error(f"Audio fallback failed: {e}")
            return ProcessingResult(
                component_name=self.name,
                status=ProcessingStatus.FAILED,
                data={},
                error_message=f"Audio fallback failed: {str(e)}"
            )