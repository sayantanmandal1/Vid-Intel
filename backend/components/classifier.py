"""
Content Classifier - Determines video content type and analysis strategy
"""
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available - visual classification limited")

from .base import (
    BaseProcessor, AnalysisContext, ProcessingResult, ProcessingStatus,
    ComponentInterface, ContentType
)
from .error_handler import (
    handle_component_error, ErrorSeverity, ErrorCategory, get_error_handler
)

logger = logging.getLogger(__name__)


class ContentClassifier(BaseProcessor, ComponentInterface):
    """Analyzes video content to determine type and optimal analysis strategy"""
    
    def __init__(self):
        super().__init__("content_classifier")
        self.sample_duration = 30.0  # Sample first 30 seconds for classification
        self.frame_sample_interval = 5.0  # Sample frame every 5 seconds
        
    async def initialize(self) -> bool:
        """Initialize content classifier"""
        try:
            logger.info("Initializing Content Classifier")
            
            if not CV2_AVAILABLE:
                logger.warning("OpenCV not available - using basic classification")
            
            # Register error handling strategies
            error_handler = get_error_handler()
            error_handler.register_fallback_strategy(
                'content_classifier', 
                self._classifier_fallback
            )
            
            # Set retry policy for content classification
            error_handler.set_retry_policy('content_classifier', {
                'max_retries': 1,
                'base_delay': 0.5,
                'max_delay': 2.0
            })
            
            self.is_initialized = True
            self.status = ProcessingStatus.COMPLETED
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Content Classifier: {e}")
            self.status = ProcessingStatus.FAILED
            
            # Handle initialization error
            await handle_component_error(
                component_name=self.name,
                error=e,
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.INITIALIZATION,
                allow_retry=False,
                allow_fallback=False
            )
            return False
    
    async def process(self, context: AnalysisContext) -> ProcessingResult:
        """Classify video content type"""
        start_time = time.time()
        
        try:
            logger.info(f"Classifying content for video: {Path(context.video_path).name}")
            
            # Extract sample frames and basic video info
            video_sample = await self._extract_video_sample(context.video_path)
            
            if not video_sample:
                return ProcessingResult(
                    component_name=self.name,
                    status=ProcessingStatus.FAILED,
                    data={},
                    error_message="Failed to extract video sample"
                )
            
            # Classify content type
            content_type = await self.classify_content(video_sample)
            
            # Get analysis strategy for the content type
            analysis_strategy = self.get_analysis_strategy(content_type)
            
            result_data = {
                'content_type': content_type.value,
                'analysis_strategy': analysis_strategy,
                'classification_confidence': video_sample.get('confidence', 0.5),
                'video_characteristics': video_sample.get('characteristics', {}),
                'recommended_components': analysis_strategy.get('components', [])
            }
            
            processing_time = time.time() - start_time
            logger.info(f"Content classified as {content_type.value} in {processing_time:.2f}s")
            
            return ProcessingResult(
                component_name=self.name,
                status=ProcessingStatus.COMPLETED,
                data=result_data,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Content classification failed: {e}")
            
            # Handle classification error with comprehensive error handling
            error_result = await handle_component_error(
                component_name=self.name,
                error=e,
                context={
                    'video_path': context.video_path,
                    'operation': 'content_classification'
                },
                severity=ErrorSeverity.LOW,  # Classification failure is not critical
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
                    data={'content_type': ContentType.GENERAL.value},
                    error_message=str(e),
                    processing_time=time.time() - start_time
                )
    
    async def classify_content(self, video_sample: Dict[str, Any]) -> ContentType:
        """Classify video content based on sample analysis"""
        try:
            characteristics = video_sample.get('characteristics', {})
            
            # Extract classification features
            visual_features = characteristics.get('visual', {})
            audio_features = characteristics.get('audio', {})
            metadata_features = characteristics.get('metadata', {})
            
            # Classification scores for each content type
            scores = {
                ContentType.MUSIC_VIDEO: 0.0,
                ContentType.GAMING: 0.0,
                ContentType.EDUCATIONAL: 0.0,
                ContentType.NARRATIVE: 0.0,
                ContentType.DOCUMENTARY: 0.0,
                ContentType.GENERAL: 0.1  # Base score for general content
            }
            
            # Music video classification
            scores[ContentType.MUSIC_VIDEO] += self._score_music_video(visual_features, audio_features)
            
            # Gaming content classification
            scores[ContentType.GAMING] += self._score_gaming_content(visual_features, audio_features)
            
            # Educational content classification
            scores[ContentType.EDUCATIONAL] += self._score_educational_content(visual_features, audio_features)
            
            # Narrative content classification
            scores[ContentType.NARRATIVE] += self._score_narrative_content(visual_features, audio_features)
            
            # Documentary classification
            scores[ContentType.DOCUMENTARY] += self._score_documentary_content(visual_features, audio_features)
            
            # Select content type with highest score
            best_type = max(scores.keys(), key=lambda k: scores[k])
            best_score = scores[best_type]
            
            # If no clear winner, default to GENERAL
            if best_score < 0.3:
                return ContentType.GENERAL
            
            logger.info(f"Content classified as {best_type.value} with score {best_score:.2f}")
            return best_type
            
        except Exception as e:
            logger.error(f"Content classification failed: {e}")
            return ContentType.GENERAL
    
    def get_analysis_strategy(self, content_type: ContentType) -> Dict[str, Any]:
        """Get analysis strategy for content type"""
        strategies = {
            ContentType.MUSIC_VIDEO: {
                'focus_areas': ['visual_aesthetics', 'performer_movements', 'lyrics', 'rhythm'],
                'components': ['visual_extractor', 'audio_processor', 'llm_integration'],
                'visual_priority': 'high',
                'audio_priority': 'high',
                'transcription_priority': 'medium',
                'object_detection': True,
                'scene_analysis': True,
                'movement_analysis': True,
                'music_analysis': True,
                'description_style': 'artistic'
            },
            ContentType.GAMING: {
                'focus_areas': ['gameplay_mechanics', 'player_actions', 'game_environment', 'ui_elements'],
                'components': ['visual_extractor', 'audio_processor'],
                'visual_priority': 'high',
                'audio_priority': 'low',
                'transcription_priority': 'low',
                'object_detection': True,
                'scene_analysis': True,
                'movement_analysis': True,
                'music_analysis': False,
                'description_style': 'technical'
            },
            ContentType.EDUCATIONAL: {
                'focus_areas': ['spoken_content', 'visual_aids', 'demonstrations', 'key_points'],
                'components': ['audio_processor', 'visual_extractor', 'llm_integration'],
                'visual_priority': 'medium',
                'audio_priority': 'high',
                'transcription_priority': 'high',
                'object_detection': False,
                'scene_analysis': True,
                'movement_analysis': False,
                'music_analysis': False,
                'description_style': 'informative'
            },
            ContentType.NARRATIVE: {
                'focus_areas': ['character_interactions', 'dialogue', 'plot_progression', 'scenes'],
                'components': ['audio_processor', 'visual_extractor', 'llm_integration'],
                'visual_priority': 'high',
                'audio_priority': 'high',
                'transcription_priority': 'high',
                'object_detection': True,
                'scene_analysis': True,
                'movement_analysis': True,
                'music_analysis': False,
                'description_style': 'narrative'
            },
            ContentType.DOCUMENTARY: {
                'focus_areas': ['factual_content', 'narration', 'visual_evidence', 'interviews'],
                'components': ['audio_processor', 'visual_extractor', 'llm_integration'],
                'visual_priority': 'medium',
                'audio_priority': 'high',
                'transcription_priority': 'high',
                'object_detection': True,
                'scene_analysis': True,
                'movement_analysis': False,
                'music_analysis': False,
                'description_style': 'factual'
            },
            ContentType.GENERAL: {
                'focus_areas': ['general_content', 'visual_audio_balance'],
                'components': ['visual_extractor', 'audio_processor', 'llm_integration'],
                'visual_priority': 'medium',
                'audio_priority': 'medium',
                'transcription_priority': 'medium',
                'object_detection': True,
                'scene_analysis': True,
                'movement_analysis': True,
                'music_analysis': True,
                'description_style': 'balanced'
            }
        }
        
        return strategies.get(content_type, strategies[ContentType.GENERAL])
    
    async def _extract_video_sample(self, video_path: str) -> Optional[Dict[str, Any]]:
        """Extract sample frames and metadata for classification"""
        try:
            if not CV2_AVAILABLE:
                return {
                    'characteristics': {
                        'visual': {},
                        'audio': {},
                        'metadata': {}
                    },
                    'confidence': 0.3
                }
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return None
            
            # Get video metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Sample frames for analysis
            sample_frames = []
            sample_duration = min(self.sample_duration, duration)
            current_time = 0.0
            
            while current_time < sample_duration and len(sample_frames) < 10:
                frame_number = int(current_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                ret, frame = cap.read()
                if ret:
                    sample_frames.append(frame)
                else:
                    break
                
                current_time += self.frame_sample_interval
            
            cap.release()
            
            # Analyze sample frames
            visual_characteristics = self._analyze_visual_characteristics(sample_frames)
            
            # Basic audio characteristics (would need audio extraction for full analysis)
            audio_characteristics = {
                'duration': duration,
                'estimated_speech_ratio': 0.5  # Placeholder
            }
            
            metadata_characteristics = {
                'resolution': f"{width}x{height}",
                'fps': fps,
                'duration': duration,
                'aspect_ratio': width / height if height > 0 else 1.0
            }
            
            return {
                'characteristics': {
                    'visual': visual_characteristics,
                    'audio': audio_characteristics,
                    'metadata': metadata_characteristics
                },
                'confidence': 0.7,
                'sample_frames_count': len(sample_frames)
            }
            
        except Exception as e:
            logger.error(f"Video sample extraction failed: {e}")
            return None
    
    def _analyze_visual_characteristics(self, frames: List) -> Dict[str, Any]:
        """Analyze visual characteristics of sample frames"""
        try:
            if not frames:
                return {}
            
            characteristics = {
                'frame_count': len(frames),
                'average_brightness': 0.0,
                'color_variance': 0.0,
                'edge_density': 0.0,
                'motion_estimate': 0.0,
                'scene_complexity': 0.0
            }
            
            brightness_values = []
            color_variances = []
            edge_densities = []
            
            prev_frame = None
            motion_values = []
            
            for frame in frames:
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Brightness analysis
                brightness = np.mean(gray)
                brightness_values.append(brightness)
                
                # Color variance analysis
                color_std = np.std(frame)
                color_variances.append(color_std)
                
                # Edge density analysis
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                edge_densities.append(edge_density)
                
                # Motion estimation
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray)
                    motion = np.mean(diff)
                    motion_values.append(motion)
                
                prev_frame = gray
            
            # Calculate averages
            characteristics['average_brightness'] = float(np.mean(brightness_values))
            characteristics['color_variance'] = float(np.mean(color_variances))
            characteristics['edge_density'] = float(np.mean(edge_densities))
            
            if motion_values:
                characteristics['motion_estimate'] = float(np.mean(motion_values))
            
            # Scene complexity (combination of color variance and edge density)
            characteristics['scene_complexity'] = (
                characteristics['color_variance'] / 255.0 + 
                characteristics['edge_density']
            ) / 2.0
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Visual characteristics analysis failed: {e}")
            return {}
    
    def _score_music_video(self, visual_features: Dict, audio_features: Dict) -> float:
        """Score likelihood of music video content"""
        score = 0.0
        
        # High color variance and scene complexity suggest artistic content
        color_variance = visual_features.get('color_variance', 0) / 255.0
        scene_complexity = visual_features.get('scene_complexity', 0)
        motion_estimate = visual_features.get('motion_estimate', 0) / 255.0
        
        # Music videos typically have high visual variety and movement
        if color_variance > 0.3:
            score += 0.3
        if scene_complexity > 0.4:
            score += 0.2
        if motion_estimate > 0.2:
            score += 0.3
        
        # Duration factor (music videos are typically 3-6 minutes)
        duration = audio_features.get('duration', 0)
        if 180 <= duration <= 360:  # 3-6 minutes
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_gaming_content(self, visual_features: Dict, audio_features: Dict) -> float:
        """Score likelihood of gaming content"""
        score = 0.0
        
        # Gaming content often has high edge density (UI elements, sharp graphics)
        edge_density = visual_features.get('edge_density', 0)
        motion_estimate = visual_features.get('motion_estimate', 0) / 255.0
        
        # High edge density suggests UI elements and sharp graphics
        if edge_density > 0.1:
            score += 0.4
        
        # Moderate to high motion for gameplay
        if 0.1 <= motion_estimate <= 0.5:
            score += 0.3
        
        # Gaming videos can be any length, but often longer
        duration = audio_features.get('duration', 0)
        if duration > 300:  # Longer than 5 minutes
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_educational_content(self, visual_features: Dict, audio_features: Dict) -> float:
        """Score likelihood of educational content"""
        score = 0.0
        
        # Educational content often has lower motion (static presentations)
        motion_estimate = visual_features.get('motion_estimate', 0) / 255.0
        brightness = visual_features.get('average_brightness', 128) / 255.0
        
        # Lower motion suggests presentation-style content
        if motion_estimate < 0.15:
            score += 0.3
        
        # Well-lit content (presentations, classrooms)
        if brightness > 0.4:
            score += 0.2
        
        # Educational videos are often longer
        duration = audio_features.get('duration', 0)
        if duration > 600:  # Longer than 10 minutes
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_narrative_content(self, visual_features: Dict, audio_features: Dict) -> float:
        """Score likelihood of narrative content"""
        score = 0.0
        
        # Narrative content has balanced visual characteristics
        motion_estimate = visual_features.get('motion_estimate', 0) / 255.0
        scene_complexity = visual_features.get('scene_complexity', 0)
        
        # Moderate motion and complexity for narrative scenes
        if 0.1 <= motion_estimate <= 0.3:
            score += 0.2
        if 0.2 <= scene_complexity <= 0.6:
            score += 0.2
        
        # Narrative content can vary in length
        duration = audio_features.get('duration', 0)
        if duration > 120:  # Longer than 2 minutes
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_documentary_content(self, visual_features: Dict, audio_features: Dict) -> float:
        """Score likelihood of documentary content"""
        score = 0.0
        
        # Documentary content often has varied scenes but lower motion
        motion_estimate = visual_features.get('motion_estimate', 0) / 255.0
        scene_complexity = visual_features.get('scene_complexity', 0)
        
        # Lower motion with varied scenes
        if motion_estimate < 0.2:
            score += 0.2
        if scene_complexity > 0.3:
            score += 0.2
        
        # Documentaries are typically longer
        duration = audio_features.get('duration', 0)
        if duration > 900:  # Longer than 15 minutes
            score += 0.3
        elif duration > 300:  # Longer than 5 minutes
            score += 0.1
        
        return min(score, 1.0)
    
    def get_supported_operations(self) -> List[str]:
        """Return list of supported operations"""
        operations = ['content_classification', 'analysis_strategy_selection']
        
        if CV2_AVAILABLE:
            operations.extend(['visual_sampling', 'frame_analysis'])
        
        return operations
    
    async def health_check(self) -> bool:
        """Check if component is healthy and ready"""
        return self.is_initialized and self.status != ProcessingStatus.FAILED
    
    async def cleanup(self) -> None:
        """Clean up classifier resources"""
        logger.info("Cleaning up Content Classifier")
        # No specific cleanup needed for this component
    
    async def _classifier_fallback(self, context: Dict[str, Any]) -> ProcessingResult:
        """Fallback strategy for content classification failures"""
        try:
            logger.info("Executing content classification fallback")
            
            # Provide default classification result
            fallback_data = {
                'content_type': ContentType.GENERAL.value,
                'analysis_strategy': {
                    'focus_areas': ['general_content', 'visual_audio_balance'],
                    'components': ['visual_extractor', 'audio_processor'],
                    'visual_priority': 'medium',
                    'audio_priority': 'medium',
                    'transcription_priority': 'medium',
                    'object_detection': True,
                    'scene_analysis': True,
                    'movement_analysis': True,
                    'music_analysis': True,
                    'description_style': 'balanced'
                },
                'classification_confidence': 0.3,
                'video_characteristics': {},
                'recommended_components': ['visual_extractor', 'audio_processor'],
                'fallback_used': True
            }
            
            return ProcessingResult(
                component_name=self.name,
                status=ProcessingStatus.COMPLETED,
                data=fallback_data
            )
            
        except Exception as e:
            logger.error(f"Content classifier fallback failed: {e}")
            return ProcessingResult(
                component_name=self.name,
                status=ProcessingStatus.FAILED,
                data={},
                error_message=f"Classifier fallback failed: {str(e)}"
            )