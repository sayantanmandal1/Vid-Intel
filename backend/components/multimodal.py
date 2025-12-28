"""
Multi-Modal Data Integration - Handles temporal alignment and data correlation
"""
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from .base import (
    BaseProcessor, AnalysisContext, ProcessingResult, ProcessingStatus,
    ComponentInterface, SegmentAnalysis, VideoSegment
)

logger = logging.getLogger(__name__)


@dataclass
class TemporalEvent:
    """Represents a temporal event from visual or audio analysis"""
    timestamp: float
    duration: float
    event_type: str  # 'visual', 'audio', 'speech', 'object', 'scene_change'
    content: str
    confidence: float
    source_component: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CorrelationResult:
    """Result of correlating visual and audio events"""
    visual_event: Optional[TemporalEvent]
    audio_event: Optional[TemporalEvent]
    correlation_strength: float
    temporal_alignment: float  # How well aligned in time (0-1)
    semantic_alignment: float  # How well aligned in meaning (0-1)
    conflict_detected: bool
    conflict_type: Optional[str] = None
    resolution_strategy: Optional[str] = None


class MultiModalIntegrator(BaseProcessor, ComponentInterface):
    """Handles temporal alignment and correlation of multi-modal data"""
    
    def __init__(self):
        super().__init__("multimodal_integrator")
        self.temporal_window = 2.0  # seconds - events within this window are considered aligned
        self.confidence_threshold = 0.5
        
    async def initialize(self) -> bool:
        """Initialize the multi-modal integrator"""
        try:
            logger.info("Initializing Multi-Modal Integrator")
            self.is_initialized = True
            self.status = ProcessingStatus.COMPLETED
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Multi-Modal Integrator: {e}")
            self.status = ProcessingStatus.FAILED
            return False
    
    async def process(self, context: AnalysisContext) -> ProcessingResult:
        """Process multi-modal data integration for a segment"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing multi-modal integration for segment {context.segment.segment_id if context.segment else 'full video'}")
            
            # Extract temporal events from visual and audio data
            visual_events = self._extract_visual_events(context.visual_data, context.segment)
            audio_events = self._extract_audio_events(context.audio_data, context.segment)
            
            # Perform temporal alignment
            aligned_events = await self.align_temporal_events(visual_events, audio_events)
            
            # Detect and resolve conflicts
            correlation_results = await self.correlate_events(aligned_events)
            
            # Generate coherent narrative
            narrative = await self.generate_coherent_narrative(
                correlation_results, context.content_type, context.segment
            )
            
            # Calculate integration metrics
            integration_metrics = self._calculate_integration_metrics(
                visual_events, audio_events, correlation_results
            )
            
            result_data = {
                'visual_events': [self._event_to_dict(e) for e in visual_events],
                'audio_events': [self._event_to_dict(e) for e in audio_events],
                'correlations': [self._correlation_to_dict(c) for c in correlation_results],
                'coherent_narrative': narrative,
                'integration_metrics': integration_metrics,
                'temporal_alignment_quality': integration_metrics.get('temporal_alignment_score', 0.0),
                'conflict_resolution_count': len([c for c in correlation_results if c.conflict_detected])
            }
            
            processing_time = time.time() - start_time
            logger.info(f"Multi-modal integration completed in {processing_time:.2f}s")
            
            return ProcessingResult(
                component_name=self.name,
                status=ProcessingStatus.COMPLETED,
                data=result_data,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Multi-modal integration failed: {e}")
            return ProcessingResult(
                component_name=self.name,
                status=ProcessingStatus.FAILED,
                data={},
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _extract_visual_events(self, visual_data: Optional[Dict[str, Any]], 
                              segment: Optional[VideoSegment]) -> List[TemporalEvent]:
        """Extract temporal events from visual analysis data"""
        events = []
        
        if not visual_data:
            return events
        
        segment_start = segment.start_time if segment else 0.0
        
        # Extract object detection events
        objects = visual_data.get('objects', [])
        for obj in objects:
            timestamp = segment_start + obj.get('timestamp', 0)
            events.append(TemporalEvent(
                timestamp=timestamp,
                duration=0.5,  # Assume brief object detection
                event_type='object',
                content=f"{obj.get('class_name', 'unknown')} detected",
                confidence=obj.get('confidence', 0.5),
                source_component='visual_extractor',
                metadata={'bounding_box': obj.get('bounding_box'), 'class_name': obj.get('class_name')}
            ))
        
        # Extract scene change events
        scenes = visual_data.get('scenes', [])
        for scene in scenes:
            if scene.get('is_scene_change', False):
                timestamp = segment_start + scene.get('timestamp', 0)
                events.append(TemporalEvent(
                    timestamp=timestamp,
                    duration=1.0,  # Scene changes have brief duration
                    event_type='scene_change',
                    content=f"Scene change (intensity: {scene.get('change_intensity', 0):.2f})",
                    confidence=min(scene.get('change_intensity', 0) * 2, 1.0),
                    source_component='visual_extractor',
                    metadata={'brightness': scene.get('brightness'), 'contrast': scene.get('contrast')}
                ))
        
        # Extract text events
        text_elements = visual_data.get('text_elements', [])
        for text_elem in text_elements:
            timestamp = segment_start + text_elem.get('timestamp', 0)
            events.append(TemporalEvent(
                timestamp=timestamp,
                duration=2.0,  # Text typically visible for longer
                event_type='visual_text',
                content=f"Text: {text_elem.get('text', '')}",
                confidence=text_elem.get('confidence', 0.5),
                source_component='visual_extractor',
                metadata={'text': text_elem.get('text'), 'bounding_box': text_elem.get('bounding_box')}
            ))
        
        # Extract movement events
        movement = visual_data.get('movement_analysis', {})
        if movement.get('movement_detected', False):
            # Create movement event for the entire segment
            events.append(TemporalEvent(
                timestamp=segment_start,
                duration=segment.duration if segment else 5.0,
                event_type='movement',
                content=f"Movement detected: {movement.get('camera_movement', 'unknown')} camera, {movement.get('object_movement', 'none')} objects",
                confidence=movement.get('movement_intensity', 0.5),
                source_component='visual_extractor',
                metadata=movement
            ))
        
        logger.debug(f"Extracted {len(events)} visual events")
        return events
    
    def _extract_audio_events(self, audio_data: Optional[Dict[str, Any]], 
                             segment: Optional[VideoSegment]) -> List[TemporalEvent]:
        """Extract temporal events from audio analysis data"""
        events = []
        
        if not audio_data:
            return events
        
        segment_start = segment.start_time if segment else 0.0
        
        # Extract speech events from speakers
        speakers = audio_data.get('speakers', [])
        for speaker in speakers:
            timestamp = segment_start + speaker.get('start_time', 0)
            duration = speaker.get('end_time', 0) - speaker.get('start_time', 0)
            events.append(TemporalEvent(
                timestamp=timestamp,
                duration=max(duration, 0.1),
                event_type='speech',
                content=f"{speaker.get('speaker_id', 'Unknown')}: {speaker.get('text', '')}",
                confidence=speaker.get('confidence', 0.5),
                source_component='audio_processor',
                metadata={'speaker_id': speaker.get('speaker_id'), 'text': speaker.get('text')}
            ))
        
        # Extract transcription as a single event if no speakers
        if not speakers:
            transcription = audio_data.get('transcription', '')
            if transcription.strip():
                events.append(TemporalEvent(
                    timestamp=segment_start,
                    duration=segment.duration if segment else 5.0,
                    event_type='speech',
                    content=f"Speech: {transcription}",
                    confidence=0.7,  # Default confidence for transcription
                    source_component='audio_processor',
                    metadata={'transcription': transcription}
                ))
        
        # Extract music events
        music_analysis = audio_data.get('music_analysis', {})
        if music_analysis.get('has_music', False):
            events.append(TemporalEvent(
                timestamp=segment_start,
                duration=segment.duration if segment else 5.0,
                event_type='music',
                content=f"Music detected (tempo: {music_analysis.get('tempo', 0):.0f} BPM)",
                confidence=music_analysis.get('rhythm_strength', 0.5),
                source_component='audio_processor',
                metadata=music_analysis
            ))
        
        # Extract sound effect events
        sound_effects = audio_data.get('sound_effects', [])
        for effect in sound_effects:
            timestamp = segment_start + effect.get('start_time', 0)
            duration = effect.get('duration', 1.0)
            events.append(TemporalEvent(
                timestamp=timestamp,
                duration=duration,
                event_type='sound_effect',
                content=f"Sound effect: {effect.get('type', 'unknown')}",
                confidence=effect.get('confidence', 0.5),
                source_component='audio_processor',
                metadata=effect
            ))
        
        logger.debug(f"Extracted {len(events)} audio events")
        return events
    
    async def align_temporal_events(self, visual_events: List[TemporalEvent], 
                                   audio_events: List[TemporalEvent]) -> List[Tuple[List[TemporalEvent], List[TemporalEvent]]]:
        """Align visual and audio events temporally"""
        try:
            aligned_groups = []
            
            # Create time windows and group events
            all_events = visual_events + audio_events
            if not all_events:
                return aligned_groups
            
            # Sort events by timestamp
            all_events.sort(key=lambda e: e.timestamp)
            
            # Group events within temporal windows
            current_window_start = all_events[0].timestamp
            current_visual = []
            current_audio = []
            
            for event in all_events:
                # Check if event is within current window
                if event.timestamp <= current_window_start + self.temporal_window:
                    if event.source_component == 'visual_extractor':
                        current_visual.append(event)
                    else:
                        current_audio.append(event)
                else:
                    # Save current window if it has events
                    if current_visual or current_audio:
                        aligned_groups.append((current_visual.copy(), current_audio.copy()))
                    
                    # Start new window
                    current_window_start = event.timestamp
                    current_visual = []
                    current_audio = []
                    
                    if event.source_component == 'visual_extractor':
                        current_visual.append(event)
                    else:
                        current_audio.append(event)
            
            # Add final window
            if current_visual or current_audio:
                aligned_groups.append((current_visual, current_audio))
            
            logger.debug(f"Created {len(aligned_groups)} temporal alignment groups")
            return aligned_groups
            
        except Exception as e:
            logger.error(f"Temporal alignment failed: {e}")
            return []
    
    async def correlate_events(self, aligned_groups: List[Tuple[List[TemporalEvent], List[TemporalEvent]]]) -> List[CorrelationResult]:
        """Correlate aligned visual and audio events and detect conflicts"""
        correlations = []
        
        try:
            for visual_events, audio_events in aligned_groups:
                # Handle different correlation scenarios
                if not visual_events and not audio_events:
                    continue
                elif not visual_events:
                    # Audio-only events
                    for audio_event in audio_events:
                        correlations.append(CorrelationResult(
                            visual_event=None,
                            audio_event=audio_event,
                            correlation_strength=0.5,  # Moderate since no visual correlation
                            temporal_alignment=1.0,
                            semantic_alignment=0.0,
                            conflict_detected=False
                        ))
                elif not audio_events:
                    # Visual-only events
                    for visual_event in visual_events:
                        correlations.append(CorrelationResult(
                            visual_event=visual_event,
                            audio_event=None,
                            correlation_strength=0.5,  # Moderate since no audio correlation
                            temporal_alignment=1.0,
                            semantic_alignment=0.0,
                            conflict_detected=False
                        ))
                else:
                    # Both visual and audio events - find best correlations
                    correlations.extend(await self._correlate_multimodal_events(visual_events, audio_events))
            
            logger.debug(f"Generated {len(correlations)} event correlations")
            return correlations
            
        except Exception as e:
            logger.error(f"Event correlation failed: {e}")
            return []
    
    async def _correlate_multimodal_events(self, visual_events: List[TemporalEvent], 
                                          audio_events: List[TemporalEvent]) -> List[CorrelationResult]:
        """Correlate visual and audio events within the same temporal window"""
        correlations = []
        
        # Simple correlation strategy: pair events based on semantic similarity and temporal proximity
        used_audio_events = set()
        
        for visual_event in visual_events:
            best_audio_match = None
            best_correlation_score = 0.0
            best_audio_idx = -1
            
            for i, audio_event in enumerate(audio_events):
                if i in used_audio_events:
                    continue
                
                # Calculate correlation score
                temporal_score = self._calculate_temporal_alignment(visual_event, audio_event)
                semantic_score = self._calculate_semantic_alignment(visual_event, audio_event)
                
                correlation_score = (temporal_score + semantic_score) / 2.0
                
                if correlation_score > best_correlation_score:
                    best_correlation_score = correlation_score
                    best_audio_match = audio_event
                    best_audio_idx = i
            
            # Create correlation result
            if best_audio_match and best_correlation_score > 0.3:  # Minimum correlation threshold
                used_audio_events.add(best_audio_idx)
                
                # Detect conflicts
                conflict_detected, conflict_type, resolution_strategy = self._detect_conflict(
                    visual_event, best_audio_match
                )
                
                correlations.append(CorrelationResult(
                    visual_event=visual_event,
                    audio_event=best_audio_match,
                    correlation_strength=best_correlation_score,
                    temporal_alignment=self._calculate_temporal_alignment(visual_event, best_audio_match),
                    semantic_alignment=self._calculate_semantic_alignment(visual_event, best_audio_match),
                    conflict_detected=conflict_detected,
                    conflict_type=conflict_type,
                    resolution_strategy=resolution_strategy
                ))
            else:
                # Visual event without good audio correlation
                correlations.append(CorrelationResult(
                    visual_event=visual_event,
                    audio_event=None,
                    correlation_strength=0.5,
                    temporal_alignment=1.0,
                    semantic_alignment=0.0,
                    conflict_detected=False
                ))
        
        # Add remaining unmatched audio events
        for i, audio_event in enumerate(audio_events):
            if i not in used_audio_events:
                correlations.append(CorrelationResult(
                    visual_event=None,
                    audio_event=audio_event,
                    correlation_strength=0.5,
                    temporal_alignment=1.0,
                    semantic_alignment=0.0,
                    conflict_detected=False
                ))
        
        return correlations
    
    def _calculate_temporal_alignment(self, visual_event: TemporalEvent, audio_event: TemporalEvent) -> float:
        """Calculate how well two events are aligned temporally"""
        try:
            # Calculate overlap between events
            visual_start = visual_event.timestamp
            visual_end = visual_event.timestamp + visual_event.duration
            audio_start = audio_event.timestamp
            audio_end = audio_event.timestamp + audio_event.duration
            
            # Calculate overlap
            overlap_start = max(visual_start, audio_start)
            overlap_end = min(visual_end, audio_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # Calculate total duration covered by both events
            total_start = min(visual_start, audio_start)
            total_end = max(visual_end, audio_end)
            total_duration = total_end - total_start
            
            if total_duration == 0:
                return 1.0  # Perfect alignment if both are instantaneous
            
            # Alignment score based on overlap ratio
            alignment_score = overlap_duration / total_duration
            return min(alignment_score * 2, 1.0)  # Boost score for good overlaps
            
        except Exception:
            return 0.5  # Default moderate alignment
    
    def _calculate_semantic_alignment(self, visual_event: TemporalEvent, audio_event: TemporalEvent) -> float:
        """Calculate semantic similarity between visual and audio events"""
        try:
            # Simple semantic alignment based on event types and content
            visual_content = visual_event.content.lower()
            audio_content = audio_event.content.lower()
            
            # Type-based alignment
            type_alignment = 0.0
            
            # Speech and visual text alignment
            if visual_event.event_type == 'visual_text' and audio_event.event_type == 'speech':
                # Check if visual text matches spoken content
                visual_text = visual_event.metadata.get('text', '').lower()
                if visual_text and any(word in audio_content for word in visual_text.split() if len(word) > 3):
                    type_alignment = 0.9
                else:
                    type_alignment = 0.6  # Still related (text + speech)
            
            # Movement and sound effects
            elif visual_event.event_type == 'movement' and audio_event.event_type == 'sound_effect':
                type_alignment = 0.7  # Movement often correlates with sound effects
            
            # Scene changes and music/speech changes
            elif visual_event.event_type == 'scene_change' and audio_event.event_type in ['music', 'speech']:
                type_alignment = 0.6  # Scene changes often align with audio changes
            
            # Objects and related speech
            elif visual_event.event_type == 'object' and audio_event.event_type == 'speech':
                # Check if object is mentioned in speech
                object_name = visual_event.metadata.get('class_name', '').lower()
                if object_name and object_name in audio_content:
                    type_alignment = 0.8
                else:
                    type_alignment = 0.4  # Weak correlation
            
            # Music and movement
            elif visual_event.event_type == 'movement' and audio_event.event_type == 'music':
                type_alignment = 0.7  # Movement often correlates with music
            
            # Default weak alignment for other combinations
            else:
                type_alignment = 0.3
            
            # Content-based alignment (simple keyword matching)
            content_alignment = 0.0
            visual_words = set(visual_content.split())
            audio_words = set(audio_content.split())
            
            if visual_words and audio_words:
                common_words = visual_words.intersection(audio_words)
                if common_words:
                    content_alignment = min(len(common_words) / max(len(visual_words), len(audio_words)), 1.0)
            
            # Combine type and content alignment
            semantic_score = max(type_alignment, content_alignment)
            return semantic_score
            
        except Exception:
            return 0.3  # Default weak alignment
    
    def _detect_conflict(self, visual_event: TemporalEvent, audio_event: TemporalEvent) -> Tuple[bool, Optional[str], Optional[str]]:
        """Detect conflicts between visual and audio events"""
        try:
            # Check for contradictory information
            visual_content = visual_event.content.lower()
            audio_content = audio_event.content.lower()
            
            # Confidence-based conflicts
            if abs(visual_event.confidence - audio_event.confidence) > 0.4:
                return True, "confidence_mismatch", "use_higher_confidence"
            
            # Content-based conflicts
            # Example: Visual shows text saying one thing, audio says another
            if visual_event.event_type == 'visual_text' and audio_event.event_type == 'speech':
                visual_text = visual_event.metadata.get('text', '').lower()
                if visual_text and len(visual_text) > 2:
                    # Simple contradiction detection (would be more sophisticated in practice)
                    contradictory_pairs = [
                        (['yes', 'true', 'correct'], ['no', 'false', 'incorrect']),
                        (['start', 'begin'], ['stop', 'end']),
                        (['up', 'increase'], ['down', 'decrease'])
                    ]
                    
                    for positive_words, negative_words in contradictory_pairs:
                        visual_has_positive = any(word in visual_text for word in positive_words)
                        visual_has_negative = any(word in visual_text for word in negative_words)
                        audio_has_positive = any(word in audio_content for word in positive_words)
                        audio_has_negative = any(word in audio_content for word in negative_words)
                        
                        if (visual_has_positive and audio_has_negative) or (visual_has_negative and audio_has_positive):
                            return True, "content_contradiction", "note_both_sources"
            
            # Temporal conflicts (events that shouldn't happen simultaneously)
            if visual_event.event_type == 'scene_change' and audio_event.event_type == 'speech':
                # Scene changes during continuous speech might indicate editing
                if audio_event.duration > 10.0:  # Long speech
                    return True, "temporal_inconsistency", "note_possible_editing"
            
            return False, None, None
            
        except Exception:
            return False, None, None
    
    async def generate_coherent_narrative(self, correlations: List[CorrelationResult], 
                                        content_type, segment: Optional[VideoSegment]) -> str:
        """Generate coherent narrative from correlated multi-modal events"""
        try:
            if not correlations:
                return "No multi-modal events to correlate."
            
            narrative_parts = []
            
            # Sort correlations by timestamp
            correlations.sort(key=lambda c: (
                c.visual_event.timestamp if c.visual_event else c.audio_event.timestamp
            ))
            
            # Group correlations into narrative segments
            current_time = 0.0
            for correlation in correlations:
                event_time = (correlation.visual_event.timestamp if correlation.visual_event 
                            else correlation.audio_event.timestamp)
                
                # Generate narrative for this correlation
                narrative_segment = self._generate_correlation_narrative(correlation, content_type)
                
                if narrative_segment:
                    # Add temporal context if there's a significant time gap
                    if event_time - current_time > 5.0:  # 5 second gap
                        narrative_parts.append(f"After a pause, {narrative_segment}")
                    else:
                        narrative_parts.append(narrative_segment)
                    
                    current_time = event_time
            
            # Combine narrative parts with appropriate connectors
            if len(narrative_parts) == 1:
                coherent_narrative = narrative_parts[0]
            elif len(narrative_parts) == 2:
                coherent_narrative = f"{narrative_parts[0]}. Meanwhile, {narrative_parts[1]}"
            else:
                # Use varied connectors for multiple parts
                connectors = ["Subsequently", "At the same time", "Additionally", "Furthermore", "Concurrently"]
                connected_parts = [narrative_parts[0]]
                
                for i, part in enumerate(narrative_parts[1:], 1):
                    connector = connectors[min(i-1, len(connectors)-1)]
                    connected_parts.append(f"{connector.lower()}, {part}")
                
                coherent_narrative = ". ".join(connected_parts)
            
            # Add conflict resolution notes if any
            conflicts = [c for c in correlations if c.conflict_detected]
            if conflicts:
                conflict_note = f" Note: {len(conflicts)} potential inconsistenc{'ies' if len(conflicts) > 1 else 'y'} detected between visual and audio streams"
                coherent_narrative += conflict_note
            
            return coherent_narrative
            
        except Exception as e:
            logger.error(f"Coherent narrative generation failed: {e}")
            return "Multi-modal analysis completed with temporal correlation."
    
    def _generate_correlation_narrative(self, correlation: CorrelationResult, content_type) -> str:
        """Generate narrative text for a single correlation"""
        try:
            if correlation.visual_event and correlation.audio_event:
                # Both visual and audio events
                if correlation.conflict_detected:
                    if correlation.resolution_strategy == "note_both_sources":
                        return f"{correlation.visual_event.content} while {correlation.audio_event.content} (conflicting information noted)"
                    elif correlation.resolution_strategy == "use_higher_confidence":
                        higher_conf_event = (correlation.visual_event if correlation.visual_event.confidence > correlation.audio_event.confidence 
                                           else correlation.audio_event)
                        return f"{higher_conf_event.content} (primary source)"
                    else:
                        return f"{correlation.visual_event.content} with {correlation.audio_event.content}"
                else:
                    # No conflict - combine naturally
                    if correlation.semantic_alignment > 0.7:
                        # High semantic alignment - events reinforce each other
                        return f"{correlation.visual_event.content} accompanied by {correlation.audio_event.content}"
                    else:
                        # Lower alignment - events happen simultaneously
                        return f"{correlation.visual_event.content} while {correlation.audio_event.content}"
            
            elif correlation.visual_event:
                # Visual-only event
                return correlation.visual_event.content
            
            elif correlation.audio_event:
                # Audio-only event
                return correlation.audio_event.content
            
            return ""
            
        except Exception:
            return ""
    
    def _calculate_integration_metrics(self, visual_events: List[TemporalEvent], 
                                     audio_events: List[TemporalEvent], 
                                     correlations: List[CorrelationResult]) -> Dict[str, Any]:
        """Calculate metrics for multi-modal integration quality"""
        try:
            metrics = {
                'total_visual_events': len(visual_events),
                'total_audio_events': len(audio_events),
                'total_correlations': len(correlations),
                'correlated_pairs': len([c for c in correlations if c.visual_event and c.audio_event]),
                'conflicts_detected': len([c for c in correlations if c.conflict_detected]),
                'temporal_alignment_score': 0.0,
                'semantic_alignment_score': 0.0,
                'integration_completeness': 0.0
            }
            
            if correlations:
                # Calculate average alignment scores
                temporal_scores = [c.temporal_alignment for c in correlations if c.temporal_alignment > 0]
                semantic_scores = [c.semantic_alignment for c in correlations if c.semantic_alignment > 0]
                
                metrics['temporal_alignment_score'] = np.mean(temporal_scores) if temporal_scores else 0.0
                metrics['semantic_alignment_score'] = np.mean(semantic_scores) if semantic_scores else 0.0
                
                # Integration completeness (how many events were successfully integrated)
                total_events = len(visual_events) + len(audio_events)
                if total_events > 0:
                    integrated_events = len([c for c in correlations if c.visual_event and c.audio_event])
                    metrics['integration_completeness'] = integrated_events / total_events
            
            return metrics
            
        except Exception as e:
            logger.error(f"Integration metrics calculation failed: {e}")
            return {}
    
    def _event_to_dict(self, event: TemporalEvent) -> Dict[str, Any]:
        """Convert TemporalEvent to dictionary for serialization"""
        return {
            'timestamp': event.timestamp,
            'duration': event.duration,
            'event_type': event.event_type,
            'content': event.content,
            'confidence': event.confidence,
            'source_component': event.source_component,
            'metadata': event.metadata
        }
    
    def _correlation_to_dict(self, correlation: CorrelationResult) -> Dict[str, Any]:
        """Convert CorrelationResult to dictionary for serialization"""
        return {
            'visual_event': self._event_to_dict(correlation.visual_event) if correlation.visual_event else None,
            'audio_event': self._event_to_dict(correlation.audio_event) if correlation.audio_event else None,
            'correlation_strength': correlation.correlation_strength,
            'temporal_alignment': correlation.temporal_alignment,
            'semantic_alignment': correlation.semantic_alignment,
            'conflict_detected': correlation.conflict_detected,
            'conflict_type': correlation.conflict_type,
            'resolution_strategy': correlation.resolution_strategy
        }
    
    def get_supported_operations(self) -> List[str]:
        """Return list of supported operations"""
        return [
            'temporal_alignment',
            'event_correlation',
            'conflict_detection',
            'narrative_generation',
            'multi_modal_integration'
        ]
    
    async def health_check(self) -> bool:
        """Check if component is healthy and ready"""
        return self.is_initialized and self.status != ProcessingStatus.FAILED
    
    async def cleanup(self) -> None:
        """Clean up multi-modal integration resources"""
        logger.info("Cleaning up Multi-Modal Integrator")
        # No specific cleanup needed for this component
        pass