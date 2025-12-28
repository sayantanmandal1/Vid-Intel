"""
Analysis Orchestrator - Central coordinator for video analysis pipeline
"""
import asyncio
import logging
import time
import math
import gc
import psutil
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue, Empty
import threading
from dataclasses import dataclass

try:
    from moviepy import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("MoviePy not available - using fallback metadata extraction")

from .base import (
    BaseProcessor, AnalysisContext, ProcessingResult, ProcessingStatus,
    VideoSegment, AnalysisSession, SegmentAnalysis, AnalysisReport,
    VideoMetadata, ContentType
)
from .error_handler import (
    handle_component_error, ErrorSeverity, ErrorCategory, get_error_handler
)
from .output_formatter import OutputFormatter, FormattedOutput

logger = logging.getLogger(__name__)


@dataclass
class ResourceLimits:
    """Resource limits for video processing"""
    max_memory_mb: int = 2048  # Maximum memory usage in MB
    max_concurrent_segments: int = 4  # Maximum segments processed in parallel
    max_file_size_mb: int = 1024  # Maximum video file size in MB
    segment_batch_size: int = 2  # Number of segments to process in each batch


@dataclass
class ProcessingQueue:
    """Processing queue for managing segment analysis tasks"""
    pending_segments: Queue
    active_segments: Dict[str, asyncio.Task]
    completed_segments: List[str]
    failed_segments: List[str]
    max_concurrent: int
    
    def __post_init__(self):
        self.pending_segments = Queue()
        self.active_segments = {}
        self.completed_segments = []
        self.failed_segments = []


class ResourceManager:
    """Manages system resources during video processing"""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.current_memory_mb = 0
        self.active_tasks = 0
        self._lock = asyncio.Lock()  # Use asyncio.Lock instead of threading.Lock
    
    def check_memory_usage(self) -> float:
        """Check current memory usage in MB"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb
        except Exception:
            # Return 0 if unable to check memory
            return 0.0
    
    async def can_process_segment(self) -> bool:
        """Check if we can process another segment based on resource limits"""
        async with self._lock:
            memory_usage = self.check_memory_usage()
            return (
                self.active_tasks < self.limits.max_concurrent_segments and
                memory_usage < self.limits.max_memory_mb
            )
    
    async def acquire_resources(self) -> bool:
        """Acquire resources for processing a segment"""
        async with self._lock:
            # Check directly without calling can_process_segment to avoid deadlock
            memory_usage = self.check_memory_usage()
            can_process = (
                self.active_tasks < self.limits.max_concurrent_segments and
                memory_usage < self.limits.max_memory_mb
            )
            if can_process:
                self.active_tasks += 1
                return True
            return False
    
    async def release_resources(self):
        """Release resources after processing a segment"""
        async with self._lock:
            if self.active_tasks > 0:
                self.active_tasks -= 1
        
        # Force garbage collection to free memory
        gc.collect()
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource usage status"""
        memory_usage = self.check_memory_usage()
        return {
            'memory_usage_mb': memory_usage,
            'memory_limit_mb': self.limits.max_memory_mb,
            'memory_usage_percent': (memory_usage / self.limits.max_memory_mb) * 100,
            'active_tasks': self.active_tasks,
            'max_concurrent_tasks': self.limits.max_concurrent_segments,
            'can_process_more': asyncio.create_task(self.can_process_segment()) if hasattr(self, '_lock') else False
        }


class AnalysisOrchestrator(BaseProcessor):
    """Central coordinator that manages the entire video analysis pipeline"""
    
    def __init__(self, database_manager=None, progress_callback: Optional[Callable] = None, resource_limits: Optional[ResourceLimits] = None):
        super().__init__("AnalysisOrchestrator")
        self.components = {}
        self.active_sessions = {}
        self.database_manager = database_manager
        self.progress_callback = progress_callback
        
        # Initialize resource management
        self.resource_limits = resource_limits or ResourceLimits()
        self.resource_manager = ResourceManager(self.resource_limits)
        
        # Initialize thread pools for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=self.resource_limits.max_concurrent_segments)
        self.process_pool = ProcessPoolExecutor(max_workers=min(4, self.resource_limits.max_concurrent_segments))
        
        # Processing queues for different sessions
        self.processing_queues: Dict[str, ProcessingQueue] = {}
        
    async def initialize(self) -> bool:
        """Initialize the orchestrator"""
        try:
            logger.info("Initializing Analysis Orchestrator")
            
            # Register error handling fallback strategies
            error_handler = get_error_handler()
            error_handler.register_fallback_strategy(
                'orchestrator', 
                self._orchestrator_fallback
            )
            
            # Set retry policy for orchestrator
            error_handler.set_retry_policy('orchestrator', {
                'max_retries': 2,
                'base_delay': 2.0,
                'max_delay': 10.0
            })
            
            self.is_initialized = True
            self.status = ProcessingStatus.COMPLETED
            return True
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            self.status = ProcessingStatus.FAILED
            
            # Handle initialization error
            await handle_component_error(
                component_name=self.name,
                error=e,
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.INITIALIZATION,
                allow_retry=False,
                allow_fallback=False
            )
            return False
    
    def register_component(self, name: str, component: BaseProcessor):
        """Register a processing component"""
        self.components[name] = component
        logger.info(f"Registered component: {name}")
    
    def set_progress_callback(self, callback: Callable):
        """Set callback function for progress updates"""
        self.progress_callback = callback
    
    async def analyze_video(self, video_path: str, session_id: str) -> AnalysisReport:
        """
        Main entry point for video analysis
        Coordinates all processing components and combines results
        
        Requirements addressed:
        - 1.2: Divide video into 5-minute segments for detailed analysis
        - 1.5: Compile all segments into comprehensive Analysis_Report
        - 5.1: Create unique session identifier for analysis
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting video analysis for session {session_id}")
            
            # Update progress: Starting analysis
            await self._update_progress(session_id, "Initializing analysis", 0)
            
            # Extract video metadata using moviepy
            metadata = await self._extract_metadata(video_path)
            logger.info(f"Video metadata extracted: {metadata.duration:.2f}s, {metadata.resolution}, {metadata.fps} fps")
            
            # Update progress: Metadata extracted
            await self._update_progress(session_id, "Metadata extracted", 5)
            
            # Classify content type
            content_type = await self._classify_content(video_path, metadata)
            logger.info(f"Content classified as: {content_type.value}")
            
            # Update progress: Content classified
            await self._update_progress(session_id, "Content classified", 10)
            
            # Create segments using 5-minute intervals
            segments = self._create_segments(metadata.duration, segment_length=300.0)
            logger.info(f"Created {len(segments)} segments for analysis")
            
            # Create analysis session
            session = AnalysisSession(
                session_id=session_id,
                video_path=video_path,
                metadata=metadata,
                content_type=content_type,
                segments=segments,
                created_at=datetime.now(),
                status=ProcessingStatus.IN_PROGRESS
            )
            
            self.active_sessions[session_id] = session
            
            # Save session to database if available
            if self.database_manager:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.database_manager.create_analysis_session, session
                )
            
            # Update progress: Session created
            await self._update_progress(session_id, "Session created", 15)
            
            # Check file size and memory requirements
            file_size_mb = metadata.file_size / (1024 * 1024) if metadata.file_size else 0
            if file_size_mb > self.resource_limits.max_file_size_mb:
                logger.warning(f"Large file detected: {file_size_mb:.1f}MB (limit: {self.resource_limits.max_file_size_mb}MB)")
                # Adjust processing parameters for large files
                self.resource_limits.max_concurrent_segments = max(1, self.resource_limits.max_concurrent_segments // 2)
                self.resource_limits.segment_batch_size = 1
            
            # Initialize processing queue for this session
            processing_queue = ProcessingQueue(
                pending_segments=Queue(),
                active_segments={},
                completed_segments=[],
                failed_segments=[],
                max_concurrent=self.resource_limits.max_concurrent_segments
            )
            self.processing_queues[session_id] = processing_queue
            
            # Process segments with parallel processing and resource management
            segment_analyses = await self._process_segments_parallel(session, segments, processing_queue)
            
            # Clean up processing queue
            del self.processing_queues[session_id]
            
            # Update progress: Segments processed
            await self._update_progress(session_id, "Combining analyses", 85)
            
            # Combine analyses into final report
            report = await self.combine_analyses(session, segment_analyses)
            report.processing_time = time.time() - start_time
            
            # Update session status
            session.status = ProcessingStatus.COMPLETED
            
            # Save final report to database
            if self.database_manager:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.database_manager.update_session_status, 
                    session_id, ProcessingStatus.COMPLETED, report.summary, report.processing_time,
                    report.output_validation_passed, report.output_completeness_score,
                    report.validation_errors, report.validation_warnings
                )
            
            # Final progress update
            await self._update_progress(session_id, "Analysis complete", 100)
            
            logger.info(f"Completed video analysis for session {session_id} in {report.processing_time:.2f}s")
            return report
            
        except Exception as e:
            logger.error(f"Video analysis failed for session {session_id}: {e}")
            
            # Update session status to failed
            if session_id in self.active_sessions:
                self.active_sessions[session_id].status = ProcessingStatus.FAILED
            
            if self.database_manager:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.database_manager.update_session_status, 
                    session_id, ProcessingStatus.FAILED, str(e)
                )
            
            await self._update_progress(session_id, f"Analysis failed: {str(e)}", 0)
            
            # Handle orchestrator error with comprehensive error handling
            error_result = await handle_component_error(
                component_name=self.name,
                error=e,
                context={
                    'session_id': session_id,
                    'video_path': video_path,
                    'operation': 'analyze_video'
                },
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.PROCESSING,
                allow_retry=False,  # Don't retry full analysis automatically
                allow_fallback=True
            )
            
            # If fallback was successful, return partial results
            if error_result.status == ProcessingStatus.COMPLETED and error_result.data.get('fallback_used'):
                logger.info(f"Using fallback analysis results for session {session_id}")
                # Create minimal report from fallback
                return self._create_fallback_report(session_id, video_path, metadata, str(e))
            
            raise
    
    async def _process_segments_parallel(self, session: AnalysisSession, segments: List[VideoSegment], processing_queue: ProcessingQueue) -> List[SegmentAnalysis]:
        """
        Process video segments in parallel with resource management
        
        Requirements addressed:
        - 1.1: Extract both visual and audio content for processing
        - 1.2: Divide video into segments for detailed analysis
        
        Args:
            session: Analysis session containing video metadata
            segments: List of video segments to process
            processing_queue: Queue for managing parallel processing
            
        Returns:
            List of completed segment analyses
        """
        try:
            logger.info(f"Starting parallel processing of {len(segments)} segments for session {session.session_id}")
            
            # Add all segments to the processing queue
            for segment in segments:
                processing_queue.pending_segments.put(segment)
            
            segment_analyses = []
            total_segments = len(segments)
            completed_count = 0
            
            # Process segments in batches to manage memory usage
            batch_size = self.resource_limits.segment_batch_size
            
            while completed_count < total_segments:
                # Start processing tasks up to the concurrent limit
                active_tasks = []
                batch_segments = []
                
                # Fill up to batch size or remaining segments
                for _ in range(min(batch_size, total_segments - completed_count)):
                    if not processing_queue.pending_segments.empty():
                        try:
                            segment = processing_queue.pending_segments.get_nowait()
                            batch_segments.append(segment)
                        except Empty:
                            break
                
                if not batch_segments:
                    break
                
                logger.info(f"Processing batch of {len(batch_segments)} segments (completed: {completed_count}/{total_segments})")
                
                # Check resource availability before starting batch
                resource_status = self.resource_manager.get_resource_status()
                logger.info(f"Resource status: {resource_status['memory_usage_mb']:.1f}MB memory, {resource_status['active_tasks']} active tasks")
                
                # Process batch segments in parallel
                batch_tasks = []
                for segment in batch_segments:
                    if await self.resource_manager.acquire_resources():
                        task = asyncio.create_task(
                            self._process_segment_with_resources(session, segment)
                        )
                        batch_tasks.append((segment, task))
                        processing_queue.active_segments[segment.segment_id] = task
                    else:
                        # Put segment back in queue if resources not available
                        processing_queue.pending_segments.put(segment)
                        logger.warning(f"Insufficient resources for segment {segment.segment_id}, queuing for later")
                
                if not batch_tasks:
                    # Wait a bit and try again if no resources available
                    await asyncio.sleep(1.0)
                    continue
                
                # Wait for batch completion
                batch_results = await asyncio.gather(
                    *[task for _, task in batch_tasks], 
                    return_exceptions=True
                )
                
                # Process batch results
                for i, (segment, task) in enumerate(batch_tasks):
                    result = batch_results[i]
                    
                    # Remove from active segments
                    if segment.segment_id in processing_queue.active_segments:
                        del processing_queue.active_segments[segment.segment_id]
                    
                    if isinstance(result, Exception):
                        logger.error(f"Segment {segment.segment_id} processing failed: {result}")
                        processing_queue.failed_segments.append(segment.segment_id)
                        
                        # Create fallback analysis for failed segment
                        fallback_analysis = SegmentAnalysis(
                            segment=segment,
                            processing_time=0.0,
                            confidence_score=0.0,
                            visual_description=f"Processing failed: {str(result)}",
                            audio_description="",
                            combined_narrative=f"Analysis failed for segment {segment.start_time:.0f}-{segment.end_time:.0f} minutes due to processing error."
                        )
                        segment_analyses.append(fallback_analysis)
                    else:
                        processing_queue.completed_segments.append(segment.segment_id)
                        segment_analyses.append(result)
                        
                        # Save segment analysis to database
                        if self.database_manager:
                            await asyncio.get_event_loop().run_in_executor(
                                None, self.database_manager.save_segment_analysis, session.session_id, result
                            )
                        
                        # Update session results
                        session.results[segment.segment_id] = result
                    
                    completed_count += 1
                    
                    # Update progress
                    progress_percent = 15 + (completed_count / total_segments) * 70  # 15-85% for segment processing
                    await self._update_progress(
                        session.session_id, 
                        f"Processed {completed_count}/{total_segments} segments", 
                        progress_percent
                    )
                
                # Log batch completion
                logger.info(f"Completed batch processing: {completed_count}/{total_segments} segments done")
                
                # Force garbage collection between batches for memory management
                gc.collect()
                
                # Brief pause between batches to allow system recovery
                if completed_count < total_segments:
                    await asyncio.sleep(0.5)
            
            # Sort analyses by segment start time to maintain order
            segment_analyses.sort(key=lambda x: x.segment.start_time)
            
            logger.info(f"Parallel processing completed: {len(processing_queue.completed_segments)} successful, {len(processing_queue.failed_segments)} failed")
            
            return segment_analyses
            
        except Exception as e:
            logger.error(f"Parallel segment processing failed for session {session.session_id}: {e}")
            
            # Clean up any remaining active tasks
            for task in processing_queue.active_segments.values():
                if not task.done():
                    task.cancel()
            
            raise
    
    async def _process_segment_with_resources(self, session: AnalysisSession, segment: VideoSegment) -> SegmentAnalysis:
        """
        Process a single segment with resource management
        
        Args:
            session: Analysis session
            segment: Video segment to process
            
        Returns:
            Completed segment analysis
        """
        try:
            # Process the segment
            analysis = await self.process_segment(session, segment)
            return analysis
            
        except Exception as e:
            logger.error(f"Error processing segment {segment.segment_id}: {e}")
            # Create fallback analysis for failed segment
            fallback_analysis = SegmentAnalysis(
                segment=segment,
                processing_time=0.0,
                confidence_score=0.0,
                visual_description=f"Processing failed: {str(e)}",
                audio_description="",
                combined_narrative=f"Analysis failed for segment {segment.start_time:.0f}-{segment.end_time:.0f} minutes due to processing error."
            )
            return fallback_analysis
            
        finally:
            # Always release resources when done
            await self.resource_manager.release_resources()
    
    async def process_segment(self, session: AnalysisSession, segment: VideoSegment) -> SegmentAnalysis:
        """
        Process a single video segment
        Coordinates visual and audio analysis for the segment
        
        Requirements addressed:
        - 1.3: Provide descriptions of visual elements, character actions, dialogue, scene changes
        - 4.2: Display real-time progress and preliminary results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing segment {segment.segment_id} ({segment.start_time:.0f}-{segment.end_time:.0f}s)")
            
            context = AnalysisContext(
                video_path=session.video_path,
                session_id=session.session_id,
                content_type=session.content_type,
                metadata=session.metadata,
                segment=segment
            )
            
            # Initialize segment analysis
            analysis = SegmentAnalysis(
                segment=segment,
                processing_time=0.0
            )
            
            # Process visual and audio in parallel for efficiency
            tasks = []
            task_names = []
            
            if 'visual_extractor' in self.components:
                tasks.append(self.components['visual_extractor'].process(context))
                task_names.append('visual_extractor')
                logger.debug(f"Added visual processing task for segment {segment.segment_id}")
            
            if 'audio_processor' in self.components:
                tasks.append(self.components['audio_processor'].process(context))
                task_names.append('audio_processor')
                logger.debug(f"Added audio processing task for segment {segment.segment_id}")
            
            # Wait for all processing to complete
            if tasks:
                logger.debug(f"Starting parallel processing of {len(tasks)} components for segment {segment.segment_id}")
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results from each component
                for i, result in enumerate(results):
                    component_name = task_names[i]
                    
                    if isinstance(result, Exception):
                        logger.warning(f"Component {component_name} failed for segment {segment.segment_id}: {result}")
                        
                        # Handle component failure with error handler
                        error_result = await handle_component_error(
                            component_name=component_name,
                            error=result,
                            context={
                                'session_id': session.session_id,
                                'segment_id': segment.segment_id,
                                'operation': 'process_segment'
                            },
                            severity=ErrorSeverity.MEDIUM,
                            category=ErrorCategory.PROCESSING,
                            allow_retry=True,
                            allow_fallback=True
                        )
                        
                        # Use fallback result if available
                        if error_result.status == ProcessingStatus.COMPLETED:
                            result = error_result
                        else:
                            continue
                    
                    if not isinstance(result, ProcessingResult):
                        logger.warning(f"Invalid result type from {component_name}: {type(result)}")
                        continue
                        
                    # Process visual results
                    if result.component_name == 'visual_extractor' and result.status == ProcessingStatus.COMPLETED:
                        analysis.visual_description = result.data.get('description', '')
                        analysis.detected_objects = result.data.get('objects', [])
                        logger.debug(f"Visual analysis completed for segment {segment.segment_id}")
                    
                    # Process audio results
                    elif result.component_name == 'audio_processor' and result.status == ProcessingStatus.COMPLETED:
                        analysis.audio_description = result.data.get('description', '')
                        analysis.transcription = result.data.get('transcription', '')
                        logger.debug(f"Audio analysis completed for segment {segment.segment_id}")
                    
                    # Save component result to database if available
                    if self.database_manager:
                        await asyncio.get_event_loop().run_in_executor(
                            None, self.database_manager.save_processing_result,
                            session.session_id, segment.segment_id, result.component_name,
                            result.data, result.status, result.error_message, result.processing_time
                        )
            
            # Perform multi-modal data integration if available
            if 'multimodal_integrator' in self.components:
                logger.debug(f"Performing multi-modal integration for segment {segment.segment_id}")
                
                # Prepare context with visual and audio data for multi-modal integration
                multimodal_context = AnalysisContext(
                    video_path=session.video_path,
                    session_id=session.session_id,
                    content_type=session.content_type,
                    metadata=session.metadata,
                    segment=segment
                )
                
                # Add processed data to context
                multimodal_context.visual_data = {
                    'description': analysis.visual_description,
                    'objects': analysis.detected_objects
                }
                multimodal_context.audio_data = {
                    'description': analysis.audio_description,
                    'transcription': analysis.transcription
                }
                
                # Add detailed data from component results for better integration
                for i, result in enumerate(results):
                    if isinstance(result, ProcessingResult) and result.status == ProcessingStatus.COMPLETED:
                        if result.component_name == 'visual_extractor':
                            multimodal_context.visual_data.update(result.data)
                        elif result.component_name == 'audio_processor':
                            multimodal_context.audio_data.update(result.data)
                
                try:
                    multimodal_result = await self.components['multimodal_integrator'].process(multimodal_context)
                    if multimodal_result.status == ProcessingStatus.COMPLETED:
                        # Use the coherent narrative from multi-modal integration
                        analysis.combined_narrative = multimodal_result.data.get('coherent_narrative', '')
                        
                        # Store integration metrics for analysis quality assessment
                        analysis.integration_metrics = multimodal_result.data.get('integration_metrics', {})
                        
                        logger.debug(f"Multi-modal integration completed for segment {segment.segment_id}")
                    else:
                        logger.warning(f"Multi-modal integration failed for segment {segment.segment_id}: {multimodal_result.error_message}")
                        analysis.combined_narrative = self._create_fallback_narrative(analysis)
                        
                except Exception as e:
                    logger.warning(f"Multi-modal integration failed for segment {segment.segment_id}: {e}")
                    
                    # Handle multimodal integration error
                    error_result = await handle_component_error(
                        component_name='multimodal_integrator',
                        error=e,
                        context={
                            'session_id': session.session_id,
                            'segment_id': segment.segment_id,
                            'visual_data': multimodal_context.visual_data,
                            'audio_data': multimodal_context.audio_data
                        },
                        severity=ErrorSeverity.MEDIUM,
                        category=ErrorCategory.PROCESSING,
                        allow_retry=False,
                        allow_fallback=True
                    )
                    
                    if error_result.status == ProcessingStatus.COMPLETED and error_result.data.get('fallback_used'):
                        analysis.combined_narrative = error_result.data.get('coherent_narrative', '')
                        analysis.integration_metrics = error_result.data.get('integration_metrics', {})
                    else:
                        analysis.combined_narrative = self._create_fallback_narrative(analysis)
            
            # Generate combined narrative using LLM if available and multi-modal integration didn't provide one
            elif 'llm_integration' in self.components:
                logger.debug(f"Generating combined narrative for segment {segment.segment_id}")
                
                # Prepare context with visual and audio data for LLM
                llm_context = AnalysisContext(
                    video_path=session.video_path,
                    session_id=session.session_id,
                    content_type=session.content_type,
                    metadata=session.metadata,
                    segment=segment
                )
                
                # Add processed data to context
                llm_context.visual_data = {
                    'description': analysis.visual_description,
                    'objects': analysis.detected_objects
                }
                llm_context.audio_data = {
                    'description': analysis.audio_description,
                    'transcription': analysis.transcription
                }
                
                try:
                    narrative_result = await self.components['llm_integration'].process(llm_context)
                    if narrative_result.status == ProcessingStatus.COMPLETED:
                        analysis.combined_narrative = narrative_result.data.get('narrative', '')
                        logger.debug(f"LLM narrative generated for segment {segment.segment_id}")
                    else:
                        logger.warning(f"LLM processing failed for segment {segment.segment_id}: {narrative_result.error_message}")
                        # Create fallback narrative
                        analysis.combined_narrative = self._create_fallback_narrative(analysis)
                        
                except Exception as e:
                    logger.warning(f"LLM integration failed for segment {segment.segment_id}: {e}")
                    
                    # Handle LLM integration error
                    error_result = await handle_component_error(
                        component_name='llm_integration',
                        error=e,
                        context={
                            'session_id': session.session_id,
                            'segment_id': segment.segment_id,
                            'visual_data': llm_context.visual_data,
                            'audio_data': llm_context.audio_data
                        },
                        severity=ErrorSeverity.MEDIUM,
                        category=ErrorCategory.PROCESSING,
                        allow_retry=True,
                        allow_fallback=True
                    )
                    
                    if error_result.status == ProcessingStatus.COMPLETED and error_result.data.get('fallback_used'):
                        analysis.combined_narrative = error_result.data.get('narrative', '')
                    else:
                        analysis.combined_narrative = self._create_fallback_narrative(analysis)
            else:
                # Create fallback narrative when neither multi-modal nor LLM is available
                analysis.combined_narrative = self._create_fallback_narrative(analysis)
            
            # Calculate confidence score based on available data
            analysis.confidence_score = self._calculate_segment_confidence(analysis)
            analysis.processing_time = time.time() - start_time
            
            logger.info(f"Completed segment {segment.segment_id} processing in {analysis.processing_time:.2f}s (confidence: {analysis.confidence_score:.2f})")
            return analysis
            
        except Exception as e:
            logger.error(f"Segment processing failed for {segment.segment_id}: {e}")
            
            # Handle segment processing error
            error_result = await handle_component_error(
                component_name=self.name,
                error=e,
                context={
                    'session_id': session.session_id,
                    'segment_id': segment.segment_id,
                    'operation': 'process_segment'
                },
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.PROCESSING,
                allow_retry=False,
                allow_fallback=True
            )
            
            # Return partial analysis with error information
            if error_result.status == ProcessingStatus.COMPLETED and error_result.data.get('fallback_used'):
                return SegmentAnalysis(
                    segment=segment,
                    processing_time=time.time() - start_time,
                    confidence_score=0.2,
                    visual_description=error_result.data.get('description', 'Processing failed'),
                    audio_description="",
                    combined_narrative=f"Partial analysis for segment {segment.start_time:.0f}-{segment.end_time:.0f} minutes (error handled gracefully)."
                )
            else:
                return SegmentAnalysis(
                    segment=segment,
                    processing_time=time.time() - start_time,
                    confidence_score=0.0,
                    visual_description=f"Processing failed: {str(e)}",
                    audio_description="",
                    combined_narrative=f"Analysis failed for segment {segment.start_time:.0f}-{segment.end_time:.0f} minutes due to processing error."
                )
    
    async def combine_analyses(self, session: AnalysisSession, segments: List[SegmentAnalysis]) -> AnalysisReport:
        """
        Combine segment analyses into final report
        Generates overall summary and formats output
        
        Requirements addressed:
        - 1.4: Format output as "X-Y minutes: [detailed description]"
        - 1.5: Compile all segments into comprehensive Analysis_Report
        """
        try:
            logger.info(f"Combining analyses for session {session.session_id} ({len(segments)} segments)")
            
            # Create report with basic information
            report = AnalysisReport(
                session_id=session.session_id,
                video_metadata=session.metadata,
                content_type=session.content_type,
                segments=segments,
                total_duration=session.metadata.duration,
                created_at=datetime.now()
            )
            
            # Use output formatter for proper formatting and validation
            output_formatter = OutputFormatter()
            await output_formatter.initialize()
            
            # Format and validate the analysis output
            formatted_output = output_formatter.format_analysis_output(report)
            
            # Log validation results
            validation = formatted_output.validation_result
            if not validation.is_valid:
                logger.warning(f"Output validation failed for session {session.session_id}: {len(validation.errors)} errors")
                for error in validation.errors:
                    logger.warning(f"Validation error: {error}")
            
            if validation.warnings:
                logger.info(f"Output validation warnings for session {session.session_id}: {len(validation.warnings)} warnings")
                for warning in validation.warnings:
                    logger.info(f"Validation warning: {warning}")
            
            logger.info(f"Output completeness score: {validation.completeness_score:.2f}")
            
            # Use formatted segments from the formatter
            formatted_segments = formatted_output.formatted_segments
            
            # Use enhanced summary from formatter (includes validation info)
            report.summary = formatted_output.summary
            
            # Store validation results in the report
            report.output_validation_passed = validation.is_valid
            report.output_completeness_score = validation.completeness_score
            report.validation_errors = validation.errors
            report.validation_warnings = validation.warnings
            
            # Generate overall summary if LLM is available
            if 'llm_integration' in self.components:
                logger.info("Generating overall summary using LLM")
                
                summary_context = AnalysisContext(
                    video_path=session.video_path,
                    session_id=session.session_id,
                    content_type=session.content_type,
                    metadata=session.metadata
                )
                
                # Prepare segment data for summary generation
                summary_context.segments_data = [
                    {
                        'time_range': f"{int(seg.segment.start_time // 60)}-{int(seg.segment.end_time // 60)} minutes",
                        'visual': seg.visual_description,
                        'audio': seg.audio_description,
                        'transcription': seg.transcription,
                        'narrative': seg.combined_narrative,
                        'confidence': seg.confidence_score,
                        'objects_count': len(seg.detected_objects)
                    }
                    for seg in segments
                ]
                
                # Add overall video statistics
                summary_context.video_stats = {
                    'total_duration_minutes': session.metadata.duration / 60,
                    'total_segments': len(segments),
                    'content_type': session.content_type.value,
                    'average_confidence': sum(seg.confidence_score for seg in segments) / len(segments) if segments else 0,
                    'total_objects_detected': sum(len(seg.detected_objects) for seg in segments)
                }
                
                try:
                    summary_result = await self.components['llm_integration'].process(summary_context)
                    if summary_result.status == ProcessingStatus.COMPLETED:
                        report.summary = summary_result.data.get('summary', '')
                        logger.info("LLM summary generated successfully")
                    else:
                        logger.warning(f"LLM summary generation failed: {summary_result.error_message}")
                        report.summary = self._create_fallback_summary(segments, session)
                        
                except Exception as e:
                    logger.warning(f"LLM summary generation failed: {e}")
                    
                    # Handle LLM summary error
                    error_result = await handle_component_error(
                        component_name='llm_integration',
                        error=e,
                        context={
                            'session_id': session.session_id,
                            'operation': 'generate_summary',
                            'segments_count': len(segments)
                        },
                        severity=ErrorSeverity.MEDIUM,
                        category=ErrorCategory.PROCESSING,
                        allow_retry=True,
                        allow_fallback=True
                    )
                    
                    if error_result.status == ProcessingStatus.COMPLETED and error_result.data.get('fallback_used'):
                        report.summary = error_result.data.get('summary', '')
                    else:
                        report.summary = self._create_fallback_summary(segments, session)
            else:
                logger.info("Creating fallback summary (no LLM available)")
                report.summary = self._create_fallback_summary(segments, session)
            
            # Add formatted segments to report
            report.formatted_segments = formatted_segments
            
            # Calculate overall statistics
            report.total_objects_detected = sum(len(seg.detected_objects) for seg in segments)
            report.average_confidence = sum(seg.confidence_score for seg in segments) / len(segments) if segments else 0
            report.segments_with_transcription = sum(1 for seg in segments if seg.transcription.strip())
            
            logger.info(f"Successfully combined analyses for session {session.session_id}")
            logger.info(f"Report stats: {len(segments)} segments, {report.total_objects_detected} objects, {report.average_confidence:.2f} avg confidence")
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to combine analyses for session {session.session_id}: {e}")
            
            # Handle combine analyses error
            error_result = await handle_component_error(
                component_name=self.name,
                error=e,
                context={
                    'session_id': session.session_id,
                    'segments_count': len(segments),
                    'operation': 'combine_analyses'
                },
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.PROCESSING,
                allow_retry=False,
                allow_fallback=True
            )
            
            # Return minimal report with error information
            if error_result.status == ProcessingStatus.COMPLETED and error_result.data.get('fallback_used'):
                return AnalysisReport(
                    session_id=session.session_id,
                    video_metadata=session.metadata,
                    content_type=session.content_type,
                    segments=segments,
                    total_duration=session.metadata.duration,
                    summary=error_result.data.get('summary', f"Analysis compilation completed with errors: {str(e)}"),
                    created_at=datetime.now()
                )
            else:
                return AnalysisReport(
                    session_id=session.session_id,
                    video_metadata=session.metadata,
                    content_type=session.content_type,
                    segments=segments,
                    total_duration=session.metadata.duration,
                    summary=f"Analysis compilation failed: {str(e)}",
                    created_at=datetime.now()
                )
    
    def _create_segments(self, duration: float, segment_length: float = 300.0) -> List[VideoSegment]:
        """
        Divide video into segments (default 5-minute intervals)
        
        Requirements addressed:
        - 1.2: Divide video into 5-minute segments for detailed analysis
        
        Args:
            duration: Video duration in seconds
            segment_length: Segment length in seconds (default 300 = 5 minutes)
            
        Returns:
            List of VideoSegment objects with proper time boundaries
        """
        segments = []
        current_time = 0.0
        segment_count = math.ceil(duration / segment_length)
        
        logger.info(f"Creating {segment_count} segments for {duration:.2f}s video (segment length: {segment_length}s)")
        
        while current_time < duration:
            end_time = min(current_time + segment_length, duration)
            segment = VideoSegment(
                start_time=current_time,
                end_time=end_time,
                duration=end_time - current_time
            )
            segments.append(segment)
            
            logger.debug(f"Created segment {len(segments)}: {current_time:.1f}s - {end_time:.1f}s ({segment.duration:.1f}s)")
            current_time = end_time
        
        logger.info(f"Successfully created {len(segments)} segments for {duration:.2f}s video")
        return segments
    
    async def _extract_metadata(self, video_path: str) -> VideoMetadata:
        """
        Extract video metadata using moviepy or fallback method
        
        Args:
            video_path: Path to the video file
            
        Returns:
            VideoMetadata object with extracted information
        """
        try:
            logger.info(f"Extracting metadata from: {video_path}")
            
            if MOVIEPY_AVAILABLE:
                # Use moviepy to extract metadata
                video = VideoFileClip(video_path)
                
                # Get basic video properties
                duration = video.duration  # in seconds
                fps = video.fps
                resolution = f"{video.w}x{video.h}"
                
                # Close the video clip to free resources
                video.close()
            else:
                # Fallback metadata extraction
                logger.warning("MoviePy not available - using fallback metadata extraction")
                duration = 300.0  # Default 5 minutes
                fps = 30.0
                resolution = "1920x1080"
            
            # Get file information
            file_path = Path(video_path)
            file_size = file_path.stat().st_size if file_path.exists() else 0
            file_format = file_path.suffix.lower().lstrip('.')
            
            # Generate simple hash based on file size and name for now
            # In production, you might want to use actual file hash
            file_hash = f"{file_path.name}_{file_size}_{int(duration)}"
            
            metadata = VideoMetadata(
                duration=duration,
                fps=fps,
                resolution=resolution,
                format=file_format,
                file_size=file_size,
                hash=file_hash
            )
            
            logger.info(f"Extracted metadata: {duration:.2f}s, {resolution}, {fps:.2f} fps, {file_size} bytes")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from {video_path}: {e}")
            
            # Handle metadata extraction error
            error_result = await handle_component_error(
                component_name=self.name,
                error=e,
                context={
                    'video_path': video_path,
                    'operation': 'extract_metadata'
                },
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.DATA,
                allow_retry=True,
                allow_fallback=True
            )
            
            # Return placeholder metadata if extraction fails
            file_size = Path(video_path).stat().st_size if Path(video_path).exists() else 0
            return VideoMetadata(
                duration=300.0,  # Default 5 minutes
                fps=30.0,
                resolution="1920x1080",
                format="mp4",
                file_size=file_size,
                hash=f"placeholder_{int(time.time())}"
            )
    
    async def _classify_content(self, video_path: str, metadata: VideoMetadata) -> ContentType:
        """Classify video content type"""
        if 'content_classifier' in self.components:
            context = AnalysisContext(
                video_path=video_path,
                session_id="temp",
                content_type=ContentType.GENERAL,
                metadata=metadata
            )
            result = await self.components['content_classifier'].process(context)
            if result.status == ProcessingStatus.COMPLETED:
                return ContentType(result.data.get('content_type', ContentType.GENERAL.value))
        
        return ContentType.GENERAL
    
    def _create_fallback_narrative(self, analysis: SegmentAnalysis) -> str:
        """Create a fallback narrative when LLM is not available"""
        parts = []
        
        if analysis.visual_description:
            parts.append(f"Visual: {analysis.visual_description}")
        
        if analysis.transcription:
            parts.append(f"Audio: {analysis.transcription}")
        elif analysis.audio_description:
            parts.append(f"Audio: {analysis.audio_description}")
        
        if analysis.detected_objects:
            object_names = [obj.get('class_name', 'unknown') for obj in analysis.detected_objects[:5]]  # Limit to 5
            parts.append(f"Objects detected: {', '.join(object_names)}")
        
        return ". ".join(parts) if parts else "No analysis data available."
    
    def _combine_segment_descriptions(self, analysis: SegmentAnalysis) -> str:
        """Combine visual and audio descriptions for a segment"""
        parts = []
        
        if analysis.visual_description:
            parts.append(analysis.visual_description)
        
        if analysis.transcription:
            parts.append(f"Transcript: {analysis.transcription}")
        elif analysis.audio_description:
            parts.append(analysis.audio_description)
        
        return ". ".join(parts) if parts else "No description available."
    
    def _create_fallback_summary(self, segments: List[SegmentAnalysis], session: AnalysisSession) -> str:
        """Create a fallback summary when LLM is not available"""
        total_segments = len(segments)
        duration_minutes = session.metadata.duration / 60
        
        # Count segments with different types of content
        segments_with_transcription = sum(1 for seg in segments if seg.transcription.strip())
        segments_with_objects = sum(1 for seg in segments if seg.detected_objects)
        total_objects = sum(len(seg.detected_objects) for seg in segments)
        
        summary_parts = [
            f"Video analysis completed for {duration_minutes:.1f} minute video",
            f"Processed {total_segments} segments",
            f"Content type: {session.content_type.value}"
        ]
        
        if segments_with_transcription > 0:
            summary_parts.append(f"Speech detected in {segments_with_transcription} segments")
        
        if total_objects > 0:
            summary_parts.append(f"Detected {total_objects} objects across all segments")
        
        return ". ".join(summary_parts) + "."
    
    def _calculate_segment_confidence(self, analysis: SegmentAnalysis) -> float:
        """Calculate confidence score for a segment analysis"""
        scores = []
        
        # Visual confidence
        if analysis.visual_description:
            scores.append(0.8 if len(analysis.visual_description) > 50 else 0.5)
        
        if analysis.detected_objects:
            avg_object_confidence = sum(obj.get('confidence', 0.5) for obj in analysis.detected_objects) / len(analysis.detected_objects)
            scores.append(avg_object_confidence)
        
        # Audio confidence
        if analysis.transcription:
            scores.append(0.9 if len(analysis.transcription) > 20 else 0.6)
        elif analysis.audio_description:
            scores.append(0.7 if len(analysis.audio_description) > 20 else 0.4)
        
        # Combined narrative confidence
        if analysis.combined_narrative:
            scores.append(0.8 if len(analysis.combined_narrative) > 50 else 0.6)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    async def process(self, context: AnalysisContext) -> ProcessingResult:
        """Process method for BaseProcessor interface"""
        # This orchestrator doesn't process individual contexts
        # It coordinates other components
        return ProcessingResult(
            component_name=self.name,
            status=ProcessingStatus.COMPLETED,
            data={"message": "Orchestrator coordinates other components"}
        )
    
    async def cleanup(self) -> None:
        """Clean up orchestrator resources"""
        logger.info("Cleaning up Analysis Orchestrator")
        
        # Cancel any active processing tasks
        for processing_queue in self.processing_queues.values():
            for task in processing_queue.active_segments.values():
                if not task.done():
                    task.cancel()
        
        self.processing_queues.clear()
        self.active_sessions.clear()
        
        # Cleanup thread pools
        try:
            self.thread_pool.shutdown(wait=True, timeout=30)
            logger.info("Thread pool shutdown completed")
        except Exception as e:
            logger.warning(f"Thread pool shutdown failed: {e}")
        
        try:
            self.process_pool.shutdown(wait=True, timeout=30)
            logger.info("Process pool shutdown completed")
        except Exception as e:
            logger.warning(f"Process pool shutdown failed: {e}")
        
        # Cleanup all registered components
        for component in self.components.values():
            try:
                await component.cleanup()
            except Exception as e:
                logger.warning(f"Failed to cleanup component: {e}")
        
        # Force final garbage collection
        gc.collect()
    
    async def _update_progress(self, session_id: str, message: str, progress_percent: float):
        """
        Update analysis progress and notify callbacks
        
        Args:
            session_id: Session identifier
            message: Progress message
            progress_percent: Progress percentage (0-100)
        """
        try:
            logger.debug(f"Progress update for {session_id}: {message} ({progress_percent:.1f}%)")
            
            # Update database progress if available
            if self.database_manager and session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                completed_segments = len(session.results) if hasattr(session, 'results') and session.results else 0
                
                await asyncio.get_event_loop().run_in_executor(
                    None, self.database_manager.update_progress, 
                    session_id, completed_segments, message
                )
            
            # Call progress callback if available
            if self.progress_callback:
                try:
                    await self.progress_callback(session_id, message, progress_percent)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to update progress: {e}")

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of an active session including resource usage
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session status information or None if not found
        """
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            completed_segments = len(session.results) if hasattr(session, 'results') and session.results else 0
            total_segments = len(session.segments) if session.segments else 0
            
            # Get processing queue status if available
            queue_status = {}
            if session_id in self.processing_queues:
                queue = self.processing_queues[session_id]
                queue_status = {
                    'active_segments': len(queue.active_segments),
                    'completed_segments': len(queue.completed_segments),
                    'failed_segments': len(queue.failed_segments),
                    'pending_segments': queue.pending_segments.qsize()
                }
            
            # Get resource status
            resource_status = self.resource_manager.get_resource_status()
            
            return {
                'session_id': session_id,
                'status': session.status.value,
                'progress': (completed_segments / total_segments * 100) if total_segments > 0 else 0,
                'segments_completed': completed_segments,
                'total_segments': total_segments,
                'content_type': session.content_type.value,
                'video_duration': session.metadata.duration,
                'created_at': session.created_at.isoformat() if hasattr(session.created_at, 'isoformat') else str(session.created_at),
                'queue_status': queue_status,
                'resource_status': resource_status
            }
        return None
    
    def get_resource_status(self) -> Dict[str, Any]:
        """
        Get current system resource status
        
        Returns:
            Dictionary with resource usage information
        """
        return self.resource_manager.get_resource_status()
    
    def update_resource_limits(self, new_limits: ResourceLimits):
        """
        Update resource limits during runtime
        
        Args:
            new_limits: New resource limits to apply
        """
        logger.info(f"Updating resource limits: {new_limits}")
        self.resource_limits = new_limits
        self.resource_manager.limits = new_limits
        
        # Update thread pool sizes if needed
        current_max_workers = self.thread_pool._max_workers
        if current_max_workers != new_limits.max_concurrent_segments:
            logger.info(f"Updating thread pool size from {current_max_workers} to {new_limits.max_concurrent_segments}")
            # Note: ThreadPoolExecutor doesn't support dynamic resizing
            # In production, you might want to recreate the pool
    
    async def _orchestrator_fallback(self, context: Dict[str, Any]) -> ProcessingResult:
        """Fallback strategy for orchestrator failures"""
        try:
            session_id = context.get('session_id', 'unknown')
            operation = context.get('operation', 'unknown')
            
            logger.info(f"Executing orchestrator fallback for {operation} in session {session_id}")
            
            if operation == 'analyze_video':
                # Create minimal analysis report
                return ProcessingResult(
                    component_name=self.name,
                    status=ProcessingStatus.COMPLETED,
                    data={
                        'fallback_report': True,
                        'summary': 'Video analysis completed with limited functionality due to system errors',
                        'segments': [],
                        'processing_time': 0.0
                    }
                )
            elif operation == 'process_segment':
                # Create minimal segment analysis
                return ProcessingResult(
                    component_name=self.name,
                    status=ProcessingStatus.COMPLETED,
                    data={
                        'description': 'Segment processed with limited analysis due to component failures',
                        'confidence': 0.3,
                        'fallback_used': True
                    }
                )
            else:
                # Generic fallback
                return ProcessingResult(
                    component_name=self.name,
                    status=ProcessingStatus.COMPLETED,
                    data={
                        'message': f'Operation {operation} completed with fallback mechanism',
                        'fallback_used': True
                    }
                )
                
        except Exception as e:
            logger.error(f"Orchestrator fallback failed: {e}")
            return ProcessingResult(
                component_name=self.name,
                status=ProcessingStatus.FAILED,
                data={},
                error_message=f"Fallback mechanism failed: {str(e)}"
            )
    
    def _create_fallback_report(self, session_id: str, video_path: str, metadata: VideoMetadata, error_message: str) -> AnalysisReport:
        """Create a fallback analysis report when main analysis fails"""
        return AnalysisReport(
            session_id=session_id,
            video_metadata=metadata,
            content_type=ContentType.GENERAL,
            segments=[],
            total_duration=metadata.duration,
            summary=f"Video analysis completed with errors. System used fallback mechanisms to provide basic analysis. Original error: {error_message}",
            processing_time=0.0,
            created_at=datetime.now(),
            formatted_segments=[f"0-{int(metadata.duration//60)} minutes: Analysis completed with system fallback due to processing errors."]
        )