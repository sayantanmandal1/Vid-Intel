"""
Base classes and interfaces for all processing components
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import uuid
from datetime import datetime


class ContentType(Enum):
    """Video content types for adaptive analysis"""
    MUSIC_VIDEO = "music_video"
    GAMING = "gaming"
    EDUCATIONAL = "educational"
    NARRATIVE = "narrative"
    DOCUMENTARY = "documentary"
    GENERAL = "general"


class ProcessingStatus(Enum):
    """Processing status for components"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class VideoMetadata:
    """Video file metadata"""
    duration: float
    fps: float
    resolution: str
    format: str
    file_size: int
    hash: str


@dataclass
class VideoSegment:
    """Video segment for processing"""
    start_time: float
    end_time: float
    duration: float
    segment_id: str = None
    
    def __post_init__(self):
        if self.segment_id is None:
            self.segment_id = str(uuid.uuid4())


@dataclass
class AnalysisContext:
    """Context information for analysis"""
    video_path: str
    session_id: str
    content_type: ContentType
    metadata: VideoMetadata
    segment: Optional[VideoSegment] = None
    visual_data: Optional[Dict[str, Any]] = None
    audio_data: Optional[Dict[str, Any]] = None
    segments_data: Optional[List[Dict[str, Any]]] = None
    video_stats: Optional[Dict[str, Any]] = None


@dataclass
class ProcessingResult:
    """Base result from processing components"""
    component_name: str
    status: ProcessingStatus
    data: Dict[str, Any]
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BaseProcessor(ABC):
    """Base class for all processing components"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_initialized = False
        self.status = ProcessingStatus.NOT_STARTED
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the processor with required models/resources"""
        pass
    
    @abstractmethod
    async def process(self, context: AnalysisContext) -> ProcessingResult:
        """Process the given context and return results"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources"""
        pass
    
    def get_status(self) -> ProcessingStatus:
        """Get current processing status"""
        return self.status


class ComponentInterface(ABC):
    """Interface for component communication"""
    
    @abstractmethod
    def get_supported_operations(self) -> List[str]:
        """Return list of supported operations"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if component is healthy and ready"""
        pass


@dataclass
class AnalysisSession:
    """Analysis session data"""
    session_id: str
    video_path: str
    metadata: VideoMetadata
    content_type: ContentType
    segments: List[VideoSegment]
    created_at: datetime
    status: ProcessingStatus = ProcessingStatus.NOT_STARTED
    results: Dict[str, ProcessingResult] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = {}


@dataclass
class SegmentAnalysis:
    """Analysis results for a video segment"""
    segment: VideoSegment
    visual_description: str = ""
    audio_description: str = ""
    combined_narrative: str = ""
    detected_objects: List[Dict] = None
    transcription: str = ""
    confidence_score: float = 0.0
    processing_time: float = 0.0
    integration_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.detected_objects is None:
            self.detected_objects = []
        if self.integration_metrics is None:
            self.integration_metrics = {}


@dataclass
class AnalysisReport:
    """Final comprehensive analysis report"""
    session_id: str
    video_metadata: VideoMetadata
    content_type: ContentType
    segments: List[SegmentAnalysis]
    summary: str = ""
    total_duration: float = 0.0
    processing_time: float = 0.0
    created_at: datetime = None
    formatted_segments: List[str] = None
    total_objects_detected: int = 0
    average_confidence: float = 0.0
    segments_with_transcription: int = 0
    # Output validation fields
    output_validation_passed: bool = True
    output_completeness_score: float = 1.0
    validation_errors: List[str] = None
    validation_warnings: List[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.formatted_segments is None:
            self.formatted_segments = []
        if self.validation_errors is None:
            self.validation_errors = []
        if self.validation_warnings is None:
            self.validation_warnings = []