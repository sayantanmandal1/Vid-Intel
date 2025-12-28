"""
Output Formatter - Handles analysis output formatting and validation
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .base import (
    BaseProcessor, AnalysisContext, ProcessingResult, ProcessingStatus,
    SegmentAnalysis, AnalysisReport, VideoSegment
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of output validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    completeness_score: float


@dataclass
class FormattedOutput:
    """Formatted analysis output"""
    formatted_segments: List[str]
    summary: str
    validation_result: ValidationResult
    metadata: Dict[str, Any]


class OutputFormatter(BaseProcessor):
    """
    Handles analysis output formatting and validation
    
    Requirements addressed:
    - 1.3: Format output as "X-Y minutes: [detailed description]"
    - 1.4: Analysis completeness validation
    """
    
    def __init__(self):
        super().__init__("OutputFormatter")
        self.format_pattern = re.compile(r'^\d+-\d+ minutes: .+')
        self.min_description_length = 20
        self.min_completeness_score = 0.7
        
    async def initialize(self) -> bool:
        """Initialize the output formatter"""
        try:
            logger.info("Initializing Output Formatter")
            self.is_initialized = True
            self.status = ProcessingStatus.COMPLETED
            return True
        except Exception as e:
            logger.error(f"Failed to initialize output formatter: {e}")
            self.status = ProcessingStatus.FAILED
            return False
    
    async def process(self, context: AnalysisContext) -> ProcessingResult:
        """Process analysis results for formatting and validation"""
        try:
            logger.info(f"Processing output formatting for session {context.session_id}")
            
            # This method is for interface compliance
            # Main functionality is in format_analysis_output
            return ProcessingResult(
                component_name=self.name,
                status=ProcessingStatus.COMPLETED,
                data={"message": "Output formatter ready"}
            )
            
        except Exception as e:
            logger.error(f"Output formatter processing failed: {e}")
            return ProcessingResult(
                component_name=self.name,
                status=ProcessingStatus.FAILED,
                data={},
                error_message=str(e)
            )
    
    def format_analysis_output(self, report: AnalysisReport) -> FormattedOutput:
        """
        Format analysis report with proper time-based descriptions
        
        Requirements addressed:
        - 1.3: Format output as "X-Y minutes: [detailed description]"
        - 1.4: Analysis completeness validation
        
        Args:
            report: AnalysisReport to format
            
        Returns:
            FormattedOutput with validated and formatted segments
        """
        try:
            logger.info(f"Formatting analysis output for session {report.session_id}")
            
            # Format each segment according to "X-Y minutes: [description]" pattern
            formatted_segments = []
            validation_errors = []
            validation_warnings = []
            
            for i, segment_analysis in enumerate(report.segments):
                try:
                    formatted_segment = self._format_segment(segment_analysis)
                    
                    # Validate format
                    format_validation = self._validate_segment_format(formatted_segment)
                    if not format_validation.is_valid:
                        validation_errors.extend(format_validation.errors)
                        validation_warnings.extend(format_validation.warnings)
                        
                        # Apply format correction
                        formatted_segment = self._correct_segment_format(
                            segment_analysis, formatted_segment
                        )
                        logger.warning(f"Corrected format for segment {i+1}")
                    
                    formatted_segments.append(formatted_segment)
                    
                except Exception as e:
                    logger.error(f"Failed to format segment {i+1}: {e}")
                    # Create fallback formatted segment
                    fallback_segment = self._create_fallback_segment(segment_analysis)
                    formatted_segments.append(fallback_segment)
                    validation_errors.append(f"Segment {i+1} formatting failed, used fallback")
            
            # Validate overall completeness
            completeness_validation = self._validate_completeness(report, formatted_segments)
            validation_errors.extend(completeness_validation.errors)
            validation_warnings.extend(completeness_validation.warnings)
            
            # Create validation result
            validation_result = ValidationResult(
                is_valid=len(validation_errors) == 0,
                errors=validation_errors,
                warnings=validation_warnings,
                completeness_score=completeness_validation.completeness_score
            )
            
            # Format summary with validation info
            formatted_summary = self._format_summary(report, validation_result)
            
            # Create metadata
            metadata = self._create_output_metadata(report, validation_result)
            
            formatted_output = FormattedOutput(
                formatted_segments=formatted_segments,
                summary=formatted_summary,
                validation_result=validation_result,
                metadata=metadata
            )
            
            logger.info(f"Successfully formatted output for session {report.session_id}")
            logger.info(f"Validation: {validation_result.is_valid}, "
                       f"Completeness: {validation_result.completeness_score:.2f}, "
                       f"Errors: {len(validation_errors)}, Warnings: {len(validation_warnings)}")
            
            return formatted_output
            
        except Exception as e:
            logger.error(f"Failed to format analysis output: {e}")
            # Return minimal formatted output with error information
            return self._create_error_output(report, str(e))
    
    def _format_segment(self, segment_analysis: SegmentAnalysis) -> str:
        """
        Format a single segment according to "X-Y minutes: [description]" pattern
        
        Args:
            segment_analysis: SegmentAnalysis to format
            
        Returns:
            Formatted segment string
        """
        # Calculate time range in minutes
        start_min = int(segment_analysis.segment.start_time // 60)
        end_min = int(segment_analysis.segment.end_time // 60)
        
        # Get the best available description
        description = self._get_best_description(segment_analysis)
        
        # Ensure description meets minimum quality standards
        description = self._enhance_description(description, segment_analysis)
        
        # Format according to requirement pattern
        formatted_segment = f"{start_min}-{end_min} minutes: {description}"
        
        logger.debug(f"Formatted segment {segment_analysis.segment.segment_id}: {start_min}-{end_min} minutes")
        
        return formatted_segment
    
    def _get_best_description(self, segment_analysis: SegmentAnalysis) -> str:
        """Get the best available description from segment analysis"""
        # Priority order: combined_narrative > visual + audio > visual > audio > fallback
        
        if segment_analysis.combined_narrative and len(segment_analysis.combined_narrative.strip()) > self.min_description_length:
            return segment_analysis.combined_narrative.strip()
        
        # Combine visual and audio descriptions
        parts = []
        if segment_analysis.visual_description and segment_analysis.visual_description.strip():
            parts.append(segment_analysis.visual_description.strip())
        
        if segment_analysis.transcription and segment_analysis.transcription.strip():
            parts.append(f"Transcript: {segment_analysis.transcription.strip()}")
        elif segment_analysis.audio_description and segment_analysis.audio_description.strip():
            parts.append(segment_analysis.audio_description.strip())
        
        if parts:
            return ". ".join(parts)
        
        # Fallback to basic information
        return "Video segment processed with limited analysis data available."
    
    def _enhance_description(self, description: str, segment_analysis: SegmentAnalysis) -> str:
        """Enhance description with additional context if needed"""
        enhanced_parts = [description]
        
        # Add object detection information if available and not already included
        if (segment_analysis.detected_objects and 
            len(segment_analysis.detected_objects) > 0 and 
            "objects" not in description.lower() and 
            "detected" not in description.lower()):
            
            # Get top 3 most confident objects
            sorted_objects = sorted(
                segment_analysis.detected_objects, 
                key=lambda x: x.get('confidence', 0), 
                reverse=True
            )[:3]
            
            object_names = [obj.get('class_name', 'unknown') for obj in sorted_objects]
            if object_names:
                enhanced_parts.append(f"Key objects detected: {', '.join(object_names)}")
        
        # Add confidence information if low
        if segment_analysis.confidence_score < 0.5:
            enhanced_parts.append("(Analysis confidence: limited)")
        
        return ". ".join(enhanced_parts)
    
    def _validate_segment_format(self, formatted_segment: str) -> ValidationResult:
        """Validate that segment follows the required format pattern"""
        errors = []
        warnings = []
        
        # Check format pattern: "X-Y minutes: [description]"
        if not self.format_pattern.match(formatted_segment):
            errors.append("Segment does not match required 'X-Y minutes: [description]' format")
        
        # Check description length
        if ":" in formatted_segment:
            description_part = formatted_segment.split(":", 1)[1].strip()
            if len(description_part) < self.min_description_length:
                warnings.append("Segment description is shorter than recommended minimum")
        else:
            errors.append("Segment missing description separator ':'")
        
        # Check for empty or placeholder descriptions
        placeholder_phrases = [
            "no description available",
            "processing failed",
            "limited analysis data",
            "analysis not available"
        ]
        
        if any(phrase in formatted_segment.lower() for phrase in placeholder_phrases):
            warnings.append("Segment contains placeholder or error description")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            completeness_score=1.0 if len(errors) == 0 and len(warnings) == 0 else 0.5
        )
    
    def _correct_segment_format(self, segment_analysis: SegmentAnalysis, malformed_segment: str) -> str:
        """Correct malformed segment to match required format"""
        try:
            # Extract time information
            start_min = int(segment_analysis.segment.start_time // 60)
            end_min = int(segment_analysis.segment.end_time // 60)
            
            # Extract description from malformed segment or use fallback
            description = "Video segment analysis"
            if ":" in malformed_segment:
                description = malformed_segment.split(":", 1)[1].strip()
            elif len(malformed_segment) > 20:
                description = malformed_segment
            
            # Ensure minimum description quality
            if len(description) < self.min_description_length:
                description = self._get_best_description(segment_analysis)
            
            return f"{start_min}-{end_min} minutes: {description}"
            
        except Exception as e:
            logger.error(f"Failed to correct segment format: {e}")
            return self._create_fallback_segment(segment_analysis)
    
    def _create_fallback_segment(self, segment_analysis: SegmentAnalysis) -> str:
        """Create fallback formatted segment when formatting fails"""
        start_min = int(segment_analysis.segment.start_time // 60)
        end_min = int(segment_analysis.segment.end_time // 60)
        
        return f"{start_min}-{end_min} minutes: Video segment processed with basic analysis."
    
    def _validate_completeness(self, report: AnalysisReport, formatted_segments: List[str]) -> ValidationResult:
        """Validate overall analysis completeness"""
        errors = []
        warnings = []
        completeness_factors = []
        
        # Check segment coverage
        if len(formatted_segments) == 0:
            errors.append("No segments were successfully formatted")
            completeness_factors.append(0.0)
        else:
            # Calculate expected segments based on video duration
            expected_segments = max(1, int(report.video_metadata.duration // 300) + 
                                  (1 if report.video_metadata.duration % 300 > 0 else 0))
            
            segment_coverage = len(formatted_segments) / expected_segments
            completeness_factors.append(min(1.0, segment_coverage))
            
            if segment_coverage < 0.8:
                warnings.append(f"Only {len(formatted_segments)}/{expected_segments} expected segments were processed")
        
        # Check description quality
        quality_scores = []
        for segment in formatted_segments:
            if ":" in segment:
                description = segment.split(":", 1)[1].strip()
                # Score based on length and content quality
                length_score = min(1.0, len(description) / 100)  # 100 chars = full score
                
                # Penalty for placeholder content
                placeholder_penalty = 0.0
                placeholder_phrases = ["limited analysis", "processing failed", "not available"]
                if any(phrase in description.lower() for phrase in placeholder_phrases):
                    placeholder_penalty = 0.3
                
                quality_score = max(0.0, length_score - placeholder_penalty)
                quality_scores.append(quality_score)
        
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            completeness_factors.append(avg_quality)
            
            if avg_quality < 0.6:
                warnings.append("Average segment description quality is below recommended threshold")
        
        # Check for transcription coverage
        segments_with_transcription = sum(1 for seg in report.segments if seg.transcription.strip())
        transcription_coverage = segments_with_transcription / len(report.segments) if report.segments else 0
        completeness_factors.append(transcription_coverage)
        
        if transcription_coverage < 0.3:
            warnings.append("Low transcription coverage - most segments lack audio transcription")
        
        # Check for object detection coverage
        segments_with_objects = sum(1 for seg in report.segments if seg.detected_objects)
        object_coverage = segments_with_objects / len(report.segments) if report.segments else 0
        completeness_factors.append(object_coverage)
        
        if object_coverage < 0.5:
            warnings.append("Low object detection coverage - many segments lack visual object detection")
        
        # Calculate overall completeness score
        overall_completeness = sum(completeness_factors) / len(completeness_factors) if completeness_factors else 0.0
        
        # Add completeness warnings
        if overall_completeness < self.min_completeness_score:
            warnings.append(f"Overall analysis completeness ({overall_completeness:.2f}) is below recommended threshold ({self.min_completeness_score})")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            completeness_score=overall_completeness
        )
    
    def _format_summary(self, report: AnalysisReport, validation_result: ValidationResult) -> str:
        """Format analysis summary with validation information"""
        summary_parts = []
        
        # Add original summary if available
        if report.summary and report.summary.strip():
            summary_parts.append(report.summary.strip())
        
        # Add validation summary
        validation_summary = self._create_validation_summary(report, validation_result)
        summary_parts.append(validation_summary)
        
        return "\n\n".join(summary_parts)
    
    def _create_validation_summary(self, report: AnalysisReport, validation_result: ValidationResult) -> str:
        """Create summary of validation results"""
        summary_parts = [
            f"Analysis Quality Report:",
            f"- Segments processed: {len(report.segments)}",
            f"- Completeness score: {validation_result.completeness_score:.2f}",
            f"- Format validation: {'PASSED' if validation_result.is_valid else 'FAILED'}"
        ]
        
        if report.segments:
            avg_confidence = sum(seg.confidence_score for seg in report.segments) / len(report.segments)
            summary_parts.append(f"- Average confidence: {avg_confidence:.2f}")
        
        if validation_result.errors:
            summary_parts.append(f"- Validation errors: {len(validation_result.errors)}")
        
        if validation_result.warnings:
            summary_parts.append(f"- Validation warnings: {len(validation_result.warnings)}")
        
        return "\n".join(summary_parts)
    
    def _create_output_metadata(self, report: AnalysisReport, validation_result: ValidationResult) -> Dict[str, Any]:
        """Create metadata for formatted output"""
        return {
            "session_id": report.session_id,
            "formatted_at": datetime.now().isoformat(),
            "total_segments": len(report.segments),
            "validation_passed": validation_result.is_valid,
            "completeness_score": validation_result.completeness_score,
            "error_count": len(validation_result.errors),
            "warning_count": len(validation_result.warnings),
            "content_type": report.content_type.value,
            "video_duration_minutes": report.video_metadata.duration / 60,
            "processing_time": report.processing_time
        }
    
    def _create_error_output(self, report: AnalysisReport, error_message: str) -> FormattedOutput:
        """Create error output when formatting fails completely"""
        error_segment = f"0-{int(report.video_metadata.duration // 60)} minutes: Analysis formatting failed - {error_message}"
        
        validation_result = ValidationResult(
            is_valid=False,
            errors=[f"Output formatting failed: {error_message}"],
            warnings=[],
            completeness_score=0.0
        )
        
        return FormattedOutput(
            formatted_segments=[error_segment],
            summary=f"Analysis completed with formatting errors. {error_message}",
            validation_result=validation_result,
            metadata={
                "session_id": report.session_id,
                "formatted_at": datetime.now().isoformat(),
                "error": True,
                "error_message": error_message
            }
        )
    
    def validate_output_format(self, formatted_segments: List[str]) -> ValidationResult:
        """
        Validate a list of formatted segments
        
        Args:
            formatted_segments: List of formatted segment strings
            
        Returns:
            ValidationResult with validation details
        """
        all_errors = []
        all_warnings = []
        segment_scores = []
        
        for i, segment in enumerate(formatted_segments):
            segment_validation = self._validate_segment_format(segment)
            
            if segment_validation.errors:
                all_errors.extend([f"Segment {i+1}: {error}" for error in segment_validation.errors])
            
            if segment_validation.warnings:
                all_warnings.extend([f"Segment {i+1}: {warning}" for warning in segment_validation.warnings])
            
            segment_scores.append(segment_validation.completeness_score)
        
        overall_score = sum(segment_scores) / len(segment_scores) if segment_scores else 0.0
        
        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            completeness_score=overall_score
        )
    
    async def cleanup(self) -> None:
        """Clean up formatter resources"""
        logger.info("Cleaning up Output Formatter")
        # No specific cleanup needed for formatter
        pass