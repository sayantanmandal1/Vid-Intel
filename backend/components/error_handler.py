"""
Comprehensive Error Handler - Centralized error handling and recovery mechanisms
"""
import logging
import traceback
import time
from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import asyncio

from .base import ProcessingStatus, ProcessingResult

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    INITIALIZATION = "initialization"
    PROCESSING = "processing"
    RESOURCE = "resource"
    NETWORK = "network"
    DATA = "data"
    TIMEOUT = "timeout"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"


@dataclass
class ErrorInfo:
    """Comprehensive error information"""
    error_id: str
    component_name: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    timestamp: datetime
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    fallback_used: bool = False
    retry_count: int = 0


class ComponentErrorHandler:
    """Centralized error handling for all components"""
    
    def __init__(self):
        self.error_history: List[ErrorInfo] = []
        self.component_health: Dict[str, Dict[str, Any]] = {}
        self.fallback_strategies: Dict[str, Callable] = {}
        self.retry_policies: Dict[str, Dict[str, Any]] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Default retry policies
        self.default_retry_policy = {
            'max_retries': 3,
            'base_delay': 1.0,
            'max_delay': 30.0,
            'exponential_backoff': True,
            'jitter': True
        }
        
        # Initialize circuit breaker states
        self._initialize_circuit_breakers()
        
    def _initialize_circuit_breakers(self):
        """Initialize circuit breaker states for components"""
        components = [
            'visual_extractor', 'audio_processor', 'llm_integration',
            'content_classifier', 'multimodal_integrator', 'orchestrator'
        ]
        
        for component in components:
            self.circuit_breakers[component] = {
                'state': 'closed',  # closed, open, half_open
                'failure_count': 0,
                'failure_threshold': 5,
                'recovery_timeout': 60.0,  # seconds
                'last_failure_time': None,
                'success_count': 0,
                'success_threshold': 3  # for half_open -> closed transition
            }
    
    async def handle_error(
        self,
        component_name: str,
        error: Exception,
        context: Dict[str, Any] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        allow_retry: bool = True,
        allow_fallback: bool = True
    ) -> ProcessingResult:
        """
        Comprehensive error handling with recovery mechanisms
        
        Args:
            component_name: Name of the component that failed
            error: The exception that occurred
            context: Additional context information
            severity: Error severity level
            category: Error category for classification
            allow_retry: Whether to attempt retry
            allow_fallback: Whether to use fallback mechanisms
            
        Returns:
            ProcessingResult with error handling outcome
        """
        try:
            # Generate unique error ID
            error_id = f"{component_name}_{int(time.time() * 1000)}"
            
            # Create error info
            error_info = ErrorInfo(
                error_id=error_id,
                component_name=component_name,
                error_type=type(error).__name__,
                error_message=str(error),
                severity=severity,
                category=category,
                timestamp=datetime.now(),
                context=context or {},
                stack_trace=traceback.format_exc()
            )
            
            # Log error with appropriate level
            self._log_error(error_info)
            
            # Update component health
            self._update_component_health(component_name, error_info)
            
            # Check circuit breaker
            if self._is_circuit_open(component_name):
                logger.warning(f"Circuit breaker open for {component_name}, skipping processing")
                return self._create_circuit_breaker_result(component_name, error_info)
            
            # Attempt recovery based on error category and severity
            recovery_result = await self._attempt_recovery(
                component_name, error_info, allow_retry, allow_fallback
            )
            
            # Update error history
            self.error_history.append(error_info)
            
            # Limit error history size
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-500:]
            
            return recovery_result
            
        except Exception as recovery_error:
            logger.critical(f"Error handler itself failed: {recovery_error}")
            return ProcessingResult(
                component_name=component_name,
                status=ProcessingStatus.FAILED,
                data={},
                error_message=f"Critical error handling failure: {str(recovery_error)}"
            )
    
    async def _attempt_recovery(
        self,
        component_name: str,
        error_info: ErrorInfo,
        allow_retry: bool,
        allow_fallback: bool
    ) -> ProcessingResult:
        """Attempt error recovery using various strategies"""
        
        # Strategy 1: Retry with exponential backoff
        if allow_retry and self._should_retry(component_name, error_info):
            retry_result = await self._attempt_retry(component_name, error_info)
            if retry_result.status == ProcessingStatus.COMPLETED:
                error_info.recovery_attempted = True
                error_info.recovery_successful = True
                self._record_circuit_breaker_success(component_name)
                return retry_result
        
        # Strategy 2: Use fallback mechanisms
        if allow_fallback and component_name in self.fallback_strategies:
            fallback_result = await self._attempt_fallback(component_name, error_info)
            if fallback_result.status == ProcessingStatus.COMPLETED:
                error_info.recovery_attempted = True
                error_info.fallback_used = True
                return fallback_result
        
        # Strategy 3: Graceful degradation
        degraded_result = self._attempt_graceful_degradation(component_name, error_info)
        if degraded_result.status == ProcessingStatus.COMPLETED:
            error_info.recovery_attempted = True
            error_info.fallback_used = True
            return degraded_result
        
        # All recovery attempts failed
        self._record_circuit_breaker_failure(component_name)
        
        return ProcessingResult(
            component_name=component_name,
            status=ProcessingStatus.FAILED,
            data={
                'error_id': error_info.error_id,
                'error_category': error_info.category.value,
                'severity': error_info.severity.value,
                'recovery_attempted': error_info.recovery_attempted,
                'fallback_used': error_info.fallback_used
            },
            error_message=error_info.error_message
        )
    
    def _should_retry(self, component_name: str, error_info: ErrorInfo) -> bool:
        """Determine if retry should be attempted"""
        # Don't retry critical errors
        if error_info.severity == ErrorSeverity.CRITICAL:
            return False
        
        # Don't retry certain error categories
        non_retryable_categories = [
            ErrorCategory.DATA,  # Bad data won't fix itself
            ErrorCategory.INITIALIZATION  # Initialization errors need manual fix
        ]
        
        if error_info.category in non_retryable_categories:
            return False
        
        # Check retry policy
        policy = self.retry_policies.get(component_name, self.default_retry_policy)
        max_retries = policy.get('max_retries', 3)
        
        return error_info.retry_count < max_retries
    
    async def _attempt_retry(self, component_name: str, error_info: ErrorInfo) -> ProcessingResult:
        """Attempt retry with exponential backoff"""
        policy = self.retry_policies.get(component_name, self.default_retry_policy)
        
        error_info.retry_count += 1
        
        # Calculate delay
        base_delay = policy.get('base_delay', 1.0)
        max_delay = policy.get('max_delay', 30.0)
        
        if policy.get('exponential_backoff', True):
            delay = min(base_delay * (2 ** (error_info.retry_count - 1)), max_delay)
        else:
            delay = base_delay
        
        # Add jitter if enabled
        if policy.get('jitter', True):
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        logger.info(f"Retrying {component_name} after {delay:.2f}s (attempt {error_info.retry_count})")
        
        # Wait before retry
        await asyncio.sleep(delay)
        
        # Return a result indicating retry was attempted
        # The actual retry logic would be handled by the calling component
        return ProcessingResult(
            component_name=component_name,
            status=ProcessingStatus.FAILED,  # Still failed, but retry was attempted
            data={'retry_attempted': True, 'retry_count': error_info.retry_count},
            error_message=f"Retry {error_info.retry_count} attempted"
        )
    
    async def _attempt_fallback(self, component_name: str, error_info: ErrorInfo) -> ProcessingResult:
        """Attempt fallback mechanism"""
        try:
            fallback_strategy = self.fallback_strategies[component_name]
            
            logger.info(f"Attempting fallback for {component_name}")
            
            # Execute fallback strategy
            fallback_result = await fallback_strategy(error_info.context)
            
            if isinstance(fallback_result, ProcessingResult):
                fallback_result.data['fallback_used'] = True
                return fallback_result
            else:
                # Convert to ProcessingResult if needed
                return ProcessingResult(
                    component_name=component_name,
                    status=ProcessingStatus.COMPLETED,
                    data={'fallback_result': fallback_result, 'fallback_used': True}
                )
                
        except Exception as fallback_error:
            logger.error(f"Fallback failed for {component_name}: {fallback_error}")
            return ProcessingResult(
                component_name=component_name,
                status=ProcessingStatus.FAILED,
                data={'fallback_attempted': True, 'fallback_failed': True},
                error_message=f"Fallback failed: {str(fallback_error)}"
            )
    
    def _attempt_graceful_degradation(self, component_name: str, error_info: ErrorInfo) -> ProcessingResult:
        """Attempt graceful degradation"""
        
        # Component-specific degradation strategies
        degradation_strategies = {
            'visual_extractor': self._degrade_visual_processing,
            'audio_processor': self._degrade_audio_processing,
            'llm_integration': self._degrade_llm_processing,
            'content_classifier': self._degrade_content_classification,
            'multimodal_integrator': self._degrade_multimodal_integration
        }
        
        if component_name in degradation_strategies:
            try:
                logger.info(f"Attempting graceful degradation for {component_name}")
                return degradation_strategies[component_name](error_info)
            except Exception as degradation_error:
                logger.error(f"Graceful degradation failed for {component_name}: {degradation_error}")
        
        # Default degradation: return minimal result
        return ProcessingResult(
            component_name=component_name,
            status=ProcessingStatus.COMPLETED,
            data={
                'degraded': True,
                'error_handled': True,
                'description': f"{component_name} processing failed but system continued"
            },
            error_message=f"Graceful degradation applied for {component_name}"
        )
    
    def _degrade_visual_processing(self, error_info: ErrorInfo) -> ProcessingResult:
        """Graceful degradation for visual processing failures"""
        return ProcessingResult(
            component_name='visual_extractor',
            status=ProcessingStatus.COMPLETED,
            data={
                'objects': [],
                'scenes': [],
                'text_elements': [],
                'movement_analysis': {'movement_detected': False},
                'visual_style': {},
                'description': 'Visual processing unavailable - analysis continued with audio only',
                'confidence': 0.0,
                'degraded': True
            }
        )
    
    def _degrade_audio_processing(self, error_info: ErrorInfo) -> ProcessingResult:
        """Graceful degradation for audio processing failures"""
        return ProcessingResult(
            component_name='audio_processor',
            status=ProcessingStatus.COMPLETED,
            data={
                'transcription': '',
                'speakers': [],
                'music_analysis': {},
                'sound_effects': [],
                'speech_patterns': {},
                'audio_features': {},
                'description': 'Audio processing unavailable - analysis continued with visual only',
                'confidence': 0.0,
                'degraded': True
            }
        )
    
    def _degrade_llm_processing(self, error_info: ErrorInfo) -> ProcessingResult:
        """Graceful degradation for LLM processing failures"""
        context = error_info.context
        
        # Create basic description from available data
        basic_description = "Content analysis completed"
        
        if context.get('visual_data'):
            basic_description += " with visual elements"
        if context.get('audio_data'):
            basic_description += " and audio content"
        
        return ProcessingResult(
            component_name='llm_integration',
            status=ProcessingStatus.COMPLETED,
            data={
                'narrative': basic_description,
                'summary': basic_description,
                'scene_interpretation': 'Scene analysis completed',
                'model_used': 'fallback',
                'confidence': 0.3,
                'degraded': True
            }
        )
    
    def _degrade_content_classification(self, error_info: ErrorInfo) -> ProcessingResult:
        """Graceful degradation for content classification failures"""
        return ProcessingResult(
            component_name='content_classifier',
            status=ProcessingStatus.COMPLETED,
            data={
                'content_type': 'general',
                'analysis_strategy': {
                    'focus_areas': ['general_content'],
                    'components': ['visual_extractor', 'audio_processor'],
                    'visual_priority': 'medium',
                    'audio_priority': 'medium',
                    'description_style': 'balanced'
                },
                'classification_confidence': 0.3,
                'degraded': True
            }
        )
    
    def _degrade_multimodal_integration(self, error_info: ErrorInfo) -> ProcessingResult:
        """Graceful degradation for multimodal integration failures"""
        context = error_info.context
        
        # Combine visual and audio descriptions manually
        combined_narrative = []
        
        if context.get('visual_data', {}).get('description'):
            combined_narrative.append(f"Visual: {context['visual_data']['description']}")
        
        if context.get('audio_data', {}).get('transcription'):
            combined_narrative.append(f"Audio: {context['audio_data']['transcription']}")
        elif context.get('audio_data', {}).get('description'):
            combined_narrative.append(f"Audio: {context['audio_data']['description']}")
        
        narrative = ". ".join(combined_narrative) if combined_narrative else "Multimodal analysis completed"
        
        return ProcessingResult(
            component_name='multimodal_integrator',
            status=ProcessingStatus.COMPLETED,
            data={
                'coherent_narrative': narrative,
                'integration_metrics': {
                    'temporal_alignment': 0.5,
                    'content_coherence': 0.5,
                    'confidence': 0.3
                },
                'degraded': True
            }
        )
    
    def _is_circuit_open(self, component_name: str) -> bool:
        """Check if circuit breaker is open for component"""
        breaker = self.circuit_breakers.get(component_name, {})
        
        if breaker.get('state') == 'open':
            # Check if recovery timeout has passed
            last_failure = breaker.get('last_failure_time')
            recovery_timeout = breaker.get('recovery_timeout', 60.0)
            
            if last_failure and (time.time() - last_failure) > recovery_timeout:
                # Move to half-open state
                breaker['state'] = 'half_open'
                breaker['success_count'] = 0
                logger.info(f"Circuit breaker for {component_name} moved to half-open state")
                return False
            
            return True
        
        return False
    
    def _record_circuit_breaker_failure(self, component_name: str):
        """Record failure for circuit breaker"""
        breaker = self.circuit_breakers.get(component_name, {})
        breaker['failure_count'] = breaker.get('failure_count', 0) + 1
        breaker['last_failure_time'] = time.time()
        
        failure_threshold = breaker.get('failure_threshold', 5)
        
        if breaker['failure_count'] >= failure_threshold:
            breaker['state'] = 'open'
            logger.warning(f"Circuit breaker opened for {component_name} after {breaker['failure_count']} failures")
    
    def _record_circuit_breaker_success(self, component_name: str):
        """Record success for circuit breaker"""
        breaker = self.circuit_breakers.get(component_name, {})
        
        if breaker.get('state') == 'half_open':
            breaker['success_count'] = breaker.get('success_count', 0) + 1
            success_threshold = breaker.get('success_threshold', 3)
            
            if breaker['success_count'] >= success_threshold:
                breaker['state'] = 'closed'
                breaker['failure_count'] = 0
                logger.info(f"Circuit breaker closed for {component_name} after {breaker['success_count']} successes")
        elif breaker.get('state') == 'closed':
            # Reset failure count on success
            breaker['failure_count'] = max(0, breaker.get('failure_count', 0) - 1)
    
    def _create_circuit_breaker_result(self, component_name: str, error_info: ErrorInfo) -> ProcessingResult:
        """Create result when circuit breaker is open"""
        return ProcessingResult(
            component_name=component_name,
            status=ProcessingStatus.FAILED,
            data={
                'circuit_breaker_open': True,
                'error_id': error_info.error_id,
                'degraded': True
            },
            error_message=f"Circuit breaker open for {component_name} - service temporarily unavailable"
        )
    
    def _update_component_health(self, component_name: str, error_info: ErrorInfo):
        """Update component health metrics"""
        if component_name not in self.component_health:
            self.component_health[component_name] = {
                'total_errors': 0,
                'recent_errors': [],
                'last_error_time': None,
                'error_rate': 0.0,
                'health_score': 1.0
            }
        
        health = self.component_health[component_name]
        health['total_errors'] += 1
        health['last_error_time'] = error_info.timestamp
        
        # Keep recent errors (last 10)
        health['recent_errors'].append({
            'error_id': error_info.error_id,
            'timestamp': error_info.timestamp,
            'severity': error_info.severity.value,
            'category': error_info.category.value
        })
        
        if len(health['recent_errors']) > 10:
            health['recent_errors'] = health['recent_errors'][-10:]
        
        # Calculate error rate (errors per hour)
        recent_errors_count = len(health['recent_errors'])
        if recent_errors_count > 0:
            time_span = (datetime.now() - health['recent_errors'][0]['timestamp']).total_seconds() / 3600
            health['error_rate'] = recent_errors_count / max(time_span, 0.1)
        
        # Calculate health score (0.0 to 1.0)
        health['health_score'] = max(0.0, 1.0 - (health['error_rate'] / 10.0))
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level"""
        log_message = (
            f"[{error_info.error_id}] {error_info.component_name}: "
            f"{error_info.error_message} "
            f"(Category: {error_info.category.value}, Severity: {error_info.severity.value})"
        )
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # Log stack trace for high severity errors
        if error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] and error_info.stack_trace:
            logger.error(f"Stack trace for {error_info.error_id}:\n{error_info.stack_trace}")
    
    def register_fallback_strategy(self, component_name: str, strategy: Callable):
        """Register fallback strategy for component"""
        self.fallback_strategies[component_name] = strategy
        logger.info(f"Registered fallback strategy for {component_name}")
    
    def set_retry_policy(self, component_name: str, policy: Dict[str, Any]):
        """Set retry policy for component"""
        self.retry_policies[component_name] = {**self.default_retry_policy, **policy}
        logger.info(f"Set retry policy for {component_name}: {self.retry_policies[component_name]}")
    
    def get_component_health(self, component_name: str = None) -> Dict[str, Any]:
        """Get component health information"""
        if component_name:
            return self.component_health.get(component_name, {})
        return self.component_health.copy()
    
    def get_error_history(self, component_name: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get error history"""
        errors = self.error_history
        
        if component_name:
            errors = [e for e in errors if e.component_name == component_name]
        
        # Convert to dict format and limit results
        return [
            {
                'error_id': e.error_id,
                'component_name': e.component_name,
                'error_type': e.error_type,
                'error_message': e.error_message,
                'severity': e.severity.value,
                'category': e.category.value,
                'timestamp': e.timestamp.isoformat(),
                'recovery_attempted': e.recovery_attempted,
                'recovery_successful': e.recovery_successful,
                'fallback_used': e.fallback_used,
                'retry_count': e.retry_count
            }
            for e in errors[-limit:]
        ]
    
    def reset_circuit_breaker(self, component_name: str) -> bool:
        """Manually reset circuit breaker for component"""
        if component_name in self.circuit_breakers:
            self.circuit_breakers[component_name].update({
                'state': 'closed',
                'failure_count': 0,
                'success_count': 0,
                'last_failure_time': None
            })
            logger.info(f"Circuit breaker reset for {component_name}")
            return True
        return False
    
    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        return {
            name: {
                'state': breaker['state'],
                'failure_count': breaker['failure_count'],
                'success_count': breaker.get('success_count', 0),
                'last_failure_time': breaker.get('last_failure_time'),
                'health_score': self.component_health.get(name, {}).get('health_score', 1.0)
            }
            for name, breaker in self.circuit_breakers.items()
        }


# Global error handler instance
global_error_handler = ComponentErrorHandler()


def get_error_handler() -> ComponentErrorHandler:
    """Get the global error handler instance"""
    return global_error_handler


async def handle_component_error(
    component_name: str,
    error: Exception,
    context: Dict[str, Any] = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    allow_retry: bool = True,
    allow_fallback: bool = True
) -> ProcessingResult:
    """
    Convenience function for handling component errors
    
    Args:
        component_name: Name of the component that failed
        error: The exception that occurred
        context: Additional context information
        severity: Error severity level
        category: Error category for classification
        allow_retry: Whether to attempt retry
        allow_fallback: Whether to use fallback mechanisms
        
    Returns:
        ProcessingResult with error handling outcome
    """
    return await global_error_handler.handle_error(
        component_name=component_name,
        error=error,
        context=context,
        severity=severity,
        category=category,
        allow_retry=allow_retry,
        allow_fallback=allow_fallback
    )