"""
LLM Integration - Provides advanced content understanding and description generation
"""
import logging
import time
import json
from typing import Dict, Any, List, Optional

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama not available - LLM integration disabled")

from .base import (
    BaseProcessor, AnalysisContext, ProcessingResult, ProcessingStatus,
    ComponentInterface, ContentType
)
from .error_handler import (
    handle_component_error, ErrorSeverity, ErrorCategory, get_error_handler
)

logger = logging.getLogger(__name__)


class LLMIntegration(BaseProcessor, ComponentInterface):
    """Provides advanced content understanding and generates human-like descriptions"""
    
    def __init__(self):
        super().__init__("llm_integration")
        self.ollama_client = None
        self.available_models = []
        self.model_capabilities = {}
        
        # Model preferences by content type and task
        self.model_preferences = {
            ContentType.MUSIC_VIDEO: {
                'primary': ['llama3.2-vision:11b', 'qwen2.5:14b', 'llama3.2:8b'],
                'vision': ['llama3.2-vision:11b', 'qwen2.5:14b'],
                'text': ['qwen2.5:14b', 'llama3.2:8b', 'deepseek-coder:6.7b']
            },
            ContentType.GAMING: {
                'primary': ['qwen2.5-coder:14b', 'deepseek-coder:6.7b', 'qwen2.5:14b'],
                'vision': ['llama3.2-vision:11b', 'qwen2.5:14b'],
                'text': ['qwen2.5-coder:14b', 'deepseek-coder:6.7b', 'qwen2.5:14b']
            },
            ContentType.EDUCATIONAL: {
                'primary': ['qwen2.5:14b', 'llama3.2:8b', 'qwen2.5-coder:14b'],
                'vision': ['llama3.2-vision:11b', 'qwen2.5:14b'],
                'text': ['qwen2.5:14b', 'llama3.2:8b', 'deepseek-coder:6.7b']
            },
            ContentType.NARRATIVE: {
                'primary': ['llama3.2:8b', 'qwen2.5:14b', 'llama4'],
                'vision': ['llama3.2-vision:11b', 'qwen2.5:14b'],
                'text': ['llama3.2:8b', 'qwen2.5:14b', 'llama4']
            },
            ContentType.DOCUMENTARY: {
                'primary': ['qwen2.5:14b', 'llama3.2:8b', 'llama4'],
                'vision': ['llama3.2-vision:11b', 'qwen2.5:14b'],
                'text': ['qwen2.5:14b', 'llama3.2:8b', 'llama4']
            },
            ContentType.GENERAL: {
                'primary': ['qwen2.5:14b', 'llama3.2:8b', 'llama3.2-vision:11b'],
                'vision': ['llama3.2-vision:11b', 'qwen2.5:14b'],
                'text': ['qwen2.5:14b', 'llama3.2:8b', 'deepseek-coder:6.7b']
            }
        }
        
        # Default fallback models
        self.default_models = {
            'primary': 'qwen2.5:14b',
            'vision': 'llama3.2-vision:11b',
            'text': 'llama3.2:8b'
        }
        
        # Prompt templates for different analysis types
        self.prompt_templates = self._initialize_prompt_templates()
        
    async def initialize(self) -> bool:
        """Initialize LLM integration with Ollama"""
        try:
            logger.info("Initializing LLM Integration")
            
            if not OLLAMA_AVAILABLE:
                logger.error("Ollama not available")
                self.status = ProcessingStatus.FAILED
                return False
            
            # Initialize Ollama client
            self.ollama_client = ollama.Client()
            
            # Check available models and their capabilities
            try:
                models_response = self.ollama_client.list()
                self.available_models = [model.model for model in models_response.models]
                logger.info(f"Available Ollama models: {self.available_models}")
                
                # Analyze model capabilities
                self._analyze_model_capabilities()
                
                # Ensure we have at least one working model
                if not self.available_models:
                    logger.warning("No Ollama models found - attempting to pull default model")
                    await self._ensure_default_models()
                
                # Validate model functionality
                if not await self._validate_model_functionality():
                    logger.error("No functional models available")
                    self.status = ProcessingStatus.FAILED
                    return False
                
            except Exception as e:
                logger.error(f"Failed to connect to Ollama: {e}")
                
                # Handle Ollama connection error
                await handle_component_error(
                    component_name=self.name,
                    error=e,
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.NETWORK,
                    allow_retry=True,
                    allow_fallback=False
                )
                
                self.status = ProcessingStatus.FAILED
                return False
            
            # Register error handling strategies
            error_handler = get_error_handler()
            error_handler.register_fallback_strategy(
                'llm_integration', 
                self._llm_fallback
            )
            
            # Set retry policy for LLM processing
            error_handler.set_retry_policy('llm_integration', {
                'max_retries': 3,
                'base_delay': 2.0,
                'max_delay': 15.0
            })
            
            self.is_initialized = True
            self.status = ProcessingStatus.COMPLETED
            logger.info(f"LLM Integration initialized with {len(self.available_models)} models")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM Integration: {e}")
            self.status = ProcessingStatus.FAILED
            return False
    
    def _initialize_prompt_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize prompt templates for different analysis types"""
        return {
            'description': {
                ContentType.MUSIC_VIDEO.value: """You are analyzing a music video segment. Focus on visual aesthetics, performer movements, choreography, and artistic elements that complement the music.

{segment_info}Video type: Music Video
Duration: {duration:.1f} seconds

Available data:
{context_data}

Provide a detailed, engaging description focusing on:
1. Visual aesthetics and cinematography
2. Performer movements and choreography  
3. Artistic elements and symbolism
4. How visuals complement the music
5. Scene transitions and effects

Description:""",
                
                ContentType.GAMING.value: """You are analyzing a gaming video segment. Emphasize gameplay mechanics, player actions, game environment, and interactive elements.

{segment_info}Video type: Gaming Content
Duration: {duration:.1f} seconds

Available data:
{context_data}

Provide a detailed description focusing on:
1. Gameplay mechanics and controls
2. Player actions and strategies
3. Game environment and level design
4. Interactive elements and UI
5. Progress and achievements

Description:""",
                
                ContentType.EDUCATIONAL.value: """You are analyzing an educational video segment. Highlight learning content, demonstrations, key information, and instructional elements.

{segment_info}Video type: Educational Content
Duration: {duration:.1f} seconds

Available data:
{context_data}

Provide a detailed description focusing on:
1. Learning objectives and key concepts
2. Demonstrations and examples
3. Visual aids and diagrams
4. Instructional techniques used
5. Important information conveyed

Description:""",
                
                ContentType.NARRATIVE.value: """You are analyzing a narrative video segment. Focus on character interactions, dialogue, story progression, and dramatic elements.

{segment_info}Video type: Narrative Content
Duration: {duration:.1f} seconds

Available data:
{context_data}

Provide a detailed description focusing on:
1. Character interactions and development
2. Dialogue and conversations
3. Plot progression and story beats
4. Emotional moments and tension
5. Setting and atmosphere

Description:""",
                
                ContentType.DOCUMENTARY.value: """You are analyzing a documentary segment. Balance factual information with visual storytelling and narrative flow.

{segment_info}Video type: Documentary
Duration: {duration:.1f} seconds

Available data:
{context_data}

Provide a detailed description focusing on:
1. Factual information presented
2. Visual evidence and examples
3. Narrative structure and flow
4. Expert commentary or interviews
5. Documentary techniques used

Description:""",
                
                'default': """You are analyzing a video segment. Provide comprehensive description of all visual and audio elements.

{segment_info}Video type: {content_type}
Duration: {duration:.1f} seconds

Available data:
{context_data}

Provide a detailed, engaging description focusing on:
1. Visual elements and scenes
2. Audio content and dialogue
3. Actions and movements
4. Context and meaning
5. Overall narrative flow

Description:"""
            },
            
            'scene_interpretation': {
                'default': """You are analyzing a video scene to understand its deeper meaning and context.

Video type: {content_type}
Duration: {duration:.1f} seconds

Available data:
{context_data}

Analyze this scene by examining:
1. Setting and environmental context
2. Character relationships and dynamics
3. Emotional undertones and mood
4. Cultural or contextual references
5. Implied meanings and subtext
6. Symbolic elements and metaphors

Provide insights that go beyond surface-level description to understand the scene's significance and deeper meaning.

Interpretation:"""
            },
            
            'narrative': {
                'default': """You are creating a comprehensive summary of a video analysis.

Video type: {content_type}
Total duration: {duration:.1f} seconds

Segment analyses:
{segments_data}

Create a coherent, engaging summary that:
1. Captures the overall narrative or theme
2. Highlights key moments and transitions
3. Integrates visual and audio elements seamlessly
4. Maintains chronological flow
5. Provides meaningful insights and context
6. Uses engaging, descriptive language

Format as a flowing narrative that conveys the video's content and significance.

Summary:"""
            }
        }
    
    def _analyze_model_capabilities(self) -> None:
        """Analyze capabilities of available models"""
        self.model_capabilities = {}
        
        for model in self.available_models:
            capabilities = {
                'vision': 'vision' in model.lower() or 'llava' in model.lower(),
                'coding': 'coder' in model.lower() or 'code' in model.lower(),
                'large': any(size in model.lower() for size in ['14b', '32b', '70b']),
                'fast': any(size in model.lower() for size in ['1b', '3b', '7b', '8b'])
            }
            self.model_capabilities[model] = capabilities
            
        logger.info(f"Model capabilities analyzed: {self.model_capabilities}")
    
    async def _ensure_default_models(self) -> None:
        """Ensure at least one default model is available"""
        default_priority = ['qwen2.5:14b', 'llama3.2:8b', 'qwen2.5:7b', 'llama3.2:3b']
        
        for model in default_priority:
            try:
                logger.info(f"Attempting to pull model: {model}")
                self.ollama_client.pull(model)
                self.available_models.append(model)
                logger.info(f"Successfully pulled model: {model}")
                break
            except Exception as e:
                logger.warning(f"Failed to pull model {model}: {e}")
                continue
    
    async def _validate_model_functionality(self) -> bool:
        """Validate that at least one model is functional"""
        if not self.available_models:
            return False
        
        # Test the first available model with a simple prompt
        test_model = self.available_models[0]
        try:
            response = self.ollama_client.generate(
                model=test_model,
                prompt="Test prompt. Respond with 'OK'.",
                options={'max_tokens': 10, 'temperature': 0.1}
            )
            
            if response and 'response' in response:
                logger.info(f"Model {test_model} validated successfully")
                return True
            else:
                logger.warning(f"Model {test_model} validation failed - no response")
                return False
                
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
        """Process content using LLM for advanced understanding"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing LLM analysis for session {context.session_id}")
            
            if not self.ollama_client:
                return ProcessingResult(
                    component_name=self.name,
                    status=ProcessingStatus.FAILED,
                    data={},
                    error_message="Ollama client not initialized"
                )
            
            # Determine the type of processing needed
            if hasattr(context, 'segments_data') and context.segments_data:
                # Summary generation
                result = await self.create_narrative(context)
            elif context.segment:
                # Segment description generation
                result = await self.generate_description(context)
            else:
                # Scene interpretation
                result = await self.interpret_scene(context)
            
            processing_time = time.time() - start_time
            logger.info(f"LLM processing completed in {processing_time:.2f}s")
            
            return ProcessingResult(
                component_name=self.name,
                status=ProcessingStatus.COMPLETED,
                data=result,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"LLM processing failed: {e}")
            return ProcessingResult(
                component_name=self.name,
                status=ProcessingStatus.FAILED,
                data={},
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    async def generate_description(self, context: AnalysisContext) -> Dict[str, Any]:
        """Generate coherent, contextual descriptions for video segments"""
        try:
            # Select appropriate model based on content type
            model = self._select_model(context.content_type, 'text')
            
            # Get optimized options for this content type and task
            options = self._get_model_options(context.content_type, 'description')
            
            # Create prompt based on content type and available data
            prompt = self._create_description_prompt(context)
            
            # Generate description
            response = self.ollama_client.generate(
                model=model,
                prompt=prompt,
                options=options
            )
            
            description = response.get('response', '').strip()
            
            # Validate and clean up response
            if not description or len(description) < 10:
                logger.warning("Generated description too short, using fallback")
                return self._fallback_description(context)
            
            return {
                'narrative': description,
                'model_used': model,
                'confidence': self._calculate_confidence(description, context),
                'prompt_type': 'description',
                'content_type': context.content_type.value
            }
            
        except Exception as e:
            logger.error(f"Description generation failed: {e}")
            return self._fallback_description(context)
    
    async def interpret_scene(self, context: AnalysisContext) -> Dict[str, Any]:
        """Interpret complex visual scenes and relationships"""
        try:
            # Prefer vision model for scene interpretation
            model = self._select_model(context.content_type, 'vision')
            
            # Get optimized options
            options = self._get_model_options(context.content_type, 'scene_interpretation')
            
            prompt = self._create_scene_interpretation_prompt(context)
            
            response = self.ollama_client.generate(
                model=model,
                prompt=prompt,
                options=options
            )
            
            interpretation = response.get('response', '').strip()
            
            # Validate response
            if not interpretation or len(interpretation) < 10:
                logger.warning("Generated interpretation too short, using fallback")
                interpretation = f"Scene analysis completed for {context.content_type.value} content"
            
            return {
                'scene_interpretation': interpretation,
                'model_used': model,
                'confidence': self._calculate_confidence(interpretation, context),
                'prompt_type': 'scene_interpretation',
                'content_type': context.content_type.value
            }
            
        except Exception as e:
            logger.error(f"Scene interpretation failed: {e}")
            return {
                'scene_interpretation': f'Scene analysis completed for {context.content_type.value} content', 
                'model_used': 'fallback',
                'confidence': 0.3,
                'prompt_type': 'fallback'
            }
    
    async def create_narrative(self, context: AnalysisContext) -> Dict[str, Any]:
        """Create coherent narrative from multiple segment analyses"""
        try:
            # Select model optimized for narrative generation
            model = self._select_model(context.content_type, 'text')
            
            # Get optimized options for narrative generation
            options = self._get_model_options(context.content_type, 'narrative')
            
            prompt = self._create_narrative_prompt(context)
            
            response = self.ollama_client.generate(
                model=model,
                prompt=prompt,
                options=options
            )
            
            narrative = response.get('response', '').strip()
            
            # Validate response
            if not narrative or len(narrative) < 20:
                logger.warning("Generated narrative too short, using fallback")
                return self._fallback_summary(context)
            
            return {
                'summary': narrative,
                'model_used': model,
                'confidence': self._calculate_confidence(narrative, context),
                'prompt_type': 'narrative',
                'content_type': context.content_type.value
            }
            
        except Exception as e:
            logger.error(f"Narrative creation failed: {e}")
            return self._fallback_summary(context)
    
    def _select_model(self, content_type: ContentType, task_type: str = 'primary') -> str:
        """Select appropriate model based on content type and task requirements"""
        
        # Get preferences for this content type
        preferences = self.model_preferences.get(content_type, self.model_preferences[ContentType.GENERAL])
        
        # Get preferred models for the task type
        preferred_models = preferences.get(task_type, preferences['primary'])
        
        # Find first available model from preferences
        for model in preferred_models:
            if model in self.available_models:
                logger.debug(f"Selected model {model} for {content_type.value} {task_type} task")
                return model
        
        # Fallback to any available model
        if self.available_models:
            fallback_model = self.available_models[0]
            logger.warning(f"Using fallback model {fallback_model} for {content_type.value} {task_type} task")
            return fallback_model
        
        # Final fallback to default
        default_model = self.default_models.get(task_type, self.default_models['primary'])
        logger.error(f"No models available, using default {default_model}")
        return default_model
    
    def _get_model_options(self, content_type: ContentType, task_type: str) -> Dict[str, Any]:
        """Get optimized options for model based on content type and task"""
        base_options = {
            'temperature': 0.7,
            'top_p': 0.9,
            'max_tokens': 500
        }
        
        # Adjust options based on content type
        if content_type == ContentType.EDUCATIONAL:
            base_options.update({
                'temperature': 0.6,  # More factual, less creative
                'max_tokens': 600    # Allow longer explanations
            })
        elif content_type == ContentType.NARRATIVE:
            base_options.update({
                'temperature': 0.8,  # More creative
                'max_tokens': 700    # Allow longer narratives
            })
        elif content_type == ContentType.GAMING:
            base_options.update({
                'temperature': 0.5,  # More precise for technical content
                'max_tokens': 400    # Concise descriptions
            })
        
        # Adjust options based on task type
        if task_type == 'narrative':
            base_options.update({
                'max_tokens': 800,   # Longer summaries
                'temperature': 0.8   # More creative flow
            })
        elif task_type == 'scene_interpretation':
            base_options.update({
                'max_tokens': 300,   # Focused interpretations
                'temperature': 0.6   # Balanced creativity and accuracy
            })
        
        return base_options
    
    def _create_description_prompt(self, context: AnalysisContext) -> str:
        """Create prompt for segment description generation using templates"""
        content_type_key = context.content_type.value
        
        # Get template for this content type or use default
        template = self.prompt_templates['description'].get(
            content_type_key, 
            self.prompt_templates['description']['default']
        )
        
        # Prepare segment info
        segment_info = ""
        if context.segment:
            segment_info = f"Time segment: {context.segment.start_time:.0f}-{context.segment.end_time:.0f} seconds\n"
        
        # Prepare context data
        context_data = self._format_context_data(context)
        
        # Format template
        return template.format(
            segment_info=segment_info,
            duration=context.metadata.duration,
            content_type=context.content_type.value,
            context_data=context_data
        )
    
    def _create_scene_interpretation_prompt(self, context: AnalysisContext) -> str:
        """Create prompt for scene interpretation using templates"""
        template = self.prompt_templates['scene_interpretation']['default']
        
        context_data = self._format_context_data(context)
        
        return template.format(
            content_type=context.content_type.value,
            duration=context.metadata.duration,
            context_data=context_data
        )
    
    def _create_narrative_prompt(self, context: AnalysisContext) -> str:
        """Create prompt for narrative summary generation using templates"""
        template = self.prompt_templates['narrative']['default']
        
        # Format segments data
        segments_text = ""
        if hasattr(context, 'segments_data') and context.segments_data:
            for i, segment in enumerate(context.segments_data, 1):
                segments_text += f"\nSegment {i} ({segment.get('time_range', f'{i*5-5}-{i*5} min')}):\n"
                if segment.get('visual'):
                    segments_text += f"Visual: {segment['visual']}\n"
                if segment.get('audio'):
                    segments_text += f"Audio: {segment['audio']}\n"
                if segment.get('narrative'):
                    segments_text += f"Analysis: {segment['narrative']}\n"
                segments_text += "\n"
        
        return template.format(
            content_type=context.content_type.value,
            duration=context.metadata.duration,
            segments_data=segments_text
        )
    
    def _format_context_data(self, context: AnalysisContext) -> str:
        """Format available context data for prompts"""
        data_parts = []
        
        # Add visual data if available
        if hasattr(context, 'visual_data') and context.visual_data:
            visual_info = []
            if context.visual_data.get('detected_objects'):
                objects = [obj.get('class_name', 'object') for obj in context.visual_data['detected_objects']]
                visual_info.append(f"Detected objects: {', '.join(set(objects))}")
            
            if context.visual_data.get('scene_description'):
                visual_info.append(f"Scene: {context.visual_data['scene_description']}")
            
            if visual_info:
                data_parts.append(f"Visual data: {'; '.join(visual_info)}")
        
        # Add audio data if available
        if hasattr(context, 'audio_data') and context.audio_data:
            audio_info = []
            if context.audio_data.get('transcription'):
                transcription = context.audio_data['transcription'][:200]  # Limit length
                audio_info.append(f"Transcription: {transcription}...")
            
            if context.audio_data.get('speakers'):
                speakers = len(context.audio_data['speakers'])
                audio_info.append(f"Speakers detected: {speakers}")
            
            if audio_info:
                data_parts.append(f"Audio data: {'; '.join(audio_info)}")
        
        # Add metadata
        if hasattr(context, 'metadata') and context.metadata:
            metadata_info = []
            if hasattr(context.metadata, 'resolution'):
                metadata_info.append(f"Resolution: {context.metadata.resolution}")
            if hasattr(context.metadata, 'fps'):
                metadata_info.append(f"FPS: {context.metadata.fps}")
            
            if metadata_info:
                data_parts.append(f"Video info: {'; '.join(metadata_info)}")
        
        return '\n'.join(data_parts) if data_parts else "No additional context data available"
    
    async def process(self, context: AnalysisContext) -> ProcessingResult:
        """Process content using LLM for advanced understanding"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing LLM analysis for session {context.session_id}")
            
            if not self.ollama_client:
                return ProcessingResult(
                    component_name=self.name,
                    status=ProcessingStatus.FAILED,
                    data={},
                    error_message="Ollama client not initialized"
                )
            
            # Determine the type of processing needed
            if hasattr(context, 'segments_data') and context.segments_data:
                # Summary generation
                result = await self.create_narrative(context)
            elif context.segment:
                # Segment description generation
                result = await self.generate_description(context)
            else:
                # Scene interpretation
                result = await self.interpret_scene(context)
            
            processing_time = time.time() - start_time
            logger.info(f"LLM processing completed in {processing_time:.2f}s")
            
            return ProcessingResult(
                component_name=self.name,
                status=ProcessingStatus.COMPLETED,
                data=result,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"LLM processing failed: {e}")
            return ProcessingResult(
                component_name=self.name,
                status=ProcessingStatus.FAILED,
                data={},
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _calculate_confidence(self, generated_text: str, context: AnalysisContext) -> float:
        """Calculate confidence score for generated text"""
        base_confidence = 0.7
        
        # Adjust based on text length and quality indicators
        if len(generated_text) < 20:
            base_confidence -= 0.3
        elif len(generated_text) > 100:
            base_confidence += 0.1
        
        # Adjust based on content specificity
        content_indicators = {
            ContentType.MUSIC_VIDEO: ['music', 'visual', 'performer', 'aesthetic'],
            ContentType.GAMING: ['game', 'player', 'gameplay', 'level'],
            ContentType.EDUCATIONAL: ['learn', 'explain', 'demonstrate', 'concept'],
            ContentType.NARRATIVE: ['character', 'story', 'dialogue', 'scene'],
            ContentType.DOCUMENTARY: ['information', 'fact', 'evidence', 'documentary']
        }
        
        indicators = content_indicators.get(context.content_type, [])
        matches = sum(1 for indicator in indicators if indicator.lower() in generated_text.lower())
        
        if matches > 0:
            base_confidence += min(0.2, matches * 0.05)
        
        # Ensure confidence is within valid range
        return max(0.1, min(0.95, base_confidence))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about available models and their capabilities"""
        return {
            'available_models': self.available_models.copy(),
            'model_capabilities': self.model_capabilities.copy(),
            'model_preferences': {
                content_type.value: prefs for content_type, prefs in self.model_preferences.items()
            },
            'default_models': self.default_models.copy(),
            'ollama_available': OLLAMA_AVAILABLE,
            'initialized': self.is_initialized
        }
    
    async def generate_contextual_description(self, context: AnalysisContext, focus_areas: List[str] = None) -> Dict[str, Any]:
        """Generate description with specific focus areas for enhanced context"""
        try:
            # Select appropriate model
            model = self._select_model(context.content_type, 'text')
            
            # Create enhanced prompt with focus areas
            prompt = self._create_focused_description_prompt(context, focus_areas)
            
            # Get optimized options
            options = self._get_model_options(context.content_type, 'description')
            
            response = self.ollama_client.generate(
                model=model,
                prompt=prompt,
                options=options
            )
            
            description = response.get('response', '').strip()
            
            if not description or len(description) < 10:
                return self._fallback_description(context)
            
            return {
                'narrative': description,
                'model_used': model,
                'confidence': self._calculate_confidence(description, context),
                'prompt_type': 'contextual_description',
                'focus_areas': focus_areas or [],
                'content_type': context.content_type.value
            }
            
        except Exception as e:
            logger.error(f"Contextual description generation failed: {e}")
            return self._fallback_description(context)
    
    async def interpret_cultural_context(self, context: AnalysisContext) -> Dict[str, Any]:
        """Interpret cultural references and contextual meanings"""
        try:
            # Use the most capable model for cultural interpretation
            model = self._select_model(context.content_type, 'primary')
            
            prompt = self._create_cultural_interpretation_prompt(context)
            
            options = {
                'temperature': 0.6,
                'max_tokens': 400,
                'top_p': 0.8
            }
            
            response = self.ollama_client.generate(
                model=model,
                prompt=prompt,
                options=options
            )
            
            interpretation = response.get('response', '').strip()
            
            return {
                'cultural_interpretation': interpretation,
                'model_used': model,
                'confidence': self._calculate_confidence(interpretation, context),
                'prompt_type': 'cultural_context',
                'content_type': context.content_type.value
            }
            
        except Exception as e:
            logger.error(f"Cultural interpretation failed: {e}")
            return {
                'cultural_interpretation': 'Cultural context analysis completed',
                'model_used': 'fallback',
                'confidence': 0.3,
                'prompt_type': 'fallback'
            }
    
    async def analyze_implied_meanings(self, context: AnalysisContext) -> Dict[str, Any]:
        """Analyze implied meanings and subtext in the content"""
        try:
            model = self._select_model(context.content_type, 'primary')
            
            prompt = self._create_subtext_analysis_prompt(context)
            
            options = {
                'temperature': 0.7,
                'max_tokens': 350,
                'top_p': 0.85
            }
            
            response = self.ollama_client.generate(
                model=model,
                prompt=prompt,
                options=options
            )
            
            analysis = response.get('response', '').strip()
            
            return {
                'subtext_analysis': analysis,
                'model_used': model,
                'confidence': self._calculate_confidence(analysis, context),
                'prompt_type': 'subtext_analysis',
                'content_type': context.content_type.value
            }
            
        except Exception as e:
            logger.error(f"Subtext analysis failed: {e}")
            return {
                'subtext_analysis': 'Subtext analysis completed',
                'model_used': 'fallback',
                'confidence': 0.3,
                'prompt_type': 'fallback'
            }
    def _create_focused_description_prompt(self, context: AnalysisContext, focus_areas: List[str] = None) -> str:
        """Create prompt with specific focus areas"""
        base_template = self._create_description_prompt(context)
        
        if focus_areas:
            focus_text = f"\nPay special attention to these aspects: {', '.join(focus_areas)}"
            # Insert focus areas before the final "Description:" line
            base_template = base_template.replace("Description:", f"{focus_text}\n\nDescription:")
        
        return base_template
    
    def _create_cultural_interpretation_prompt(self, context: AnalysisContext) -> str:
        """Create prompt for cultural context interpretation"""
        context_data = self._format_context_data(context)
        
        return f"""You are analyzing video content to identify cultural references, contextual meanings, and cultural significance.

Video type: {context.content_type.value}
Duration: {context.metadata.duration:.1f} seconds

Available data:
{context_data}

Analyze the cultural context by examining:
1. Cultural references and allusions
2. Historical or social context
3. Symbolic meanings and metaphors
4. Genre conventions and expectations
5. Target audience and cultural assumptions
6. Regional or demographic specificity

Provide insights into the cultural layers and contextual meanings that might not be immediately obvious to all viewers.

Cultural Interpretation:"""
    
    def _create_subtext_analysis_prompt(self, context: AnalysisContext) -> str:
        """Create prompt for subtext and implied meaning analysis"""
        context_data = self._format_context_data(context)
        
        return f"""You are analyzing video content to identify implied meanings, subtext, and deeper significance beyond the surface content.

Video type: {context.content_type.value}
Duration: {context.metadata.duration:.1f} seconds

Available data:
{context_data}

Analyze the subtext by examining:
1. What is implied but not explicitly stated
2. Underlying themes and messages
3. Emotional undertones and mood
4. Power dynamics and relationships
5. Irony, satire, or hidden criticism
6. Symbolic representations and metaphors

Focus on meanings that require interpretation and understanding beyond literal content.

Subtext Analysis:"""
    
    def update_model_preferences(self, content_type: ContentType, task_type: str, models: List[str]) -> bool:
        """Update model preferences for specific content type and task"""
        try:
            if content_type not in self.model_preferences:
                self.model_preferences[content_type] = {}
            
            # Validate that models are available
            valid_models = [model for model in models if model in self.available_models]
            
            if valid_models:
                self.model_preferences[content_type][task_type] = valid_models
                logger.info(f"Updated {content_type.value} {task_type} preferences: {valid_models}")
                return True
            else:
                logger.warning(f"No valid models provided for {content_type.value} {task_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update model preferences: {e}")
            return False
    
    def _fallback_description(self, context: AnalysisContext) -> Dict[str, Any]:
        fallback_descriptions = {
            ContentType.MUSIC_VIDEO: "Music video with visual and audio elements",
            ContentType.GAMING: "Gaming content with player interactions",
            ContentType.EDUCATIONAL: "Educational content with instructional material",
            ContentType.NARRATIVE: "Narrative content with story elements",
            ContentType.DOCUMENTARY: "Documentary content with informational material",
            ContentType.GENERAL: "Video content with visual and audio elements"
        }
        
        description = fallback_descriptions.get(context.content_type, "Video content analyzed")
        
        if context.segment:
            description += f" (segment {context.segment.start_time:.0f}-{context.segment.end_time:.0f}s)"
        
        return {
            'narrative': description,
            'model_used': 'fallback',
            'confidence': 0.3,
            'prompt_type': 'fallback'
        }
    
    def _fallback_summary(self, context: AnalysisContext) -> Dict[str, Any]:
        """Provide fallback summary when LLM fails"""
        summary = f"Video analysis completed for {context.content_type.value} content"
        
        if hasattr(context, 'segments_data') and context.segments_data:
            segment_count = len(context.segments_data)
            summary += f" across {segment_count} segments"
        
        summary += f". Total duration: {context.metadata.duration:.1f} seconds."
        
        return {
            'summary': summary,
            'model_used': 'fallback',
            'confidence': 0.3,
            'prompt_type': 'fallback'
        }
    
    def get_supported_operations(self) -> List[str]:
        """Return list of supported operations"""
        operations = []
        
        if OLLAMA_AVAILABLE and self.available_models:
            operations.extend([
                'description_generation',
                'scene_interpretation',
                'narrative_creation',
                'content_understanding'
            ])
        
        return operations
    
    async def health_check(self) -> bool:
        """Check if component is healthy and ready"""
        if not self.is_initialized or not self.ollama_client:
            return False
        
        try:
            # Test connection with a simple request
            response = self.ollama_client.list()
            return len(response.models) > 0
        except Exception as e:
            logger.warning(f"LLM health check failed: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Clean up LLM integration resources"""
        logger.info("Cleaning up LLM Integration")
        self.ollama_client = None
        self.available_models.clear()
    
    async def _llm_fallback(self, context: Dict[str, Any]) -> ProcessingResult:
        """Fallback strategy for LLM processing failures"""
        try:
            logger.info("Executing LLM processing fallback")
            
            operation = context.get('operation', 'unknown')
            
            if operation == 'llm_processing':
                # Determine what type of fallback to provide
                if context.get('has_segments_data'):
                    # Summary generation fallback
                    fallback_data = {
                        'summary': 'Video analysis completed with basic processing due to LLM unavailability',
                        'model_used': 'fallback',
                        'confidence': 0.3,
                        'fallback_used': True
                    }
                elif context.get('has_segment'):
                    # Segment description fallback
                    fallback_data = {
                        'narrative': 'Segment analysis completed with basic processing',
                        'model_used': 'fallback',
                        'confidence': 0.3,
                        'fallback_used': True
                    }
                else:
                    # Scene interpretation fallback
                    fallback_data = {
                        'scene_interpretation': 'Scene analysis completed with basic processing',
                        'model_used': 'fallback',
                        'confidence': 0.3,
                        'fallback_used': True
                    }
            else:
                # Generic fallback
                fallback_data = {
                    'narrative': 'Content analysis completed with basic processing',
                    'model_used': 'fallback',
                    'confidence': 0.3,
                    'fallback_used': True
                }
            
            return ProcessingResult(
                component_name=self.name,
                status=ProcessingStatus.COMPLETED,
                data=fallback_data
            )
            
        except Exception as e:
            logger.error(f"LLM fallback failed: {e}")
            return ProcessingResult(
                component_name=self.name,
                status=ProcessingStatus.FAILED,
                data={},
                error_message=f"LLM fallback failed: {str(e)}"
            )
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        return self.available_models.copy()
    
    def set_default_model(self, model_name: str, task_type: str = 'primary') -> bool:
        """Set default model for specific task type if available"""
        if model_name in self.available_models:
            self.default_models[task_type] = model_name
            logger.info(f"Default {task_type} model set to: {model_name}")
            return True
        else:
            logger.warning(f"Model {model_name} not available")
            return False