"""
Visual Extractor - Processes video frames for visual analysis
"""
import cv2
import numpy as np
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import os

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO not available - object detection will be limited")

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("Tesseract not available - text extraction disabled")

from .base import (
    BaseProcessor, AnalysisContext, ProcessingResult, ProcessingStatus,
    ComponentInterface
)
from .error_handler import (
    handle_component_error, ErrorSeverity, ErrorCategory, get_error_handler
)

logger = logging.getLogger(__name__)


class VisualExtractor(BaseProcessor, ComponentInterface):
    """Processes video frames to extract visual information and understand scenes"""
    
    def __init__(self):
        super().__init__("visual_extractor")
        self.yolo_model = None
        self.frame_extraction_interval = 1.0  # Extract frame every second
        
    async def initialize(self) -> bool:
        """Initialize visual processing models"""
        try:
            logger.info("Initializing Visual Extractor")
            
            # Initialize YOLO model if available
            if YOLO_AVAILABLE:
                try:
                    self.yolo_model = YOLO('yolov8n.pt')  # Use nano model for speed
                    logger.info("YOLO model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load YOLO model: {e}")
                    self.yolo_model = None
            
            # Register error handling strategies
            error_handler = get_error_handler()
            error_handler.register_fallback_strategy(
                'visual_extractor', 
                self._visual_fallback
            )
            
            # Set retry policy for visual processing
            error_handler.set_retry_policy('visual_extractor', {
                'max_retries': 2,
                'base_delay': 1.0,
                'max_delay': 5.0
            })
            
            self.is_initialized = True
            self.status = ProcessingStatus.COMPLETED
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Visual Extractor: {e}")
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
        """Process video segment for visual analysis"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing visual content for segment {context.segment.segment_id if context.segment else 'full video'}")
            
            # Extract frames from video segment
            frames = await self.extract_frames(
                context.video_path, 
                context.segment.start_time if context.segment else 0,
                context.segment.end_time if context.segment else context.metadata.duration
            )
            
            if not frames:
                return ProcessingResult(
                    component_name=self.name,
                    status=ProcessingStatus.FAILED,
                    data={},
                    error_message="No frames extracted"
                )
            
            # Analyze frames
            analysis_results = {
                'objects': [],
                'scenes': [],
                'text_elements': [],
                'movement_analysis': {},
                'visual_style': {},
                'description': '',
                'confidence': 0.0
            }
            
            # Object detection
            if self.yolo_model:
                objects = await self.detect_objects(frames)
                analysis_results['objects'] = objects
            
            # Scene analysis
            scenes = await self.analyze_scenes(frames)
            analysis_results['scenes'] = scenes
            
            # Text extraction
            if OCR_AVAILABLE:
                text_elements = await self.extract_text(frames)
                analysis_results['text_elements'] = text_elements
            
            # Movement analysis
            if len(frames) > 1:
                movement = await self.analyze_movement(frames)
                analysis_results['movement_analysis'] = movement
            
            # Visual style analysis
            style = await self.analyze_visual_style(frames)
            analysis_results['visual_style'] = style
            
            # Generate description
            description = await self.generate_description(analysis_results, context)
            analysis_results['description'] = description
            
            # Calculate confidence
            analysis_results['confidence'] = self._calculate_confidence(analysis_results)
            
            processing_time = time.time() - start_time
            logger.info(f"Visual processing completed in {processing_time:.2f}s")
            
            return ProcessingResult(
                component_name=self.name,
                status=ProcessingStatus.COMPLETED,
                data=analysis_results,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Visual processing failed: {e}")
            
            # Handle visual processing error with comprehensive error handling
            error_result = await handle_component_error(
                component_name=self.name,
                error=e,
                context={
                    'segment_id': context.segment.segment_id if context.segment else 'full_video',
                    'video_path': context.video_path,
                    'operation': 'visual_processing'
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
    
    async def extract_frames(self, video_path: str, start_time: float = 0, end_time: float = None) -> List[np.ndarray]:
        """Extract frames from video at specified intervals"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            if end_time is None:
                end_time = duration
            
            frames = []
            current_time = start_time
            
            while current_time < end_time:
                # Seek to specific time
                frame_number = int(current_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    break
                
                current_time += self.frame_extraction_interval
            
            cap.release()
            logger.info(f"Extracted {len(frames)} frames from {start_time}s to {end_time}s")
            return frames
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return []
    
    async def detect_objects(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Detect objects in frames using YOLO"""
        if not self.yolo_model:
            return []
        
        try:
            all_objects = []
            
            for i, frame in enumerate(frames):
                results = self.yolo_model(frame, verbose=False)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Extract box information
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = self.yolo_model.names[class_id]
                            
                            all_objects.append({
                                'class_name': class_name,
                                'confidence': confidence,
                                'bounding_box': [int(x1), int(y1), int(x2), int(y2)],
                                'frame_index': i,
                                'timestamp': i * self.frame_extraction_interval
                            })
            
            logger.info(f"Detected {len(all_objects)} objects across {len(frames)} frames")
            return all_objects
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    async def analyze_scenes(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Analyze scene characteristics and changes"""
        try:
            scenes = []
            prev_frame = None
            scene_change_threshold = 0.3
            
            for i, frame in enumerate(frames):
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                scene_info = {
                    'frame_index': i,
                    'timestamp': i * self.frame_extraction_interval,
                    'brightness': float(np.mean(gray)),
                    'contrast': float(np.std(gray)),
                    'is_scene_change': False
                }
                
                # Detect scene changes
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray)
                    mean_diff = np.mean(diff) / 255.0
                    
                    if mean_diff > scene_change_threshold:
                        scene_info['is_scene_change'] = True
                        scene_info['change_intensity'] = float(mean_diff)
                
                # Analyze color distribution
                color_analysis = self._analyze_colors(frame)
                scene_info.update(color_analysis)
                
                scenes.append(scene_info)
                prev_frame = gray
            
            logger.info(f"Analyzed {len(scenes)} scenes")
            return scenes
            
        except Exception as e:
            logger.error(f"Scene analysis failed: {e}")
            return []
    
    async def extract_text(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Extract text from frames using OCR"""
        if not OCR_AVAILABLE:
            return []
        
        try:
            text_elements = []
            
            for i, frame in enumerate(frames):
                try:
                    # Use pytesseract for OCR
                    data = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)
                    
                    for j, text in enumerate(data['text']):
                        if text.strip():
                            confidence = float(data['conf'][j]) / 100.0
                            if confidence > 0.5:  # Filter low confidence detections
                                x, y, w, h = data['left'][j], data['top'][j], data['width'][j], data['height'][j]
                                text_elements.append({
                                    'text': text.strip(),
                                    'confidence': confidence,
                                    'bounding_box': [x, y, x+w, y+h],
                                    'frame_index': i,
                                    'timestamp': i * self.frame_extraction_interval
                                })
                
                except Exception as e:
                    logger.warning(f"OCR failed for frame {i}: {e}")
            
            logger.info(f"Extracted {len(text_elements)} text elements")
            return text_elements
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return []
    
    async def analyze_movement(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze movement between frames with enhanced motion tracking"""
        try:
            if len(frames) < 2:
                return {'movement_detected': False}
            
            movement_data = {
                'movement_detected': False,
                'movement_intensity': 0.0,
                'movement_areas': [],
                'motion_vectors': [],
                'camera_movement': 'static',
                'object_movement': 'none'
            }
            
            # Convert frames to grayscale
            gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
            
            total_movement = 0.0
            movement_count = 0
            motion_vectors = []
            
            for i in range(1, len(gray_frames)):
                prev_frame = gray_frames[i-1]
                curr_frame = gray_frames[i]
                
                # Calculate optical flow for motion vectors
                try:
                    # Use Lucas-Kanade optical flow
                    corners = cv2.goodFeaturesToTrack(prev_frame, maxCorners=100, 
                                                    qualityLevel=0.01, minDistance=10)
                    
                    if corners is not None and len(corners) > 0:
                        # Calculate optical flow
                        next_corners, status, error = cv2.calcOpticalFlowPyrLK(
                            prev_frame, curr_frame, corners, None)
                        
                        # Filter good points
                        good_new = next_corners[status == 1]
                        good_old = corners[status == 1]
                        
                        if len(good_new) > 0:
                            # Calculate motion vectors
                            vectors = good_new - good_old
                            avg_motion = np.mean(np.linalg.norm(vectors, axis=1))
                            motion_vectors.append(avg_motion)
                            
                            # Analyze motion patterns
                            if avg_motion > 5.0:
                                movement_data['movement_detected'] = True
                                
                                # Determine movement type
                                horizontal_motion = np.mean(np.abs(vectors[:, 0]))
                                vertical_motion = np.mean(np.abs(vectors[:, 1]))
                                
                                if horizontal_motion > vertical_motion * 2:
                                    movement_data['camera_movement'] = 'horizontal_pan'
                                elif vertical_motion > horizontal_motion * 2:
                                    movement_data['camera_movement'] = 'vertical_tilt'
                                elif avg_motion > 10.0:
                                    movement_data['camera_movement'] = 'complex'
                                else:
                                    movement_data['object_movement'] = 'moderate'
                
                except Exception as e:
                    logger.warning(f"Optical flow calculation failed for frame {i}: {e}")
                
                # Fallback: Frame difference analysis
                diff = cv2.absdiff(prev_frame, curr_frame)
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                
                # Find movement areas
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                frame_movement_areas = []
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:  # Filter small movements
                        x, y, w, h = cv2.boundingRect(contour)
                        frame_movement_areas.append({
                            'bounding_box': [x, y, x+w, y+h],
                            'area': area,
                            'timestamp': i * self.frame_extraction_interval
                        })
                
                movement_data['movement_areas'].extend(frame_movement_areas)
                
                # Calculate movement intensity
                movement_pixels = np.sum(thresh > 0)
                total_pixels = thresh.shape[0] * thresh.shape[1]
                movement_ratio = movement_pixels / total_pixels
                
                total_movement += movement_ratio
                movement_count += 1
                
                if movement_ratio > 0.1:
                    movement_data['movement_detected'] = True
            
            # Calculate overall movement metrics
            if movement_count > 0:
                movement_data['movement_intensity'] = total_movement / movement_count
            
            if motion_vectors:
                movement_data['motion_vectors'] = motion_vectors
                avg_motion_magnitude = np.mean(motion_vectors)
                
                if avg_motion_magnitude > 15.0:
                    movement_data['object_movement'] = 'high'
                elif avg_motion_magnitude > 5.0:
                    movement_data['object_movement'] = 'moderate'
                else:
                    movement_data['object_movement'] = 'low'
            
            return movement_data
            
        except Exception as e:
            logger.error(f"Movement analysis failed: {e}")
            return {'movement_detected': False}
    
    async def analyze_visual_style(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze visual style and aesthetics with comprehensive detection"""
        try:
            if not frames:
                return {}
            
            # Sample frames for analysis (more comprehensive sampling)
            sample_frames = frames[::max(1, len(frames)//10)]  # Sample up to 10 frames
            
            style_data = {
                'dominant_colors': [],
                'color_palette': [],
                'lighting': 'unknown',
                'composition': 'unknown',
                'color_temperature': 'neutral',
                'saturation_level': 'balanced',
                'contrast_level': 'medium',
                'visual_complexity': 'medium',
                'aesthetic_score': 0.0
            }
            
            # Analyze colors and lighting across frames
            all_colors = []
            brightness_values = []
            contrast_values = []
            saturation_values = []
            color_temperatures = []
            complexity_scores = []
            
            for frame in sample_frames:
                # Color analysis
                colors = self._extract_dominant_colors(frame, k=8)
                all_colors.extend(colors)
                
                # Convert to different color spaces for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                
                # Brightness analysis
                brightness = np.mean(gray)
                brightness_values.append(brightness)
                
                # Contrast analysis
                contrast = np.std(gray)
                contrast_values.append(contrast)
                
                # Saturation analysis
                saturation = np.mean(hsv[:, :, 1])
                saturation_values.append(saturation)
                
                # Color temperature analysis (simplified)
                b, g, r = cv2.split(frame)
                avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
                
                # Simple color temperature estimation
                if avg_b > avg_r * 1.1:
                    color_temperatures.append('cool')
                elif avg_r > avg_b * 1.1:
                    color_temperatures.append('warm')
                else:
                    color_temperatures.append('neutral')
                
                # Visual complexity (edge density)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                complexity_scores.append(edge_density)
            
            # Aggregate lighting analysis
            avg_brightness = np.mean(brightness_values)
            brightness_std = np.std(brightness_values)
            
            if avg_brightness < 60:
                style_data['lighting'] = 'very_dark'
            elif avg_brightness < 100:
                style_data['lighting'] = 'dark'
            elif avg_brightness < 140:
                style_data['lighting'] = 'dim'
            elif avg_brightness < 180:
                style_data['lighting'] = 'balanced'
            elif avg_brightness < 220:
                style_data['lighting'] = 'bright'
            else:
                style_data['lighting'] = 'very_bright'
            
            # Contrast analysis
            avg_contrast = np.mean(contrast_values)
            if avg_contrast < 30:
                style_data['contrast_level'] = 'low'
            elif avg_contrast < 60:
                style_data['contrast_level'] = 'medium'
            else:
                style_data['contrast_level'] = 'high'
            
            # Saturation analysis
            avg_saturation = np.mean(saturation_values)
            if avg_saturation < 80:
                style_data['saturation_level'] = 'desaturated'
            elif avg_saturation < 150:
                style_data['saturation_level'] = 'balanced'
            else:
                style_data['saturation_level'] = 'vibrant'
            
            # Color temperature
            temp_counts = {temp: color_temperatures.count(temp) for temp in set(color_temperatures)}
            style_data['color_temperature'] = max(temp_counts, key=temp_counts.get) if temp_counts else 'neutral'
            
            # Visual complexity
            avg_complexity = np.mean(complexity_scores)
            if avg_complexity < 0.05:
                style_data['visual_complexity'] = 'simple'
            elif avg_complexity < 0.15:
                style_data['visual_complexity'] = 'medium'
            else:
                style_data['visual_complexity'] = 'complex'
            
            # Color palette analysis
            if all_colors:
                # Cluster colors to find dominant palette
                unique_colors = self._cluster_colors(all_colors, max_colors=12)
                style_data['color_palette'] = unique_colors
                style_data['dominant_colors'] = unique_colors[:5]  # Top 5 colors
            
            # Composition analysis (rule of thirds, symmetry, etc.)
            composition_analysis = self._analyze_composition(sample_frames)
            style_data['composition'] = composition_analysis
            
            # Calculate aesthetic score (simplified)
            aesthetic_score = self._calculate_aesthetic_score(
                avg_brightness, avg_contrast, avg_saturation, avg_complexity, brightness_std
            )
            style_data['aesthetic_score'] = aesthetic_score
            
            return style_data
            
        except Exception as e:
            logger.error(f"Visual style analysis failed: {e}")
            return {}
    
    def _analyze_colors(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution in frame"""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Calculate color statistics
            hue_mean = float(np.mean(hsv[:, :, 0]))
            saturation_mean = float(np.mean(hsv[:, :, 1]))
            value_mean = float(np.mean(hsv[:, :, 2]))
            
            return {
                'hue_mean': hue_mean,
                'saturation_mean': saturation_mean,
                'value_mean': value_mean,
                'color_richness': saturation_mean / 255.0
            }
        except Exception:
            return {}
    
    def _extract_dominant_colors(self, frame: np.ndarray, k: int = 5) -> List[List[int]]:
        """Extract dominant colors from frame using k-means clustering"""
        try:
            # Reshape frame to be a list of pixels
            data = frame.reshape((-1, 3))
            data = np.float32(data)
            
            # Apply k-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert centers to integers
            centers = np.uint8(centers)
            return centers.tolist()
            
        except Exception:
            return []
    
    def _cluster_colors(self, colors: List[List[int]], max_colors: int = 10) -> List[List[int]]:
        """Cluster similar colors together to create a refined palette"""
        try:
            if not colors or len(colors) <= max_colors:
                return colors
            
            # Convert to numpy array
            color_array = np.array(colors, dtype=np.float32)
            
            # Apply k-means clustering to group similar colors
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(color_array, max_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert centers back to integers
            centers = np.uint8(centers)
            return centers.tolist()
            
        except Exception:
            # Fallback: return unique colors up to max_colors
            unique_colors = []
            for color in colors:
                if color not in unique_colors:
                    unique_colors.append(color)
                if len(unique_colors) >= max_colors:
                    break
            return unique_colors
    
    def _analyze_composition(self, frames: List[np.ndarray]) -> str:
        """Analyze composition characteristics of frames"""
        try:
            if not frames:
                return 'unknown'
            
            composition_scores = {
                'centered': 0,
                'rule_of_thirds': 0,
                'symmetrical': 0,
                'dynamic': 0
            }
            
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape
                
                # Analyze center weighting
                center_region = gray[h//3:2*h//3, w//3:2*w//3]
                center_intensity = np.mean(center_region)
                edge_intensity = (np.mean(gray[:h//3, :]) + np.mean(gray[2*h//3:, :]) + 
                                np.mean(gray[:, :w//3]) + np.mean(gray[:, 2*w//3:])) / 4
                
                if center_intensity > edge_intensity * 1.2:
                    composition_scores['centered'] += 1
                
                # Simple rule of thirds analysis
                # Check if interesting features are near rule of thirds lines
                thirds_h = [h//3, 2*h//3]
                thirds_w = [w//3, 2*w//3]
                
                # Use edge detection to find interesting features
                edges = cv2.Canny(gray, 50, 150)
                
                rule_of_thirds_score = 0
                for th in thirds_h:
                    rule_of_thirds_score += np.sum(edges[max(0, th-10):min(h, th+10), :])
                for tw in thirds_w:
                    rule_of_thirds_score += np.sum(edges[:, max(0, tw-10):min(w, tw+10)])
                
                total_edges = np.sum(edges)
                if total_edges > 0 and rule_of_thirds_score / total_edges > 0.3:
                    composition_scores['rule_of_thirds'] += 1
                
                # Symmetry analysis
                left_half = gray[:, :w//2]
                right_half = cv2.flip(gray[:, w//2:], 1)
                
                if right_half.shape == left_half.shape:
                    symmetry_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
                    if symmetry_diff < 30:  # Threshold for symmetry
                        composition_scores['symmetrical'] += 1
                
                # Dynamic composition (high edge density in multiple regions)
                edge_density = np.sum(edges) / (h * w)
                if edge_density > 0.1:
                    composition_scores['dynamic'] += 1
            
            # Return the dominant composition style
            max_score = max(composition_scores.values())
            if max_score == 0:
                return 'balanced'
            
            return max(composition_scores, key=composition_scores.get)
            
        except Exception as e:
            logger.warning(f"Composition analysis failed: {e}")
            return 'unknown'
    
    def _calculate_aesthetic_score(self, brightness: float, contrast: float, 
                                 saturation: float, complexity: float, brightness_std: float) -> float:
        """Calculate a simplified aesthetic score based on visual properties"""
        try:
            score = 0.0
            
            # Brightness score (prefer balanced lighting)
            if 100 <= brightness <= 180:
                score += 0.2
            elif 80 <= brightness <= 220:
                score += 0.1
            
            # Contrast score (prefer good contrast)
            if 40 <= contrast <= 80:
                score += 0.2
            elif 30 <= contrast <= 100:
                score += 0.1
            
            # Saturation score (prefer balanced saturation)
            if 80 <= saturation <= 180:
                score += 0.2
            elif 60 <= saturation <= 200:
                score += 0.1
            
            # Complexity score (prefer moderate complexity)
            if 0.05 <= complexity <= 0.2:
                score += 0.2
            elif 0.02 <= complexity <= 0.3:
                score += 0.1
            
            # Consistency score (prefer consistent lighting)
            if brightness_std < 20:
                score += 0.2
            elif brightness_std < 40:
                score += 0.1
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception:
            return 0.5  # Default neutral score
    
    async def generate_description(self, analysis_results: Dict[str, Any], context: AnalysisContext) -> str:
        """Generate comprehensive textual description of visual content with gaming-specific enhancements"""
        try:
            description_parts = []
            content_type = context.content_type if hasattr(context, 'content_type') else None
            is_gaming = content_type and content_type.value == 'gaming'
            
            # Describe objects with gaming-specific focus
            objects = analysis_results.get('objects', [])
            if objects:
                object_counts = {}
                for obj in objects:
                    name = obj['class_name']
                    object_counts[name] = object_counts.get(name, 0) + 1
                
                if is_gaming:
                    # Gaming-specific object descriptions
                    gaming_objects = ['person', 'car', 'truck', 'motorcycle', 'airplane', 'boat', 'weapon', 'building']
                    game_elements = {k: v for k, v in object_counts.items() if k in gaming_objects}
                    other_elements = {k: v for k, v in object_counts.items() if k not in gaming_objects}
                    
                    if game_elements:
                        game_desc = ", ".join([f"{count} {name}{'s' if count > 1 else ''}" 
                                             for name, count in game_elements.items()])
                        description_parts.append(f"Game elements: {game_desc}")
                    
                    if other_elements:
                        other_desc = ", ".join([f"{count} {name}{'s' if count > 1 else ''}" 
                                              for name, count in other_elements.items()])
                        description_parts.append(f"Other objects: {other_desc}")
                else:
                    object_desc = ", ".join([f"{count} {name}{'s' if count > 1 else ''}" 
                                           for name, count in object_counts.items()])
                    description_parts.append(f"Objects detected: {object_desc}")
            
            # Describe scenes and lighting with gaming context
            scenes = analysis_results.get('scenes', [])
            if scenes:
                scene_changes = sum(1 for scene in scenes if scene.get('is_scene_change', False))
                if scene_changes > 0:
                    if is_gaming:
                        if scene_changes > 5:
                            description_parts.append("Frequent scene transitions (fast-paced gameplay)")
                        else:
                            description_parts.append(f"{scene_changes} scene changes (level/area transitions)")
                    else:
                        description_parts.append(f"{scene_changes} scene changes detected")
                
                avg_brightness = np.mean([scene.get('brightness', 128) for scene in scenes])
                if avg_brightness < 80:
                    description_parts.append("Dark/low-light environment" if is_gaming else "Dark/low-light scenes")
                elif avg_brightness > 180:
                    description_parts.append("Bright/well-lit environment" if is_gaming else "Bright/well-lit scenes")
            
            # Describe text elements with gaming UI focus
            text_elements = analysis_results.get('text_elements', [])
            if text_elements:
                unique_texts = set(elem['text'] for elem in text_elements)
                if is_gaming:
                    if len(unique_texts) > 5:
                        description_parts.append("Rich UI with multiple text elements (HUD, menus, indicators)")
                    elif len(unique_texts) > 2:
                        description_parts.append("Game UI elements with text overlays")
                    else:
                        description_parts.append(f"UI text: {', '.join(list(unique_texts))}")
                else:
                    if len(unique_texts) <= 3:
                        description_parts.append(f"Text elements: {', '.join(list(unique_texts))}")
                    else:
                        description_parts.append(f"Multiple text elements detected ({len(unique_texts)} unique)")
            
            # Describe movement with gaming-specific analysis
            movement = analysis_results.get('movement_analysis', {})
            if movement.get('movement_detected', False):
                intensity = movement.get('movement_intensity', 0)
                camera_movement = movement.get('camera_movement', 'static')
                object_movement = movement.get('object_movement', 'none')
                
                movement_desc = []
                if is_gaming:
                    if camera_movement != 'static':
                        if camera_movement == 'horizontal_pan':
                            movement_desc.append("horizontal camera control/panning")
                        elif camera_movement == 'vertical_tilt':
                            movement_desc.append("vertical camera movement/aiming")
                        elif camera_movement == 'complex':
                            movement_desc.append("dynamic camera control (3D movement)")
                    
                    if object_movement != 'none':
                        if object_movement == 'fast':
                            movement_desc.append("fast-paced character/object movement")
                        elif object_movement == 'moderate':
                            movement_desc.append("moderate character movement")
                        else:
                            movement_desc.append(f"{object_movement} gameplay movement")
                    
                    if intensity > 0.5:
                        movement_desc.append("high-intensity action sequences")
                    elif intensity > 0.3:
                        movement_desc.append("active gameplay")
                    elif intensity > 0.1:
                        movement_desc.append("moderate gameplay activity")
                else:
                    if camera_movement != 'static':
                        if camera_movement == 'horizontal_pan':
                            movement_desc.append("horizontal camera panning")
                        elif camera_movement == 'vertical_tilt':
                            movement_desc.append("vertical camera tilting")
                        elif camera_movement == 'complex':
                            movement_desc.append("complex camera movement")
                    
                    if object_movement != 'none':
                        movement_desc.append(f"{object_movement} object movement")
                    
                    if intensity > 0.3:
                        movement_desc.append("High movement/action")
                    elif intensity > 0.1:
                        movement_desc.append("Moderate movement")
                
                if movement_desc:
                    description_parts.append(", ".join(movement_desc))
            else:
                if is_gaming:
                    description_parts.append("Static/menu screen or cutscene")
                else:
                    description_parts.append("Minimal movement")
            
            # Describe visual style with gaming-specific details
            style = analysis_results.get('visual_style', {})
            if style:
                style_desc = []
                
                lighting = style.get('lighting', 'unknown')
                if lighting != 'unknown':
                    if is_gaming:
                        lighting_map = {
                            'natural': 'realistic lighting',
                            'artificial': 'game lighting effects',
                            'dramatic': 'cinematic lighting',
                            'soft': 'ambient lighting'
                        }
                        style_desc.append(lighting_map.get(lighting, f"{lighting} lighting"))
                    else:
                        style_desc.append(f"{lighting} lighting")
                
                contrast = style.get('contrast_level', 'medium')
                if contrast != 'medium':
                    if is_gaming and contrast == 'high':
                        style_desc.append("high contrast graphics (enhanced visibility)")
                    else:
                        style_desc.append(f"{contrast} contrast")
                
                saturation = style.get('saturation_level', 'balanced')
                if saturation != 'balanced':
                    if is_gaming:
                        saturation_map = {
                            'vibrant': 'vibrant game colors',
                            'muted': 'realistic color palette',
                            'desaturated': 'stylized/filtered colors'
                        }
                        style_desc.append(saturation_map.get(saturation, f"{saturation} colors"))
                    else:
                        style_desc.append(f"{saturation} colors")
                
                complexity = style.get('visual_complexity', 'medium')
                if complexity != 'medium':
                    if is_gaming:
                        complexity_map = {
                            'high': 'detailed graphics with rich visual elements',
                            'low': 'simplified/stylized graphics',
                            'very_high': 'highly detailed AAA game graphics'
                        }
                        style_desc.append(complexity_map.get(complexity, f"{complexity} visual complexity"))
                    else:
                        style_desc.append(f"{complexity} visual complexity")
                
                aesthetic_score = style.get('aesthetic_score', 0.0)
                if is_gaming:
                    if aesthetic_score > 0.8:
                        style_desc.append("high-quality game graphics")
                    elif aesthetic_score > 0.6:
                        style_desc.append("good visual quality")
                    elif aesthetic_score < 0.3:
                        style_desc.append("basic/indie game graphics")
                else:
                    if aesthetic_score > 0.7:
                        style_desc.append("high aesthetic quality")
                    elif aesthetic_score < 0.3:
                        style_desc.append("simple aesthetic")
                
                if style_desc:
                    description_parts.append(", ".join(style_desc))
            
            # Add gaming-specific context if detected
            if is_gaming and objects:
                # Try to infer game genre based on objects
                vehicle_objects = ['car', 'truck', 'motorcycle', 'airplane', 'boat']
                person_objects = ['person']
                
                vehicles = sum(1 for obj in objects if obj['class_name'] in vehicle_objects)
                people = sum(1 for obj in objects if obj['class_name'] in person_objects)
                
                if vehicles > people and vehicles > 2:
                    description_parts.append("Vehicle-focused gameplay (racing/driving game)")
                elif people > 3:
                    description_parts.append("Character-focused gameplay (action/adventure game)")
            
            # Add color palette information
            color_palette = style.get('color_palette', [])
            if color_palette and len(color_palette) > 0:
                dominant_colors = len([c for c in color_palette if c])
                if dominant_colors > 0:
                    if is_gaming:
                        description_parts.append(f"{dominant_colors}-color game palette")
                    else:
                        description_parts.append(f"{dominant_colors}-color palette")
            
            final_description = ". ".join(description_parts) if description_parts else "Visual analysis completed"
            
            # Add gaming-specific summary if applicable
            if is_gaming and description_parts:
                final_description = f"Gaming content: {final_description}"
            
            return final_description
            
        except Exception as e:
            logger.error(f"Description generation failed: {e}")
            return "Visual content processed"
            if style:
                style_desc = []
                
                lighting = style.get('lighting', 'unknown')
                if lighting != 'unknown':
                    style_desc.append(f"{lighting} lighting")
                
                contrast = style.get('contrast_level', 'medium')
                if contrast != 'medium':
                    style_desc.append(f"{contrast} contrast")
                
                saturation = style.get('saturation_level', 'balanced')
                if saturation != 'balanced':
                    style_desc.append(f"{saturation} colors")
                
                color_temp = style.get('color_temperature', 'neutral')
                if color_temp != 'neutral':
                    style_desc.append(f"{color_temp} color temperature")
                
                complexity = style.get('visual_complexity', 'medium')
                if complexity != 'medium':
                    style_desc.append(f"{complexity} visual complexity")
                
                composition = style.get('composition', 'unknown')
                if composition != 'unknown' and composition != 'balanced':
                    style_desc.append(f"{composition} composition")
                
                aesthetic_score = style.get('aesthetic_score', 0.0)
                if aesthetic_score > 0.7:
                    style_desc.append("high aesthetic quality")
                elif aesthetic_score < 0.3:
                    style_desc.append("simple aesthetic")
                
                if style_desc:
                    description_parts.append(", ".join(style_desc))
            
            # Add color palette information
            color_palette = style.get('color_palette', [])
            if color_palette and len(color_palette) > 0:
                dominant_colors = len([c for c in color_palette if c])
                if dominant_colors > 0:
                    description_parts.append(f"{dominant_colors}-color palette")
            
            return ". ".join(description_parts) if description_parts else "Visual analysis completed"
            
        except Exception as e:
            logger.error(f"Description generation failed: {e}")
            return "Visual content processed"
    
    def _calculate_confidence(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate confidence score for visual analysis"""
        try:
            confidence_factors = []
            
            # Object detection confidence
            objects = analysis_results.get('objects', [])
            if objects:
                obj_confidences = [obj['confidence'] for obj in objects]
                confidence_factors.append(np.mean(obj_confidences))
            
            # Text extraction confidence
            text_elements = analysis_results.get('text_elements', [])
            if text_elements:
                text_confidences = [elem['confidence'] for elem in text_elements]
                confidence_factors.append(np.mean(text_confidences))
            
            # Scene analysis confidence (based on number of frames processed)
            scenes = analysis_results.get('scenes', [])
            if scenes:
                confidence_factors.append(min(len(scenes) / 10.0, 1.0))  # More frames = higher confidence
            
            return np.mean(confidence_factors) if confidence_factors else 0.5
            
        except Exception:
            return 0.5
    
    def get_supported_operations(self) -> List[str]:
        """Return list of supported operations"""
        operations = ['frame_extraction', 'scene_analysis', 'movement_analysis', 'visual_style_analysis']
        
        if YOLO_AVAILABLE:
            operations.append('object_detection')
        
        if OCR_AVAILABLE:
            operations.append('text_extraction')
        
        return operations
    
    async def health_check(self) -> bool:
        """Check if component is healthy and ready"""
        return self.is_initialized and self.status != ProcessingStatus.FAILED
    
    async def cleanup(self) -> None:
        """Clean up visual processing resources"""
        logger.info("Cleaning up Visual Extractor")
        if self.yolo_model:
            del self.yolo_model
            self.yolo_model = None
    
    async def _visual_fallback(self, context: Dict[str, Any]) -> ProcessingResult:
        """Fallback strategy for visual processing failures"""
        try:
            logger.info("Executing visual processing fallback")
            
            # Create minimal visual analysis result
            fallback_data = {
                'objects': [],
                'scenes': [{'frame_index': 0, 'timestamp': 0.0, 'brightness': 128.0, 'contrast': 50.0}],
                'text_elements': [],
                'movement_analysis': {'movement_detected': False, 'movement_intensity': 0.0},
                'visual_style': {
                    'lighting': 'unknown',
                    'composition': 'unknown',
                    'color_temperature': 'neutral'
                },
                'description': 'Visual processing unavailable - basic analysis provided',
                'confidence': 0.2,
                'fallback_used': True
            }
            
            return ProcessingResult(
                component_name=self.name,
                status=ProcessingStatus.COMPLETED,
                data=fallback_data
            )
            
        except Exception as e:
            logger.error(f"Visual fallback failed: {e}")
            return ProcessingResult(
                component_name=self.name,
                status=ProcessingStatus.FAILED,
                data={},
                error_message=f"Visual fallback failed: {str(e)}"
            )