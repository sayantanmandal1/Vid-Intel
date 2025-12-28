"""
Database schema and management for comprehensive video analysis system
"""
import sqlite3
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import uuid

from components.base import AnalysisSession, SegmentAnalysis, AnalysisReport, ProcessingStatus

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database operations for video analysis system"""
    
    def __init__(self, db_path: str = "vidintel.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Analysis sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_sessions (
                    id TEXT PRIMARY KEY,
                    video_filename TEXT NOT NULL,
                    video_path TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    total_duration REAL,
                    processing_time REAL,
                    metadata TEXT,  -- JSON metadata
                    summary TEXT,
                    -- Output validation fields
                    output_validation_passed BOOLEAN DEFAULT 1,
                    output_completeness_score REAL DEFAULT 1.0,
                    validation_errors TEXT,  -- JSON array of validation errors
                    validation_warnings TEXT  -- JSON array of validation warnings
                )
            ''')
            
            # Add validation columns to existing tables if they don't exist
            try:
                cursor.execute('ALTER TABLE analysis_sessions ADD COLUMN output_validation_passed BOOLEAN DEFAULT 1')
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                cursor.execute('ALTER TABLE analysis_sessions ADD COLUMN output_completeness_score REAL DEFAULT 1.0')
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                cursor.execute('ALTER TABLE analysis_sessions ADD COLUMN validation_errors TEXT')
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                cursor.execute('ALTER TABLE analysis_sessions ADD COLUMN validation_warnings TEXT')
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            # Segment analyses table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS segment_analyses (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    segment_id TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    duration REAL NOT NULL,
                    visual_description TEXT,
                    audio_description TEXT,
                    combined_narrative TEXT,
                    transcription TEXT,
                    confidence_score REAL,
                    processing_time REAL,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES analysis_sessions (id)
                )
            ''')
            
            # Detected objects table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detected_objects (
                    id TEXT PRIMARY KEY,
                    segment_id TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bounding_box TEXT,  -- JSON bounding box coordinates
                    timestamp REAL NOT NULL,
                    frame_index INTEGER,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (segment_id) REFERENCES segment_analyses (segment_id)
                )
            ''')
            
            # Processing results table (for component-specific results)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_results (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    segment_id TEXT,
                    component_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_data TEXT,  -- JSON result data
                    error_message TEXT,
                    processing_time REAL,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES analysis_sessions (id)
                )
            ''')
            
            # Video metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS video_metadata (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    duration REAL,
                    fps REAL,
                    resolution TEXT,
                    format TEXT,
                    hash TEXT,
                    upload_time TIMESTAMP NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES analysis_sessions (id)
                )
            ''')
            
            # Analysis progress tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_progress (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    total_segments INTEGER NOT NULL,
                    completed_segments INTEGER DEFAULT 0,
                    current_segment TEXT,
                    progress_percentage REAL DEFAULT 0.0,
                    estimated_completion TIMESTAMP,
                    last_updated TIMESTAMP NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES analysis_sessions (id)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_status ON analysis_sessions (status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_created ON analysis_sessions (created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_segments_session ON segment_analyses (session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_objects_segment ON detected_objects (segment_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_session ON processing_results (session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_progress_session ON analysis_progress (session_id)')
            
            conn.commit()
            conn.close()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def create_analysis_session(self, session: AnalysisSession) -> bool:
        """Create a new analysis session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert session
            cursor.execute('''
                INSERT INTO analysis_sessions 
                (id, video_filename, video_path, content_type, status, created_at, 
                 total_duration, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session.session_id,
                Path(session.video_path).name,
                session.video_path,
                session.content_type.value,
                session.status.value,
                session.created_at.isoformat() if hasattr(session.created_at, 'isoformat') else str(session.created_at),
                session.metadata.duration,
                json.dumps({
                    'fps': session.metadata.fps,
                    'resolution': session.metadata.resolution,
                    'format': session.metadata.format,
                    'file_size': session.metadata.file_size,
                    'hash': session.metadata.hash
                })
            ))
            
            # Insert video metadata
            cursor.execute('''
                INSERT INTO video_metadata
                (id, session_id, filename, file_path, file_size, duration, fps, 
                 resolution, format, hash, upload_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()),
                session.session_id,
                Path(session.video_path).name,
                session.video_path,
                session.metadata.file_size,
                session.metadata.duration,
                session.metadata.fps,
                session.metadata.resolution,
                session.metadata.format,
                session.metadata.hash,
                datetime.now().isoformat()
            ))
            
            # Initialize progress tracking
            cursor.execute('''
                INSERT INTO analysis_progress
                (id, session_id, total_segments, last_updated)
                VALUES (?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()),
                session.session_id,
                len(session.segments),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Created analysis session: {session.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create analysis session: {e}")
            return False
    
    def save_segment_analysis(self, session_id: str, analysis: SegmentAnalysis) -> bool:
        """Save segment analysis results"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert segment analysis
            cursor.execute('''
                INSERT INTO segment_analyses
                (id, session_id, segment_id, start_time, end_time, duration,
                 visual_description, audio_description, combined_narrative,
                 transcription, confidence_score, processing_time, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()),
                session_id,
                analysis.segment.segment_id,
                analysis.segment.start_time,
                analysis.segment.end_time,
                analysis.segment.duration,
                analysis.visual_description,
                analysis.audio_description,
                analysis.combined_narrative,
                analysis.transcription,
                analysis.confidence_score,
                analysis.processing_time,
                datetime.now().isoformat()
            ))
            
            # Insert detected objects
            for obj in analysis.detected_objects:
                cursor.execute('''
                    INSERT INTO detected_objects
                    (id, segment_id, class_name, confidence, bounding_box, 
                     timestamp, frame_index, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(uuid.uuid4()),
                    analysis.segment.segment_id,
                    obj.get('class_name', 'unknown'),
                    obj.get('confidence', 0.0),
                    json.dumps(obj.get('bounding_box', [])),
                    obj.get('timestamp', 0.0),
                    obj.get('frame_index', 0),
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved segment analysis: {analysis.segment.segment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save segment analysis: {e}")
            return False
    
    def save_processing_result(self, session_id: str, segment_id: Optional[str], 
                             component_name: str, result_data: Dict[str, Any],
                             status: ProcessingStatus, error_message: Optional[str] = None,
                             processing_time: Optional[float] = None) -> bool:
        """Save component processing result"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO processing_results
                (id, session_id, segment_id, component_name, status, result_data,
                 error_message, processing_time, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()),
                session_id,
                segment_id,
                component_name,
                status.value,
                json.dumps(result_data),
                error_message,
                processing_time,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save processing result: {e}")
            return False
    
    def update_session_status(self, session_id: str, status: ProcessingStatus,
                            summary: Optional[str] = None, processing_time: Optional[float] = None,
                            validation_passed: Optional[bool] = None,
                            completeness_score: Optional[float] = None,
                            validation_errors: Optional[List[str]] = None,
                            validation_warnings: Optional[List[str]] = None) -> bool:
        """Update analysis session status with optional validation information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            update_fields = ['status = ?']
            values = [status.value]
            
            if status == ProcessingStatus.COMPLETED:
                update_fields.append('completed_at = ?')
                values.append(datetime.now().isoformat())
            
            if summary:
                update_fields.append('summary = ?')
                values.append(summary)
            
            if processing_time:
                update_fields.append('processing_time = ?')
                values.append(processing_time)
            
            # Add validation fields
            if validation_passed is not None:
                update_fields.append('output_validation_passed = ?')
                values.append(validation_passed)
            
            if completeness_score is not None:
                update_fields.append('output_completeness_score = ?')
                values.append(completeness_score)
            
            if validation_errors is not None:
                update_fields.append('validation_errors = ?')
                values.append(json.dumps(validation_errors))
            
            if validation_warnings is not None:
                update_fields.append('validation_warnings = ?')
                values.append(json.dumps(validation_warnings))
            
            values.append(session_id)
            
            cursor.execute(f'''
                UPDATE analysis_sessions 
                SET {', '.join(update_fields)}
                WHERE id = ?
            ''', values)
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session status: {e}")
            return False
    
    def update_progress(self, session_id: str, completed_segments: int, 
                       current_segment: Optional[str] = None) -> bool:
        """Update analysis progress"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total segments
            cursor.execute('SELECT total_segments FROM analysis_progress WHERE session_id = ?', (session_id,))
            result = cursor.fetchone()
            
            if not result:
                return False
            
            total_segments = result[0]
            progress_percentage = (completed_segments / total_segments) * 100 if total_segments > 0 else 0
            
            cursor.execute('''
                UPDATE analysis_progress
                SET completed_segments = ?, current_segment = ?, 
                    progress_percentage = ?, last_updated = ?
                WHERE session_id = ?
            ''', (
                completed_segments,
                current_segment,
                progress_percentage,
                datetime.now().isoformat(),
                session_id
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update progress: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis session by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT s.*, p.progress_percentage, p.completed_segments, p.total_segments
                FROM analysis_sessions s
                LEFT JOIN analysis_progress p ON s.id = p.session_id
                WHERE s.id = ?
            ''', (session_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                columns = [desc[0] for desc in cursor.description]
                session_data = dict(zip(columns, result))
                
                # Parse JSON fields
                if session_data.get('validation_errors'):
                    try:
                        session_data['validation_errors'] = json.loads(session_data['validation_errors'])
                    except (json.JSONDecodeError, TypeError):
                        session_data['validation_errors'] = []
                else:
                    session_data['validation_errors'] = []
                
                if session_data.get('validation_warnings'):
                    try:
                        session_data['validation_warnings'] = json.loads(session_data['validation_warnings'])
                    except (json.JSONDecodeError, TypeError):
                        session_data['validation_warnings'] = []
                else:
                    session_data['validation_warnings'] = []
                
                return session_data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
    
    def get_segment_analyses(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all segment analyses for a session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM segment_analyses 
                WHERE session_id = ? 
                ORDER BY start_time
            ''', (session_id,))
            
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            conn.close()
            
            return [dict(zip(columns, result)) for result in results]
            
        except Exception as e:
            logger.error(f"Failed to get segment analyses: {e}")
            return []
    
    def get_detected_objects(self, segment_id: str) -> List[Dict[str, Any]]:
        """Get detected objects for a segment"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM detected_objects 
                WHERE segment_id = ? 
                ORDER BY timestamp
            ''', (segment_id,))
            
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            conn.close()
            
            objects = []
            for result in results:
                obj_dict = dict(zip(columns, result))
                # Parse JSON bounding box
                if obj_dict['bounding_box']:
                    try:
                        obj_dict['bounding_box'] = json.loads(obj_dict['bounding_box'])
                    except json.JSONDecodeError:
                        obj_dict['bounding_box'] = []
                objects.append(obj_dict)
            
            return objects
            
        except Exception as e:
            logger.error(f"Failed to get detected objects: {e}")
            return []
    
    def get_processing_results(self, session_id: str, component_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get processing results for a session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if component_name:
                cursor.execute('''
                    SELECT * FROM processing_results 
                    WHERE session_id = ? AND component_name = ?
                    ORDER BY created_at
                ''', (session_id, component_name))
            else:
                cursor.execute('''
                    SELECT * FROM processing_results 
                    WHERE session_id = ?
                    ORDER BY created_at
                ''', (session_id,))
            
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            conn.close()
            
            processed_results = []
            for result in results:
                result_dict = dict(zip(columns, result))
                # Parse JSON result data
                if result_dict['result_data']:
                    try:
                        result_dict['result_data'] = json.loads(result_dict['result_data'])
                    except json.JSONDecodeError:
                        result_dict['result_data'] = {}
                processed_results.append(result_dict)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Failed to get processing results: {e}")
            return []
    
    def cleanup_old_sessions(self, days_old: int = 7) -> int:
        """Clean up old analysis sessions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            # Get sessions to delete
            cursor.execute('''
                SELECT id FROM analysis_sessions 
                WHERE created_at < ? AND status IN ('completed', 'failed')
            ''', (cutoff_date.isoformat(),))
            
            session_ids = [row[0] for row in cursor.fetchall()]
            
            if not session_ids:
                conn.close()
                return 0
            
            # Delete related records
            placeholders = ','.join(['?' for _ in session_ids])
            
            cursor.execute(f'DELETE FROM analysis_progress WHERE session_id IN ({placeholders})', session_ids)
            cursor.execute(f'DELETE FROM processing_results WHERE session_id IN ({placeholders})', session_ids)
            cursor.execute(f'DELETE FROM video_metadata WHERE session_id IN ({placeholders})', session_ids)
            
            # Delete segment analyses and related objects
            cursor.execute(f'''
                DELETE FROM detected_objects WHERE segment_id IN (
                    SELECT segment_id FROM segment_analyses WHERE session_id IN ({placeholders})
                )
            ''', session_ids)
            
            cursor.execute(f'DELETE FROM segment_analyses WHERE session_id IN ({placeholders})', session_ids)
            cursor.execute(f'DELETE FROM analysis_sessions WHERE id IN ({placeholders})', session_ids)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up {len(session_ids)} old sessions")
            return len(session_ids)
            
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            return 0