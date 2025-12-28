"""
Enhanced Video Analysis API with Modular Architecture
"""
import os
import asyncio
import logging
import tempfile
import shutil
import json
from datetime import datetime
from typing import Dict, Any, Optional
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import modular components
from components.orchestrator import AnalysisOrchestrator
from components.visual import VisualExtractor
from components.audio import AudioProcessor
from components.llm import LLMIntegration
from components.classifier import ContentClassifier
from components.multimodal import MultiModalIntegrator
from components.base import (
    AnalysisSession, AnalysisReport, VideoMetadata, ContentType, ProcessingStatus
)
from components.error_handler import get_error_handler
from database import DatabaseManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="VidIntel Pro - Comprehensive Video Analysis Platform",
    description="Advanced video analysis with modular AI-powered components",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
orchestrator = None
database = None
temp_files = []

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected for session {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected for session {session_id}")

    async def send_progress_update(self, session_id: str, data: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(
                    json.dumps(data)
                )
            except Exception as e:
                logger.error(f"Failed to send WebSocket message to {session_id}: {e}")
                self.disconnect(session_id)

    async def broadcast_to_session(self, session_id: str, message: dict):
        await self.send_progress_update(session_id, message)

manager = ConnectionManager()

# Pydantic models for API
class AnalysisRequest(BaseModel):
    session_id: Optional[str] = None
    analysis_options: Dict[str, Any] = {}

class AnalysisStatusResponse(BaseModel):
    session_id: str
    status: str
    progress: float
    segments_completed: int
    total_segments: int
    current_segment: Optional[str] = None
    estimated_completion: Optional[str] = None

class AnalysisResultResponse(BaseModel):
    session_id: str
    status: str
    content_type: str
    total_duration: float
    processing_time: float
    segments: list
    summary: str
    created_at: str

async def initialize_components():
    """Initialize all processing components"""
    global orchestrator, database
    
    try:
        logger.info("Initializing comprehensive video analysis system...")
        
        # Initialize database
        database = DatabaseManager()
        
        # Initialize orchestrator
        orchestrator = AnalysisOrchestrator()
        await orchestrator.initialize()
        
        # Initialize and register components
        components = {
            'content_classifier': ContentClassifier(),
            'visual_extractor': VisualExtractor(),
            'audio_processor': AudioProcessor(),
            'llm_integration': LLMIntegration(),
            'multimodal_integrator': MultiModalIntegrator()
        }
        
        # Initialize each component
        for name, component in components.items():
            try:
                success = await component.initialize()
                if success:
                    orchestrator.register_component(name, component)
                    logger.info(f"✓ {name} initialized successfully")
                else:
                    logger.warning(f"✗ {name} initialization failed - system will continue with degraded functionality")
                    # Register component anyway for fallback mechanisms
                    orchestrator.register_component(name, component)
            except Exception as e:
                logger.error(f"✗ {name} initialization error: {e} - registering for fallback")
                # Still register component for error handling and fallback
                orchestrator.register_component(name, component)
        
        # Get error handler for system-wide error reporting
        error_handler = get_error_handler()
        logger.info(f"Error handler initialized with circuit breakers for {len(error_handler.circuit_breakers)} components")
        
        logger.info("System initialization completed with comprehensive error handling")
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    await initialize_components()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global orchestrator, temp_files
    
    if orchestrator:
        await orchestrator.cleanup()
    
    # Cleanup temporary files
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")

@app.websocket("/ws/analysis/{session_id}")
async def websocket_analysis_progress(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time analysis progress updates"""
    await manager.connect(websocket, session_id)
    
    try:
        # Send initial status
        if database:
            session_data = database.get_session(session_id)
            if session_data:
                initial_status = {
                    "session_id": session_id,
                    "status": session_data['status'],
                    "progress": session_data.get('progress_percentage', 0.0),
                    "segments_completed": session_data.get('completed_segments', 0),
                    "total_segments": session_data.get('total_segments', 0),
                    "current_segment": "Initializing..."
                }
                await manager.send_progress_update(session_id, initial_status)
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages (ping/pong, etc.)
                data = await websocket.receive_text()
                # Echo back for connection health check
                await websocket.send_text(f"pong: {data}")
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error for session {session_id}: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket connection error for session {session_id}: {e}")
    finally:
        manager.disconnect(session_id)

@app.post("/analyze/comprehensive", response_model=dict)
async def comprehensive_video_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    analysis_options: str = "{}"
):
    """
    Comprehensive video analysis using modular architecture
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Analysis system not initialized")
    
    session_id = str(uuid.uuid4())
    temp_dir = None
    
    try:
        # Save uploaded file
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, file.filename)
        
        with open(video_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        temp_files.append(video_path)
        
        logger.info(f"Starting comprehensive analysis for session {session_id}")
        
        # Start analysis in background
        background_tasks.add_task(
            run_comprehensive_analysis,
            session_id,
            video_path,
            temp_dir
        )
        
        return {
            "session_id": session_id,
            "status": "started",
            "message": "Analysis started successfully",
            "video_filename": file.filename
        }
        
    except Exception as e:
        logger.error(f"Failed to start analysis: {e}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed to start: {str(e)}")

async def run_comprehensive_analysis(session_id: str, video_path: str, temp_dir: str):
    """Run comprehensive analysis in background"""
    try:
        # Send initial progress update
        await manager.send_progress_update(session_id, {
            "session_id": session_id,
            "status": "processing",
            "progress": 0.0,
            "current_segment": "Starting analysis...",
            "segments_completed": 0,
            "total_segments": 0
        })
        
        # Run analysis with progress callbacks
        async def progress_callback(progress_data):
            """Callback to send progress updates via WebSocket"""
            await manager.send_progress_update(session_id, {
                "session_id": session_id,
                "status": "processing",
                "progress": progress_data.get('progress', 0.0),
                "current_segment": progress_data.get('current_task', 'Processing...'),
                "segments_completed": progress_data.get('segments_completed', 0),
                "total_segments": progress_data.get('total_segments', 0),
                "preliminary_results": progress_data.get('preliminary_results', [])
            })
        
        # Run analysis
        report = await orchestrator.analyze_video(video_path, session_id, progress_callback=progress_callback)
        
        # Send completion update
        await manager.send_progress_update(session_id, {
            "session_id": session_id,
            "status": "completed",
            "progress": 100.0,
            "current_segment": "Analysis complete!",
            "segments_completed": len(report.segments) if report.segments else 0,
            "total_segments": len(report.segments) if report.segments else 0
        })
        
        # Save final report to database
        if database:
            database.update_session_status(
                session_id, 
                ProcessingStatus.COMPLETED,
                report.summary,
                report.processing_time
            )
        
        logger.info(f"Analysis completed for session {session_id}")
        
    except Exception as e:
        logger.error(f"Analysis failed for session {session_id}: {e}")
        
        # Send error update
        await manager.send_progress_update(session_id, {
            "session_id": session_id,
            "status": "failed",
            "error": str(e),
            "progress": 0.0
        })
        
        # Update status to failed
        if database:
            database.update_session_status(session_id, ProcessingStatus.FAILED)
    
    finally:
        # Cleanup temporary files
        try:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if video_path in temp_files:
                temp_files.remove(video_path)
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

@app.get("/analyze/status/{session_id}", response_model=AnalysisStatusResponse)
async def get_analysis_status(session_id: str):
    """Get analysis status and progress"""
    if not database:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        session_data = database.get_session(session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get orchestrator status if available
        orchestrator_status = None
        if orchestrator:
            orchestrator_status = orchestrator.get_session_status(session_id)
        
        progress = session_data.get('progress_percentage', 0.0)
        if orchestrator_status:
            progress = orchestrator_status.get('progress', progress)
        
        return AnalysisStatusResponse(
            session_id=session_id,
            status=session_data['status'],
            progress=progress,
            segments_completed=session_data.get('completed_segments', 0),
            total_segments=session_data.get('total_segments', 0),
            current_segment=orchestrator_status.get('current_segment') if orchestrator_status else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get status")

@app.get("/analyze/results/{session_id}", response_model=dict)
async def get_analysis_results(session_id: str):
    """Get comprehensive analysis results"""
    if not database:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Get session data
        session_data = database.get_session(session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session_data['status'] != ProcessingStatus.COMPLETED.value:
            raise HTTPException(status_code=202, detail="Analysis not completed yet")
        
        # Get segment analyses
        segments = database.get_segment_analyses(session_id)
        
        # Get detected objects for each segment
        for segment in segments:
            segment['detected_objects'] = database.get_detected_objects(segment['segment_id'])
        
        # Format response with validation information
        result = {
            "session_id": session_id,
            "status": session_data['status'],
            "content_type": session_data['content_type'],
            "total_duration": session_data['total_duration'],
            "processing_time": session_data.get('processing_time', 0),
            "summary": session_data.get('summary', ''),
            "created_at": session_data['created_at'],
            "completed_at": session_data.get('completed_at'),
            "segments": segments,
            "video_metadata": {
                "filename": session_data['video_filename'],
                "duration": session_data['total_duration'],
                "content_type": session_data['content_type']
            },
            # Add output validation information
            "output_validation": {
                "validation_passed": session_data.get('output_validation_passed', True),
                "completeness_score": session_data.get('output_completeness_score', 1.0),
                "validation_errors": session_data.get('validation_errors', []),
                "validation_warnings": session_data.get('validation_warnings', [])
            }
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis results: {e}")
        raise HTTPException(status_code=500, detail="Failed to get results")

@app.get("/analyze/segments/{session_id}")
async def get_segment_analyses(session_id: str):
    """Get detailed segment analyses"""
    if not database:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        segments = database.get_segment_analyses(session_id)
        
        if not segments:
            raise HTTPException(status_code=404, detail="No segments found for session")
        
        # Add detected objects to each segment
        for segment in segments:
            segment['detected_objects'] = database.get_detected_objects(segment['segment_id'])
        
        return {
            "session_id": session_id,
            "segments": segments,
            "total_segments": len(segments)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get segment analyses: {e}")
        raise HTTPException(status_code=500, detail="Failed to get segments")

@app.get("/system/health")
async def system_health():
    """System health check with comprehensive error monitoring"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {},
        "error_handler": {},
        "circuit_breakers": {}
    }
    
    # Check orchestrator
    if orchestrator:
        health_status["components"]["orchestrator"] = await orchestrator.health_check()
        
        # Check registered components
        for name, component in orchestrator.components.items():
            try:
                health_status["components"][name] = await component.health_check()
            except Exception as e:
                health_status["components"][name] = False
                logger.warning(f"Health check failed for {name}: {e}")
    else:
        health_status["status"] = "unhealthy"
        health_status["components"]["orchestrator"] = False
    
    # Check database
    health_status["components"]["database"] = database is not None
    
    # Get error handler status
    error_handler = get_error_handler()
    health_status["error_handler"] = {
        "total_errors": len(error_handler.error_history),
        "component_health": error_handler.get_component_health(),
        "recent_errors": len([e for e in error_handler.error_history if (datetime.now() - e.timestamp).total_seconds() < 3600])
    }
    
    # Get circuit breaker status
    health_status["circuit_breakers"] = error_handler.get_circuit_breaker_status()
    
    # Overall status determination
    failed_components = sum(1 for status in health_status["components"].values() if not status)
    open_circuit_breakers = sum(1 for cb in health_status["circuit_breakers"].values() if cb["state"] == "open")
    
    if failed_components > 0 or open_circuit_breakers > 0:
        health_status["status"] = "degraded"
        health_status["issues"] = {
            "failed_components": failed_components,
            "open_circuit_breakers": open_circuit_breakers
        }
    
    return health_status

@app.get("/system/components")
async def get_system_components():
    """Get information about system components"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    components_info = {}
    
    for name, component in orchestrator.components.items():
        try:
            components_info[name] = {
                "name": component.name,
                "initialized": component.is_initialized,
                "status": component.status.value,
                "supported_operations": component.get_supported_operations() if hasattr(component, 'get_supported_operations') else []
            }
        except Exception as e:
            components_info[name] = {
                "name": name,
                "error": str(e)
            }
    
    return {
        "total_components": len(components_info),
        "components": components_info
    }

@app.get("/system/errors")
async def get_system_errors(component: str = None, limit: int = 50):
    """Get system error history"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        error_handler = get_error_handler()
        error_history = error_handler.get_error_history(component_name=component, limit=limit)
        
        return {
            "total_errors": len(error_history),
            "component_filter": component,
            "errors": error_history,
            "component_health": error_handler.get_component_health(component) if component else error_handler.get_component_health()
        }
        
    except Exception as e:
        logger.error(f"Failed to get error history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get error history")

@app.get("/system/circuit-breakers")
async def get_circuit_breaker_status():
    """Get circuit breaker status for all components"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        error_handler = get_error_handler()
        circuit_breaker_status = error_handler.get_circuit_breaker_status()
        
        return {
            "circuit_breakers": circuit_breaker_status,
            "summary": {
                "total": len(circuit_breaker_status),
                "open": sum(1 for cb in circuit_breaker_status.values() if cb["state"] == "open"),
                "half_open": sum(1 for cb in circuit_breaker_status.values() if cb["state"] == "half_open"),
                "closed": sum(1 for cb in circuit_breaker_status.values() if cb["state"] == "closed")
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get circuit breaker status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get circuit breaker status")

@app.post("/system/circuit-breakers/{component_name}/reset")
async def reset_circuit_breaker(component_name: str):
    """Manually reset circuit breaker for a component"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        error_handler = get_error_handler()
        success = error_handler.reset_circuit_breaker(component_name)
        
        if success:
            return {
                "component": component_name,
                "status": "reset",
                "message": f"Circuit breaker reset for {component_name}"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Circuit breaker not found for component: {component_name}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset circuit breaker: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset circuit breaker")

@app.delete("/analyze/session/{session_id}")
async def delete_analysis_session(session_id: str):
    """Delete analysis session and all related data"""
    if not database:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Check if session exists
        session_data = database.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Note: In a full implementation, you would implement a delete method in DatabaseManager
        # For now, return success
        return {
            "session_id": session_id,
            "message": "Session deletion requested (implementation pending)"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session")

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "VidIntel Pro - Comprehensive Video Analysis Platform",
        "version": "3.0.0",
        "architecture": "Modular Component-Based",
        "features": [
            "Content-adaptive analysis",
            "Multi-modal processing (visual + audio)",
            "LLM-powered descriptions",
            "Real-time progress tracking",
            "Comprehensive object detection",
            "Scene analysis and classification",
            "Advanced audio processing",
            "Modular component architecture"
        ],
        "endpoints": {
            "analysis": "/analyze/comprehensive",
            "status": "/analyze/status/{session_id}",
            "results": "/analyze/results/{session_id}",
            "segments": "/analyze/segments/{session_id}",
            "health": "/system/health",
            "components": "/system/components",
            "errors": "/system/errors",
            "circuit_breakers": "/system/circuit-breakers",
            "reset_circuit_breaker": "/system/circuit-breakers/{component_name}/reset"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)