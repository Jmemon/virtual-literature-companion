"""
Main FastAPI application for the Virtual Literature Companion.

This module provides:
- REST API endpoints for book processing and chat
- WebSocket support for real-time communication
- Static file serving for audio files
- Health check and monitoring endpoints
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import asyncio
from pathlib import Path
import json
from datetime import datetime
from loguru import logger

from src.config import settings
from src.models.schemas import (
    ChatRequest,
    ChatResponse,
    UploadBookRequest,
    UploadBookResponse,
    ProcessingRequest,
    GetBookResponse,
    WSMessage,
    WSChatMessage,
    WSEmotionUpdate,
    WSVoiceReady
)
from src.graph.workflow import handle_chat_request, build_literature_companion_graph
from src.models.state import ProcessingState, AgentState
from src.memory.honcho_client import HonchoMemoryClient
from src.ui.animation_engine import EmotiveAnimationEngine


# Global variables for managing state
active_sessions: Dict[str, Dict[str, Any]] = {}
processing_jobs: Dict[str, ProcessingState] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    """
    # Startup
    logger.info("Starting Virtual Literature Companion API")
    
    # Initialize Honcho
    async with HonchoMemoryClient() as honcho:
        success = await honcho.initialize_app()
        if success:
            logger.info("Honcho initialized successfully")
        else:
            logger.warning("Honcho initialization failed - memory features limited")
            
    # Ensure directories exist
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.voice_cache_dir.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Virtual Literature Companion API")
    

# Create FastAPI app
app = FastAPI(
    title="Virtual Literature Companion",
    description="An emotional AI companion for literary discussions",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for audio
app.mount("/audio", StaticFiles(directory=str(settings.voice_cache_dir)), name="audio")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API is running and healthy."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


# Book upload endpoint
@app.post("/api/books/upload", response_model=UploadBookResponse)
async def upload_book(
    file: UploadFile = File(...),
    user_id: str = "default_user"
):
    """
    Upload a PDF book for processing.
    
    Args:
        file: PDF file to upload
        user_id: User identifier
        
    Returns:
        Upload response with book ID and processing job ID
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
    # Save uploaded file
    file_path = settings.upload_dir / f"{user_id}_{file.filename}"
    
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")
        
    # Create processing job
    processing_state = ProcessingState(
        file_path=str(file_path),
        file_content=content
    )
    
    # Start processing in background
    job_id = f"job_{datetime.utcnow().timestamp()}"
    processing_jobs[job_id] = processing_state
    
    # Launch processing task
    asyncio.create_task(process_book_background(job_id, processing_state))
    
    return UploadBookResponse(
        book_id=processing_state.metadata.book_id if processing_state.metadata else "pending",
        status="processing",
        message="Book uploaded successfully. Processing started.",
        processing_job_id=job_id
    )


async def process_book_background(job_id: str, state: ProcessingState):
    """
    Process book in the background using the multi-agent system.
    
    Args:
        job_id: Processing job ID
        state: Processing state
    """
    try:
        logger.info(f"Starting background processing for job {job_id}")
        
        # Build and run the processing graph
        graph = build_literature_companion_graph()
        
        # Create initial state for graph
        graph_state = {
            "file_path": state.file_path,
            "file_content": state.file_content,
            "user_id": "system",  # System processing
            "current_task": "Extract and analyze book"
        }
        
        # Run the graph
        final_state = await graph.ainvoke(graph_state)
        
        # Update processing state
        state.processing_status = "completed"
        state.metadata = final_state.get("book_metadata")
        state.table_of_contents = final_state.get("table_of_contents")
        state.chunks = final_state.get("processed_chunks", [])
        
        logger.info(f"Completed processing for job {job_id}")
        
    except Exception as e:
        logger.error(f"Error processing book: {e}")
        state.processing_status = "failed"
        state.errors.append(str(e))


# Get processing status
@app.get("/api/books/processing/{job_id}")
async def get_processing_status(job_id: str):
    """
    Get the status of a book processing job.
    
    Args:
        job_id: Processing job ID
        
    Returns:
        Processing status and progress
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    state = processing_jobs[job_id]
    
    return {
        "job_id": job_id,
        "status": state.processing_status,
        "progress": state.progress,
        "errors": state.errors,
        "book_id": state.metadata.book_id if state.metadata else None
    }


# Get book details
@app.get("/api/books/{book_id}", response_model=GetBookResponse)
async def get_book(book_id: str):
    """
    Get details about a processed book.
    
    Args:
        book_id: Book identifier
        
    Returns:
        Book metadata and processing status
    """
    # In production, this would query a database
    # For now, search through processing jobs
    for job_id, state in processing_jobs.items():
        if state.metadata and state.metadata.book_id == book_id:
            return GetBookResponse(
                book=state.metadata,
                table_of_contents=state.table_of_contents,
                total_chunks=len(state.chunks),
                processing_status=state.processing_status
            )
            
    raise HTTPException(status_code=404, detail="Book not found")


# Chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handle a chat request with the literature companion.
    
    Args:
        request: Chat request with message and context
        
    Returns:
        Emotional response with voice URL and UI animations
    """
    try:
        # Validate book exists
        book_found = False
        book_chunks = []
        book_metadata = None
        table_of_contents = None
        
        for job_id, state in processing_jobs.items():
            if state.metadata and state.metadata.book_id == request.book_id:
                if state.processing_status != "completed":
                    raise HTTPException(
                        status_code=400, 
                        detail="Book is still being processed"
                    )
                book_found = True
                book_chunks = state.chunks
                book_metadata = state.metadata
                table_of_contents = state.table_of_contents
                break
                
        if not book_found:
            raise HTTPException(status_code=404, detail="Book not found")
            
        # Update request with book context
        request_dict = request.dict()
        request_dict["processed_chunks"] = book_chunks
        request_dict["book_metadata"] = book_metadata
        request_dict["table_of_contents"] = table_of_contents
        
        # Create ChatRequest with updated data
        enhanced_request = ChatRequest(**request_dict)
        
        # Handle through multi-agent system
        response = await handle_chat_request(enhanced_request)
        
        return response
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time chat
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for real-time chat and UI updates.
    
    Args:
        websocket: WebSocket connection
        user_id: User identifier
    """
    await websocket.accept()
    
    # Store session
    session_id = f"ws_session_{datetime.utcnow().timestamp()}"
    active_sessions[session_id] = {
        "websocket": websocket,
        "user_id": user_id,
        "connected_at": datetime.utcnow()
    }
    
    # Animation engine for UI updates
    animation_engine = EmotiveAnimationEngine()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = WSMessage(**data)
            
            if message.type == "chat":
                # Handle chat message
                chat_data = WSChatMessage(**data)
                chat_request = ChatRequest(**chat_data.data)
                
                # Process through multi-agent system
                response = await handle_chat_request(chat_request)
                
                # Send response
                await websocket.send_json({
                    "type": "chat_response",
                    "data": response.dict()
                })
                
                # Generate and send UI state update
                ui_state = animation_engine.generate_ui_state(
                    response.response.emotions,
                    response.response.emotion_intensities,
                    len(response.response.text)
                )
                
                emotion_update = WSEmotionUpdate(
                    type="emotion_update",
                    data=ui_state
                )
                
                await websocket.send_json(emotion_update.dict())
                
                # Send voice ready notification if voice was generated
                if response.voice_url:
                    voice_ready = WSVoiceReady(
                        type="voice_ready",
                        data={
                            "voice_url": response.voice_url,
                            "duration": str(len(response.response.text) * 0.05)  # Estimate
                        }
                    )
                    await websocket.send_json(voice_ready.dict())
                    
            elif message.type == "ping":
                # Respond to ping
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Clean up session
        del active_sessions[session_id]


# Get active sessions (admin endpoint)
@app.get("/api/admin/sessions")
async def get_active_sessions():
    """
    Get information about active WebSocket sessions.
    
    Returns:
        List of active sessions
    """
    return {
        "total_sessions": len(active_sessions),
        "sessions": [
            {
                "session_id": session_id,
                "user_id": session["user_id"],
                "connected_at": session["connected_at"].isoformat()
            }
            for session_id, session in active_sessions.items()
        ]
    }


# Clear cache endpoint (admin)
@app.post("/api/admin/clear-cache")
async def clear_cache(older_than_days: int = 7):
    """
    Clear old cached audio files.
    
    Args:
        older_than_days: Clear files older than this many days
        
    Returns:
        Status message
    """
    from src.voice.tts_engine import EmotionalTTSEngine
    
    try:
        engine = EmotionalTTSEngine()
        await engine.clear_cache(older_than_days)
        
        return {
            "status": "success",
            "message": f"Cleared cache files older than {older_than_days} days"
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Example HTML page for testing
@app.get("/")
async def root():
    """Serve a simple test page."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Virtual Literature Companion</title>
    </head>
    <body>
        <h1>Virtual Literature Companion API</h1>
        <p>API is running!</p>
        <ul>
            <li><a href="/docs">API Documentation</a></li>
            <li><a href="/health">Health Check</a></li>
        </ul>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )