"""
Main FastAPI application for the Literary Companion.

This module provides:
- WebSocket endpoints for real-time conversation
- REST endpoints for book management
- Static file serving for the web interface
- Session management for multiple users
"""

import os
import uuid
import json
import asyncio
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

from .models.state import ConversationState, BookLocation
from .agents.literary_companion_graph import LiteraryCompanionGraph
from .tools.book_processor import BookProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Virtual Literature Companion",
    description="An AI-powered interactive literary companion for deeper text understanding",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for active sessions
active_sessions: Dict[str, ConversationState] = {}

# Initialize the agent
agent = LiteraryCompanionGraph(
    llm_provider=os.getenv("LLM_PROVIDER", "anthropic"),
    model_name=os.getenv("MODEL_NAME", "claude-3-opus-20240229")
)


class BookIngestionRequest(BaseModel):
    """Request model for book ingestion."""
    title: str = Field(..., description="Title of the book")
    author: str = Field(..., description="Author of the book")
    text: str = Field(..., description="Full text of the book")


class SessionResponse(BaseModel):
    """Response model for session creation."""
    session_id: str
    book_title: str
    author: str
    table_of_contents: Any


class ProgressUpdateRequest(BaseModel):
    """Request model for updating reading progress."""
    chapter_number: int
    page_number: int


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting Virtual Literature Companion...")
    
    # Mount static files
    static_path = Path(__file__).parent.parent / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    else:
        logger.warning(f"Static directory not found at {static_path}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application page."""
    index_path = Path(__file__).parent.parent / "static" / "index.html"
    if index_path.exists():
        with open(index_path, 'r') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
        <html>
            <head>
                <title>Virtual Literature Companion</title>
            </head>
            <body>
                <h1>Virtual Literature Companion</h1>
                <p>The static files are not properly configured. Please check the installation.</p>
            </body>
        </html>
        """)


@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(request: BookIngestionRequest):
    """
    Create a new reading session with a book.
    
    This endpoint:
    - Processes the book text
    - Extracts table of contents
    - Creates a new conversation session
    """
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Process the book
        book_processor = BookProcessor()
        toc = book_processor.ingest_book(
            text=request.text,
            book_title=request.title,
            author=request.author
        )
        
        # Create conversation state
        conversation_state = ConversationState(
            session_id=session_id,
            book_title=request.title,
            author=request.author,
            table_of_contents=toc,
            full_text=request.text
        )
        
        # Store session
        active_sessions[session_id] = conversation_state
        
        # Update agent's book processor
        agent.book_processor = book_processor
        
        return SessionResponse(
            session_id=session_id,
            book_title=request.title,
            author=request.author,
            table_of_contents=toc.dict() if toc else None
        )
        
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sessions/{session_id}/progress")
async def update_progress(session_id: str, request: ProgressUpdateRequest):
    """Update the user's reading progress."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session = active_sessions[session_id]
        
        # Get chapter title from TOC
        chapter_title = f"Chapter {request.chapter_number}"
        if session.table_of_contents:
            for chapter in session.table_of_contents.chapters:
                if chapter.get("number") == request.chapter_number:
                    chapter_title = chapter.get("title", chapter_title)
                    break
        
        # Update location
        new_location = BookLocation(
            chapter_number=request.chapter_number,
            chapter_title=chapter_title,
            page_number=request.page_number
        )
        
        session.update_reading_progress(new_location)
        
        return {"status": "success", "location": new_location.dict()}
        
    except Exception as e:
        logger.error(f"Error updating progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time conversation.
    
    This handles:
    - Message exchange between user and AI
    - Gesture streaming
    - Real-time updates
    """
    await websocket.accept()
    
    if session_id not in active_sessions:
        await websocket.send_json({
            "type": "error",
            "message": "Session not found"
        })
        await websocket.close()
        return
    
    session = active_sessions[session_id]
    
    try:
        # Send initial greeting
        await websocket.send_json({
            "type": "system",
            "message": "Connected to your literary companion. What chapter and page have you read up to?"
        })
        
        while True:
            # Receive message from user
            data = await websocket.receive_json()
            
            if data.get("type") == "message":
                user_message = data.get("content", "")
                
                # Process message through the agent
                try:
                    # Send typing indicator
                    await websocket.send_json({
                        "type": "typing",
                        "status": "start"
                    })
                    
                    # Process with agent
                    updated_session = await agent.process_message(
                        session,
                        user_message
                    )
                    
                    # Update stored session
                    active_sessions[session_id] = updated_session
                    
                    # Get the latest response
                    if updated_session.messages:
                        latest_message = updated_session.messages[-1]
                        
                        # Send the response
                        await websocket.send_json({
                            "type": "message",
                            "content": latest_message.content,
                            "gestures": [
                                {
                                    "type": g.type.value,
                                    "parameters": g.parameters,
                                    "duration": g.duration
                                }
                                for g in latest_message.gestures
                            ],
                            "citations": [
                                {
                                    "text": c.text,
                                    "location": {
                                        "chapter": c.location.chapter_number,
                                        "page": c.location.page_number
                                    }
                                }
                                for c in latest_message.citations
                            ]
                        })
                    
                    # Send typing indicator end
                    await websocket.send_json({
                        "type": "typing",
                        "status": "end"
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Failed to process message"
                    })
            
            elif data.get("type") == "ping":
                # Handle keepalive
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    return {
        "session_id": session_id,
        "book_title": session.book_title,
        "author": session.author,
        "current_location": session.current_location.dict() if session.current_location else None,
        "message_count": len(session.messages),
        "started_at": session.started_at.isoformat()
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del active_sessions[session_id]
    return {"status": "success"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_sessions": len(active_sessions),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )