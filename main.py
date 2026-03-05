# main.py - FastAPI Integration for UrRight
# Following Module 4 from Gamma presentation

import os
import logging
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn
from dotenv import load_dotenv

# Import our agentic RAG
from agentic_rag import UrRightAgent, conversation_memory

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# Section 4.1: Pydantic Models
# ============================================

class Source(BaseModel):
    """Source document reference"""
    article: str
    excerpt: Optional[str] = None
    relevance: Optional[float] = None

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., min_length=1, description="The user's message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    language: Optional[str] = Field(None, description="Preferred language (en/sw/ki)")
    model: Optional[str] = Field(None, description="Groq model to use")
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()
    
    @field_validator('language')
    @classmethod
    def validate_language(cls, v):
        if v is not None and v not in ['en', 'sw', 'ki']:
            raise ValueError("Language must be 'en', 'sw', or 'ki'")
        return v

class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    sources: List[str] = []
    session_id: str
    timestamp: str
    request_id: str
    status: str = "success"
    language: str
    reasoning: Optional[str] = None
    chunks_retrieved: int = 0

class SessionInfo(BaseModel):
    """Session information"""
    session_id: str
    message_count: int
    last_active: str
    created_at: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str = "1.0.0"
    model: str
    constitution_indexed: bool
    active_sessions: int
    timestamp: str

# ============================================
# Section 4.2: Global State
# ============================================

# Global agent instance
agent = None
app_start_time = None

# Available models
AVAILABLE_MODELS = {
    "llama-3.3-70b-versatile": "Production-ready, best for complex reasoning",
    "llama-3.1-8b-instant": "Fast, good for simple queries",
    "llama3-70b-8192": "High quality, good for detailed analysis"
}

# ============================================
# Section 4.3: Lifespan and Initialization
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown events
    Following Gamma presentation Section 4.3
    """
    global agent, app_start_time
    
    app_start_time = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("🚀 UrRight API starting up...")
    logger.info("=" * 60)
    
    # Initialize agent
    try:
        agent = UrRightAgent()
        logger.info("✅ Agent initialized successfully")
        
        # Check if constitution is indexed
        test_search = agent.constitution_tool.search("rights", n_results=1)
        if "error" in test_search or not test_search.get("results"):
            logger.warning("⚠️  Constitution not indexed! Please run app.py first to index.")
        else:
            logger.info(f"✅ Constitution indexed with {len(test_search['results'])} chunks available")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize agent: {e}")
        agent = None
    
    logger.info("✅ API ready to handle requests")
    
    yield
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down UrRight API...")
    # Clear memory
    for session in conversation_memory.get_all_sessions():
        conversation_memory.clear_session(session)
    logger.info("✅ Cleanup complete")

# ============================================
# Section 4.4: FastAPI App
# ============================================

app = FastAPI(
    title="UrRight - Kenyan Constitution Chatbot",
    description="An AI assistant that helps citizens understand their constitutional rights",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Section 4.5: API Endpoints
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "UrRight - Kenyan Constitution Chatbot",
        "version": "1.0.0",
        "description": "Ask questions about the Kenyan Constitution (2010)",
        "endpoints": {
            "/chat": "POST - Send a message",
            "/chat/history/{session_id}": "GET - Get chat history",
            "/sessions": "GET - List active sessions",
            "/health": "GET - Health check",
            "/models": "GET - Available models",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint"""
    if agent is None:
        status = "degraded"
        model = "not_initialized"
        indexed = False
    else:
        status = "healthy"
        model = agent.model
        # Check if constitution is indexed
        test_search = agent.constitution_tool.search("test", n_results=1)
        indexed = not ("error" in test_search or not test_search.get("results"))
    
    return HealthResponse(
        status=status,
        model=model,
        constitution_indexed=indexed,
        active_sessions=len(conversation_memory.get_all_sessions()),
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.get("/models", tags=["System"])
async def get_models():
    """Get available Groq models"""
    return {
        "current": agent.model if agent else "unknown",
        "available": AVAILABLE_MODELS
    }

# ===== UPDATED CHAT ENDPOINT =====
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Main chat endpoint - send a message to UrRight
    Now supports language parameter (en/sw/ki)
    """
    request_id = str(uuid.uuid4())
    logger.info(f"📨 Request {request_id}: {request.message[:50]}... (lang: {request.language})")
    
    # Check if agent is initialized
    if agent is None:
        logger.error(f"❌ Agent not initialized for request {request_id}")
        raise HTTPException(
            status_code=503,
            detail="Service not fully initialized. Please try again in a moment."
        )
    
    # Generate or use session ID
    session_id = request.session_id or f"session_{uuid.uuid4()}"
    
    try:
        # Override model if specified
        if request.model and request.model in AVAILABLE_MODELS:
            agent.model = request.model
            logger.info(f"Using model: {request.model}")
        
        # Process query with agent, passing the requested language
        result = agent.process_query(
            query=request.message, 
            session_id=session_id,
            requested_language=request.language  # ← Pass the language to agent
        )
        
        # Prepare response
        response = ChatResponse(
            response=result["response"],
            sources=result.get("sources", []),
            session_id=result["session_id"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            request_id=request_id,
            language=result.get("language", "english"),
            reasoning=result.get("reasoning"),
            chunks_retrieved=result.get("chunks_retrieved", 0)
        )
        
        logger.info(f"✅ Request {request_id} completed successfully in {result.get('language', 'english')}")
        return response
        
    except asyncio.TimeoutError:
        logger.error(f"⏰ Timeout for request {request_id}")
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        logger.error(f"❌ Error processing request {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/chat/history/{session_id}", tags=["History"])
async def get_chat_history(session_id: str):
    """
    Get conversation history for a session
    Following Gamma presentation Section 4.5
    """
    try:
        # Validate session ID format
        if not session_id.startswith("session_"):
            # Convert to our format if needed
            session_id = f"session_{session_id}"
        
        # Get context from memory
        context = conversation_memory.get_context(session_id)
        
        if not context:
            return JSONResponse(
                status_code=404,
                content={"error": f"No history found for session {session_id}"}
            )
        
        # Parse memory into messages
        messages = []
        if session_id in conversation_memory.memories:
            for msg in conversation_memory.memories[session_id]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg["timestamp"]
                })
        
        return {
            "session_id": session_id,
            "message_count": len(messages),
            "messages": messages
        }
        
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/history/{session_id}", tags=["History"])
async def clear_chat_history(session_id: str):
    """Clear conversation history for a session"""
    try:
        if not session_id.startswith("session_"):
            session_id = f"session_{session_id}"
        
        conversation_memory.clear_session(session_id)
        return {"status": "success", "message": f"Session {session_id} cleared"}
        
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions", tags=["History"])
async def list_sessions():
    """List all active sessions"""
    sessions = []
    for session_id in conversation_memory.get_all_sessions():
        messages = conversation_memory.memories.get(session_id, [])
        sessions.append({
            "session_id": session_id,
            "message_count": len(messages),
            "last_active": messages[-1]["timestamp"] if messages else None
        })
    
    return {
        "total_sessions": len(sessions),
        "sessions": sessions
    }

@app.post("/chat/reset/{session_id}", tags=["Chat"])
async def reset_session(session_id: str):
    """Reset a conversation session"""
    try:
        if not session_id.startswith("session_"):
            session_id = f"session_{session_id}"
        
        conversation_memory.clear_session(session_id)
        return {
            "status": "success",
            "message": f"Session {session_id} reset",
            "new_session_id": f"session_{uuid.uuid4()}"
        }
        
    except Exception as e:
        logger.error(f"Error resetting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# Section 4.6: Error Handlers
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

# ============================================
# Section 4.7: Run the API
# ============================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8516))
    host = os.getenv("HOST", "0.0.0.0")
    
    print("\n" + "="*60)
    print("🌐 Starting UrRight API Server")
    print("="*60)
    print(f"📡 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"📚 API Docs: http://localhost:{port}/docs")
    print(f"🏥 Health Check: http://localhost:{port}/health")
    print("="*60 + "\n")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True  # Auto-reload on code changes
    )