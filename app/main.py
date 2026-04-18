import os
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager


from app.db.session import init_db
from app.schemas.response import APIResponse

# Import Routers (We will create these next)
from app.api import onboard, chat, analysis

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # 1. Create DB Tables (SQLModel)
        init_db()
        print("Database initialized successfully.")
        
        # 2. Create Agent Storage (SQLite for Agno memory)
        if not os.path.exists("agent_storage"):
            os.makedirs("agent_storage")
            print("Created 'agent_storage' directory.")
            
    except Exception as e:
        print(f"Startup Failure: {e}")
    yield

app = FastAPI(title="AI Psychiatrist", version="1.0.0", lifespan=lifespan)

# --- CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development, restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL EXCEPTION HANDLERS ---
# These ensure every error returns your custom JSON format

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    response = APIResponse[None](
        ErrorCode=exc.status_code,
        Data=None,
        Message=str(exc.detail)
    )
    return JSONResponse(status_code=exc.status_code, content=response.model_dump())

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Formats Pydantic validation errors into a readable string
    error_msg = "; ".join([f"{e['loc'][-1]}: {e['msg']}" for e in exc.errors()])
    response = APIResponse[str](
        ErrorCode=422,
        Data=str(exc.errors()), 
        Message=f"Validation Error: {error_msg}"
    )
    return JSONResponse(status_code=422, content=response.model_dump())

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    response = APIResponse[None](
        ErrorCode=500,
        Data=None,
        Message=f"Internal Server Error: {str(exc)}"
    )
    return JSONResponse(status_code=500, content=response.model_dump())

# --- ROUTERS ---
# Grouping endpoints logically
app.include_router(onboard.router, prefix="/api/v1/onboard", tags=["Onboarding"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat Session"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Mood Analysis"])

@app.get("/health", response_model=APIResponse[Dict[str, Any]])
def health():
    return APIResponse[Dict[str, Any]](
        ErrorCode=0,
        Data={"status": "active", "db": "connected"},
        Message="System Healthy"
    )