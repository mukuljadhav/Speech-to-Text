# api/routes.py
# API route controllers for microservice endpoints
from fastapi import APIRouter, UploadFile, File, Request, HTTPException, Depends
import httpx
import uuid
import os

from config import settings
from constants import API_PREFIX, APP_VERSION, MAX_FILE_SIZE_BYTES, SUPPORTED_FILE_TYPES
from schemas.response import ExtractionResponse, HealthResponse, ModelResponse

# Import layers
from llm.ollama_client import OllamaClient
from services.prompt_manager import PromptManager
from services.llm_service import LLMService
from services.document_reader import DocumentReader
from services.validator import SchemaValidator
from services.response_formatter import ResponseFormatter
from services.document_service import DocumentService

router = APIRouter(prefix=API_PREFIX)

# --- DEPENDENCY INJECTION PROVIDERS ---

def get_document_reader() -> DocumentReader:
    return DocumentReader()

def get_ollama_client() -> OllamaClient:
    return OllamaClient(
        host=settings.OLLAMA_URL,
        model=settings.MODEL_NAME,
        timeout=settings.REQUEST_TIMEOUT,
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS
    )

def get_prompt_manager() -> PromptManager:
    # Resolve prompts directory relative to application root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompts_dir = os.path.join(base_dir, "prompts")
    return PromptManager(prompts_dir=prompts_dir)

def get_llm_service(
    client: OllamaClient = Depends(get_ollama_client),
    prompt_manager: PromptManager = Depends(get_prompt_manager)
) -> LLMService:
    return LLMService(llm_client=client, prompt_manager=prompt_manager)

def get_validator() -> SchemaValidator:
    return SchemaValidator()

def get_formatter() -> ResponseFormatter:
    return ResponseFormatter()

def get_document_service(
    reader: DocumentReader = Depends(get_document_reader),
    llm_service: LLMService = Depends(get_llm_service),
    validator: SchemaValidator = Depends(get_validator),
    formatter: ResponseFormatter = Depends(get_formatter)
) -> DocumentService:
    return DocumentService(
        reader=reader,
        llm_service=llm_service,
        validator=validator,
        formatter=formatter
    )


# --- HTTP CONTROLLERS ---

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check API operational health and local Ollama server connectivity.
    """
    ollama_status = "disconnected"
    try:
        # Check connection status of local Ollama instance (timeout after 1 second)
        async with httpx.AsyncClient() as client:
            response = await client.get(settings.OLLAMA_URL, timeout=1.0)
            if response.status_code == 200:
                ollama_status = "connected"
    except Exception:
        # Catch and report disconnected if Ollama is unreachable
        pass

    return HealthResponse(
        status="healthy",
        ollama_status=ollama_status,
        loaded_model=settings.MODEL_NAME,
        version=APP_VERSION
    )

@router.get("/model", response_model=ModelResponse)
async def model_info():
    """
    Check the active configured model information.
    """
    return ModelResponse(
        model=settings.MODEL_NAME,
        provider="Ollama",
        runtime="local",
        version="0.1.48"
    )

@router.post("/extract", response_model=ExtractionResponse)
async def extract_document(
    request: Request,
    file: UploadFile = File(...),
    doc_service: DocumentService = Depends(get_document_service)
):
    """
    Synchronous document data extraction endpoint.
    Ingests PDF, Image, or Plain Text, and returns dynamic structured JSON.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    # 1. File Size Verification (Before loading bytes in-memory)
    # UploadFile file descriptor does not expose length directly without reading
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "FILE_SIZE_EXCEEDED",
                "message": f"File size exceeds the maximum limit of {MAX_FILE_SIZE_BYTES / (1024*1024):.0f}MB.",
                "details": {"file_size_bytes": len(file_bytes), "max_size_bytes": MAX_FILE_SIZE_BYTES}
            }
        )
    
    # 2. MIME Type Format Verification
    if file.content_type not in SUPPORTED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "INVALID_DOCUMENT_TYPE",
                "message": f"The file format '{file.content_type}' is not supported.",
                "details": {"supported_types": list(SUPPORTED_FILE_TYPES)}
            }
        )

    # 3. Execution Pipeline
    try:
        # Update logging metadata context for the request logging middleware
        request.state.model_used = settings.MODEL_NAME
        
        result = await doc_service.process_document(
            file_bytes=file_bytes,
            mime_type=file.content_type,
            request_id=request_id
        )
        
        # Set detected document type for middleware logger
        request.state.document_type = result.document_type
        
        return result
        
    except Exception as exc:
        # Map known connection, timeout, and model errors into structured HTTP 500 exceptions
        error_code = "EXTRACTION_FAILED"
        error_msg = str(exc)
        
        # Capture signature terms inside errors to classify error codes
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            error_code = "OLLAMA_TIMEOUT"
            error_msg = f"The local LLM inference request timed out after {settings.REQUEST_TIMEOUT} seconds."
        elif "connect" in error_msg.lower() or "unreachable" in error_msg.lower():
            error_code = "OLLAMA_CONNECTION_FAILED"
            error_msg = "Cannot establish connection to local Ollama server."
        elif "not found" in error_msg.lower():
            error_code = "OLLAMA_MODEL_NOT_FOUND"
        elif "json" in error_msg.lower() or "brackets" in error_msg.lower():
            error_code = "LLM_JSON_PARSING_FAILED"
            
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": error_code,
                "message": error_msg,
                "details": {}
            }
        )
