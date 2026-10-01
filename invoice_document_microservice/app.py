# app.py
# FastAPI Application Configuration & Logging Middleware
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import time
import uuid
import logging

from config import settings
from api.routes import router

# Configure logging format and level based on configuration settings
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("document-extraction-service")

app = FastAPI(
    title="Document Extraction Microservice",
    version="1.0.0",
    description="Synchronous REST API for extracting structured information from documents locally."
)

# Custom Middleware for Request ID generation, latency metrics, and safe transaction logging
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    # 1. Generate a unique transaction request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # 2. Start execution performance timer
    start_time = time.perf_counter()
    
    # 3. Proceed down the routing/middleware chain
    try:
        response: Response = await call_next(request)
    except Exception as exc:
        # Fallback to catch unhandled errors inside the call chain
        logger.error(f"Unhandled exception caught in middleware: {str(exc)}", exc_info=True)
        response = JSONResponse(
            status_code=500,
            content={
                "success": False,
                "request_id": request_id,
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred during processing.",
                "details": {}
            }
        )
    
    # 4. End execution performance timer
    process_time = time.perf_counter() - start_time
    
    # 5. Inject transactional response headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{process_time:.6f}"
    
    # 6. Extract telemetry metadata safely (no file content is logged)
    document_type = getattr(request.state, "document_type", "Unknown")
    model_used = getattr(request.state, "model_used", settings.MODEL_NAME)
    
    logger.info(
        f"Request: {request_id} | Path: {request.url.path} | Method: {request.method} | "
        f"Status: {response.status_code} | Time: {process_time:.4f}s | "
        f"DocType: {document_type} | Model: {model_used}"
    )
    
    return response

# Custom Exception Handlers for standardizing API error responses

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    error_code = "HTTP_ERROR"
    message = str(exc.detail)
    details = {}
    
    # Extract details if they were packed into a dictionary representation
    if isinstance(exc.detail, dict):
        error_code = exc.detail.get("error_code", "HTTP_ERROR")
        message = exc.detail.get("message", message)
        details = exc.detail.get("details", {})

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "request_id": request_id,
            "error_code": error_code,
            "message": message,
            "details": details
        },
        headers={"X-Request-ID": request_id}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "request_id": request_id,
            "error_code": "REQUEST_VALIDATION_FAILED",
            "message": "The request body validation failed.",
            "details": {"errors": exc.errors()}
        },
        headers={"X-Request-ID": request_id}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.error(f"Unhandled error | Request ID: {request_id} | Message: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "request_id": request_id,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "A critical system error occurred.",
            "details": {}
        },
        headers={"X-Request-ID": request_id}
    )

# Include routes under prefix defined in API router
app.include_router(router)
