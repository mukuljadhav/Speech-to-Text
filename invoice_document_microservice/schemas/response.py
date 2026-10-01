# schemas/response.py
# API standard response models using Pydantic v2
from pydantic import BaseModel, Field
from typing import Any, Optional, Dict

class ExtractionResponse(BaseModel):
    success: bool = Field(..., description="Flag indicating if the operation succeeded")
    request_id: str = Field(..., description="Unique ID for tracking the extraction request")
    document_type: Optional[str] = Field(None, description="Automatically detected document type")
    model: Optional[str] = Field(None, description="Local LLM model name used for processing")
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds")
    confidence: Optional[float] = Field(None, description="Confidence level score")
    extracted_data: Optional[Dict[str, Any]] = Field(None, description="Extracted dynamic structured JSON data")

class ErrorResponse(BaseModel):
    success: bool = Field(False, description="Flag indicating if the operation failed")
    request_id: str = Field(..., description="Unique ID for tracking the extraction request")
    error_code: str = Field(..., description="Standardized error code string")
    message: str = Field(..., description="Error message details")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional context about the error")

class HealthResponse(BaseModel):
    status: str = Field(..., description="API operational status")
    ollama_status: str = Field(..., description="Ollama connection operational status")
    loaded_model: str = Field(..., description="Active configured model name")
    version: str = Field(..., description="Microservice application version")

class ModelResponse(BaseModel):
    model: str = Field(..., description="Current configured model name")
    provider: str = Field(..., description="LLM provider name")
    runtime: str = Field(..., description="Inference runtime system name")
    version: str = Field(..., description="Model info/runtime version metadata")
