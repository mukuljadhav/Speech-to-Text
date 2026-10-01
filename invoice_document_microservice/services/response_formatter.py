# services/response_formatter.py
# Standardized response formatter for success and error payloads
from typing import Dict, Any, Optional
from schemas.response import ExtractionResponse

class ResponseFormatter:
    @staticmethod
    def format_success(
        request_id: str,
        document_type: str,
        model: str,
        processing_time: float,
        confidence: float,
        extracted_data: Dict[str, Any]
    ) -> ExtractionResponse:
        """Constructs a standardized successful ExtractionResponse object."""
        return ExtractionResponse(
            success=True,
            request_id=request_id,
            document_type=document_type,
            model=model,
            processing_time=round(processing_time, 3),
            confidence=confidence,
            extracted_data=extracted_data
        )

    @staticmethod
    def format_error(
        request_id: str,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Constructs a standardized error payload dictionary matching the ErrorResponse schema."""
        return {
            "success": False,
            "request_id": request_id,
            "error_code": error_code,
            "message": message,
            "details": details or {}
        }
