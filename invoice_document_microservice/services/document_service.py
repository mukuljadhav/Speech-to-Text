# services/document_service.py
# Core Document Service orchestrator coordinating reading, inference, validation, and formatting
import time
from typing import Dict, Any

from services.document_reader import DocumentReader
from services.llm_service import LLMService
from services.validator import SchemaValidator
from services.response_formatter import ResponseFormatter
from schemas.response import ExtractionResponse
from config import settings

class DocumentService:
    def __init__(
        self,
        reader: DocumentReader,
        llm_service: LLMService,
        validator: SchemaValidator,
        formatter: ResponseFormatter
    ):
        self.reader = reader
        self.llm_service = llm_service
        self.validator = validator
        self.formatter = formatter

    async def process_document(
        self,
        file_bytes: bytes,
        mime_type: str,
        request_id: str
    ) -> ExtractionResponse:
        """
        Orchestrates the synchronous layered document extraction pipeline.
        
        Args:
            file_bytes (bytes): Raw uploaded file content.
            mime_type (str): MIME type format of the file.
            request_id (str): Generated transaction UUID.
            
        Returns:
            ExtractionResponse: Standardized successful extraction response object.
        """
        # 1. Start latency timer
        start_time = time.perf_counter()

        # 2. Extract plain text from document bytes (native or OCR fallback)
        raw_text = self.reader.read(file_bytes, mime_type)
        if not raw_text.strip():
            raise ValueError("No readable text could be extracted from the document.")

        # 3. Compile prompts and query local LLM
        raw_json_str = await self.llm_service.extract_json_from_text(raw_text)

        # 4. Clean, parse, and validate JSON output
        extracted_data = self.validator.parse_and_clean_json(raw_json_str)

        # 5. Measure latency
        processing_time = time.perf_counter() - start_time

        # 6. Extract dynamic model attributes (MIME types, document type, confidence)
        # Look for model-generated document_type or fallback to default
        document_type = extracted_data.get("document_type", "Unknown")
        confidence = extracted_data.get("confidence", 1.0)
        
        # Ensure confidence is a float type
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 1.0

        # Clean document_type and confidence from actual data if returned inside payload
        # (Optional, but cleaner to return them in top-level fields)
        if "document_type" in extracted_data:
            document_type = extracted_data["document_type"]
        if "confidence" in extracted_data:
            pass

        # 7. Formulate and return standardized response
        return self.formatter.format_success(
            request_id=request_id,
            document_type=document_type,
            model=settings.MODEL_NAME,
            processing_time=processing_time,
            confidence=confidence,
            extracted_data=extracted_data
        )
