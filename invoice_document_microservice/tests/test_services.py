# tests/test_services.py
# Unit tests for Phase 3: DocumentReader, SchemaValidator, ResponseFormatter, and DocumentService
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from services.document_reader import DocumentReader
from services.validator import SchemaValidator, JSONValidationError
from services.response_formatter import ResponseFormatter
from services.document_service import DocumentService
from services.llm_service import LLMService

# --- DOCUMENT READER TESTS ---

def test_document_reader_plain_text():
    reader = DocumentReader()
    text = "Hello Plain Text World"
    result = reader.read(text.encode("utf-8"), "text/plain")
    assert result == text

@patch("fitz.open")
def test_document_reader_digital_pdf(mock_fitz_open):
    # Set up mock PDF document with page text
    mock_doc = MagicMock()
    mock_page1 = MagicMock()
    # Mock text must be longer than 50 characters to avoid falling back to OCR
    mock_page1.get_text.return_value = "Page 1 content text: This is a very long string designed to exceed the native PDF character threshold of fifty bytes."
    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "Page 2 content text: Another long string to ensure native text extraction fast-path works correctly without triggering Tesseract."
    
    mock_doc.__iter__.return_value = [mock_page1, mock_page2]
    mock_doc.__len__.return_value = 2
    mock_fitz_open.return_value = mock_doc

    reader = DocumentReader()
    result = reader.read(b"dummy_bytes", "application/pdf")
    
    assert "Page 1 content text" in result
    assert "Page 2 content text" in result
    mock_fitz_open.assert_called_once()


# --- VALIDATOR TESTS ---

def test_validator_markdown_fence_cleaning():
    validator = SchemaValidator()
    raw_response = "```json\n{\n  \"document_type\": \"Invoice\"\n}\n```"
    result = validator.parse_and_clean_json(raw_response)
    assert result == {"document_type": "Invoice"}

def test_validator_conversational_text_extraction():
    validator = SchemaValidator()
    raw_response = "Sure, here is the data:\n{\n  \"document_type\": \"Receipt\"\n}\nHope this helps!"
    result = validator.parse_and_clean_json(raw_response)
    assert result == {"document_type": "Receipt"}

def test_validator_invalid_json():
    validator = SchemaValidator()
    # Malformed JSON (contains brackets but has invalid trailing comma/syntax)
    raw_response = "{\n  \"document_type\": \"Invoice\",\n}"
    with pytest.raises(JSONValidationError) as exc_info:
        validator.parse_and_clean_json(raw_response)
    assert "Invalid JSON format" in str(exc_info.value)

def test_validator_no_brackets():
    validator = SchemaValidator()
    # Missing braces altogether
    raw_response = "No JSON here"
    with pytest.raises(JSONValidationError) as exc_info:
        validator.parse_and_clean_json(raw_response)
    assert "Could not locate JSON block brackets" in str(exc_info.value)

def test_validator_non_dict_object():
    validator = SchemaValidator()
    # Returns an array which does not contain dictionary curly braces
    raw_response = "[1, 2, 3]"
    with pytest.raises(JSONValidationError) as exc_info:
        validator.parse_and_clean_json(raw_response)
    assert "Could not locate JSON block brackets" in str(exc_info.value)


# --- RESPONSE FORMATTER TESTS ---

def test_response_formatter_success():
    formatter = ResponseFormatter()
    result = formatter.format_success(
        request_id="req-123",
        document_type="Invoice",
        model="gemma2:2b",
        processing_time=1.8249,
        confidence=0.97,
        extracted_data={"invoice_number": "INV-100"}
    )
    assert result.success is True
    assert result.request_id == "req-123"
    assert result.document_type == "Invoice"
    assert result.processing_time == 1.825  # Rounded to 3 decimals
    assert result.confidence == 0.97
    assert result.extracted_data == {"invoice_number": "INV-100"}

def test_response_formatter_error():
    formatter = ResponseFormatter()
    result = formatter.format_error(
        request_id="req-123",
        error_code="INVALID_DOCUMENT",
        message="Failed to parse document"
    )
    assert result["success"] is False
    assert result["request_id"] == "req-123"
    assert result["error_code"] == "INVALID_DOCUMENT"
    assert result["message"] == "Failed to parse document"


# --- DOCUMENT SERVICE ORCHESTRATION TESTS ---

@pytest.mark.anyio
async def test_document_service_orchestration():
    # Setup mocks
    mock_reader = MagicMock()
    mock_reader.read.return_value = "Extracted Text"
    
    mock_llm_service = AsyncMock()
    mock_llm_service.extract_json_from_text.return_value = '{"document_type": "Invoice", "confidence": 0.95}'
    
    mock_validator = SchemaValidator()
    mock_formatter = ResponseFormatter()
    
    service = DocumentService(
        reader=mock_reader,
        llm_service=mock_llm_service,
        validator=mock_validator,
        formatter=mock_formatter
    )
    
    result = await service.process_document(
        file_bytes=b"dummy_pdf_bytes",
        mime_type="application/pdf",
        request_id="req-123"
    )
    
    assert result.success is True
    assert result.document_type == "Invoice"
    assert result.confidence == 0.95
    assert result.extracted_data == {"document_type": "Invoice", "confidence": 0.95}
    
    # Verify sequence calls
    mock_reader.read.assert_called_once_with(b"dummy_pdf_bytes", "application/pdf")
    mock_llm_service.extract_json_from_text.assert_called_once_with("Extracted Text")
