# tests/test_api.py
# API route integration tests for FastAPI routing controllers (Phase 4)
from fastapi.testclient import TestClient
import pytest
from unittest.mock import AsyncMock

from app import app
from api.routes import get_document_service
from schemas.response import ExtractionResponse

client = TestClient(app)

# Setup mock for DocumentService dependency override
mock_doc_service = AsyncMock()

@pytest.fixture(autouse=True)
def setup_dependency_overrides():
    # Inject dependency mock override
    app.dependency_overrides[get_document_service] = lambda: mock_doc_service
    yield
    # Clean up overrides after test run
    app.dependency_overrides.clear()

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "ollama_status" in data
    assert data["loaded_model"] == "gemma2:2b"
    assert "X-Request-ID" in response.headers

def test_model_endpoint():
    response = client.get("/api/v1/model")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "gemma2:2b"
    assert data["provider"] == "Ollama"
    assert "X-Request-ID" in response.headers

def test_extract_endpoint_success():
    # Setup mock service response
    mock_doc_service.process_document.return_value = ExtractionResponse(
        success=True,
        request_id="test-req-id",
        document_type="Invoice",
        model="gemma2:2b",
        processing_time=1.25,
        confidence=0.95,
        extracted_data={"total_amount": 250.00}
    )

    files = {"file": ("test.txt", b"Mock document content", "text/plain")}
    response = client.post("/api/v1/extract", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["document_type"] == "Invoice"
    assert data["extracted_data"] == {"total_amount": 250.00}
    assert "X-Request-ID" in response.headers

def test_extract_endpoint_unsupported_type():
    files = {"file": ("test.html", b"<html></html>", "text/html")}
    response = client.post("/api/v1/extract", files=files)
    
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error_code"] == "INVALID_DOCUMENT_TYPE"
    assert "X-Request-ID" in response.headers

def test_extract_endpoint_file_size_exceeded():
    # Submit file exceeding MAX_FILE_SIZE_BYTES (10MB limit)
    # Generate 11MB file buffer in memory
    large_buffer = b"x" * (11 * 1024 * 1024)
    files = {"file": ("large.txt", large_buffer, "text/plain")}
    response = client.post("/api/v1/extract", files=files)
    
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error_code"] == "FILE_SIZE_EXCEEDED"
    assert "X-Request-ID" in response.headers

def test_extract_endpoint_failed_parsing():
    # Simulate extraction crash
    mock_doc_service.process_document.side_effect = Exception("Ollama request timed out")
    
    files = {"file": ("test.txt", b"Mock document content", "text/plain")}
    response = client.post("/api/v1/extract", files=files)
    
    assert response.status_code == 500
    data = response.json()
    assert data["success"] is False
    assert data["error_code"] == "OLLAMA_TIMEOUT"
    assert "X-Request-ID" in response.headers
