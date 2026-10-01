# tests/test_llm.py
# Unit and integration tests for LLM abstraction layer and prompt management
import pytest
import os
import tempfile
import httpx
from unittest.mock import patch, MagicMock

from llm.ollama_client import OllamaClient
from llm.exceptions import (
    LLMConnectionError,
    LLMTimeoutError,
    LLMModelNotFoundError,
    LLMServerError
)
from services.prompt_manager import PromptManager

# --- PROMPT MANAGER TESTS ---

def test_prompt_manager_loading_and_formatting():
    # Use temporary files to test loading
    with tempfile.TemporaryDirectory() as temp_dir:
        system_path = os.path.join(temp_dir, "system_prompt.txt")
        extraction_path = os.path.join(temp_dir, "extraction_v1.txt")
        
        with open(system_path, "w", encoding="utf-8") as f:
            f.write("System instruction rules")
        with open(extraction_path, "w", encoding="utf-8") as f:
            f.write("User guidelines: {document_text}")

        pm = PromptManager(prompts_dir=temp_dir)
        
        # Test loaded strings
        assert pm.get_system_prompt() == "System instruction rules"
        
        # Test formatting
        doc_text = "This is a document sample"
        user_prompt = pm.build_user_prompt(doc_text)
        assert user_prompt == f"User guidelines: {doc_text}"

def test_prompt_manager_fallbacks():
    # Constructing manager on non-existent path triggers fallback defaults
    pm = PromptManager(prompts_dir="/non-existent-directory-path")
    assert "document data extraction" in pm.get_system_prompt()
    assert "Perform data extraction" in pm.build_user_prompt("sample")

def test_prompt_manager_safe_formatting_braces():
    # Asserts that curly braces inside the raw document text do not break formatting
    with tempfile.TemporaryDirectory() as temp_dir:
        extraction_path = os.path.join(temp_dir, "extraction_v1.txt")
        with open(extraction_path, "w", encoding="utf-8") as f:
            f.write("Extract: {document_text}")

        pm = PromptManager(prompts_dir=temp_dir)
        raw_doc_with_braces = "Doc containing {JSON} with raw curly braces"
        user_prompt = pm.build_user_prompt(raw_doc_with_braces)
        assert user_prompt == f"Extract: {raw_doc_with_braces}"


# --- OLLAMA CLIENT TESTS ---

@pytest.mark.anyio
@patch("httpx.AsyncClient.post")
async def test_ollama_client_success(mock_post):
    # Set up real httpx.Response for successful generation
    mock_response = httpx.Response(
        status_code=200,
        json={"response": '{"document_type": "Invoice"}'}
    )
    mock_post.return_value = mock_response

    client = OllamaClient(host="http://localhost:11434", model="gemma2:2b")
    result = await client.generate(prompt="dummy_prompt", system_prompt="dummy_sys")

    assert result == '{"document_type": "Invoice"}'
    mock_post.assert_called_once()
    
    # Assert correct parameters were sent in payload
    args, kwargs = mock_post.call_args
    payload = kwargs.get("json", {})
    assert payload["model"] == "gemma2:2b"
    assert payload["prompt"] == "dummy_prompt"
    assert payload["system"] == "dummy_sys"
    assert payload["stream"] is False
    assert payload["format"] == "json"

@pytest.mark.anyio
@patch("httpx.AsyncClient.post")
async def test_ollama_client_connection_error(mock_post):
    # Mock connection failure
    mock_post.side_effect = httpx.ConnectError("Connection refused")

    client = OllamaClient(host="http://localhost:11434", model="gemma2:2b")
    with pytest.raises(LLMConnectionError) as exc_info:
        await client.generate(prompt="dummy")
    assert "Ensure Ollama is running" in str(exc_info.value)

@pytest.mark.anyio
@patch("httpx.AsyncClient.post")
async def test_ollama_client_timeout_error(mock_post):
    # Mock read timeout
    mock_post.side_effect = httpx.ReadTimeout("Request timed out")

    client = OllamaClient(host="http://localhost:11434", model="gemma2:2b", timeout=5)
    with pytest.raises(LLMTimeoutError) as exc_info:
        await client.generate(prompt="dummy")
    assert "timed out after 5 seconds" in str(exc_info.value)

@pytest.mark.anyio
@patch("httpx.AsyncClient.post")
async def test_ollama_client_model_not_found(mock_post):
    # Mock model missing (404 Not Found)
    mock_response = httpx.Response(status_code=404)
    mock_post.return_value = mock_response

    client = OllamaClient(host="http://localhost:11434", model="non-existent")
    with pytest.raises(LLMModelNotFoundError) as exc_info:
        await client.generate(prompt="dummy")
    # Verify model is reported missing (case-insensitive check)
    assert "model 'non-existent' was not found" in str(exc_info.value).lower()

@pytest.mark.anyio
@patch("httpx.AsyncClient.post")
async def test_ollama_client_server_error(mock_post):
    # Mock internal server error (500)
    mock_response = httpx.Response(status_code=500, text="Internal Engine Crash")
    mock_post.return_value = mock_response

    client = OllamaClient(host="http://localhost:11434", model="gemma2:2b")
    with pytest.raises(LLMServerError) as exc_info:
        await client.generate(prompt="dummy")
    assert "internal server error (500)" in str(exc_info.value).lower()
