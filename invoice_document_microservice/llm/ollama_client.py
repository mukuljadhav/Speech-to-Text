# llm/ollama_client.py
# Concrete implementation of BaseLLM for local Ollama server connectivity
import httpx
from llm.base_llm import BaseLLM
from llm.exceptions import (
    LLMConnectionError,
    LLMTimeoutError,
    LLMModelNotFoundError,
    LLMServerError
)

class OllamaClient(BaseLLM):
    def __init__(self, host: str, model: str, timeout: int = 60, temperature: float = 0.0, max_tokens: int = 4096):
        self.host = host.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        url = f"{self.host}/api/generate"
        
        # Prepare request payload conforming to Ollama generate API structure
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",  # Forces the model to output syntactically valid JSON
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt

        headers = {"Content-Type": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=float(self.timeout)) as client:
                response = await client.post(url, json=payload, headers=headers)
                
                # Check HTTP error status codes
                if response.status_code == 404:
                    raise LLMModelNotFoundError(
                        f"Model '{self.model}' was not found on the local Ollama server. "
                        f"Run 'ollama pull {self.model}' to download it."
                    )
                elif response.status_code >= 500:
                    raise LLMServerError(
                        f"Local Ollama server returned internal server error ({response.status_code}): {response.text}"
                    )
                elif response.status_code != 200:
                    raise LLMServerError(
                        f"Unexpected error from local Ollama server ({response.status_code}): {response.text}"
                    )
                
                response_json = response.json()
                return response_json.get("response", "")

        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                f"Failed to connect to local Ollama server at '{self.host}'. "
                "Ensure Ollama is running and accessible."
            ) from exc
        except httpx.ReadTimeout as exc:
            raise LLMTimeoutError(
                f"Local LLM inference timed out after {self.timeout} seconds."
            ) from exc
        except httpx.HTTPError as exc:
            # Wrap generic HTTP exception if not handled above
            raise LLMServerError(f"Local Ollama API request failed: {str(exc)}") from exc
        except ValueError as exc:
            raise LLMServerError(f"Failed to parse JSON response from local Ollama: {str(exc)}") from exc
