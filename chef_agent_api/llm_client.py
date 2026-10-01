# llm_client.py
# Async Client Wrapper to interface with local Ollama Server
import httpx
import json
from config import settings
from typing import Dict, Any

class LLMClient:
    def __init__(self):
        # Clean host URL to ensure no trailing slash
        self.host = settings.OLLAMA_URL.rstrip('/')
        self.model = settings.MODEL_NAME
        self.timeout = settings.REQUEST_TIMEOUT
        self.temperature = settings.TEMPERATURE

    async def generate_json(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """
        Sends an asynchronous generation request to the Ollama server.
        Forces JSON mode and parses the response to a dictionary.
        """
        url = f"{self.host}/api/generate"
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "format": "json",  # Force JSON schema output
            "options": {
                "temperature": self.temperature
            }
        }
        headers = {"Content-Type": "application/json"}
        
        try:
            async with httpx.AsyncClient(timeout=float(self.timeout)) as client:
                response = await client.post(url, json=payload, headers=headers)
                
                if response.status_code == 404:
                    raise Exception(
                        f"Model '{self.model}' was not found on your local Ollama server. "
                        f"Please run 'ollama pull {self.model}' to download it."
                    )
                elif response.status_code != 200:
                    raise Exception(f"Ollama server returned error status {response.status_code}: {response.text}")
                
                response_json = response.json()
                response_text = response_json.get("response", "").strip()
                
                if not response_text:
                    raise Exception("Ollama server returned an empty response.")
                
                # Deserialization with fallback cleanup
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    # Strip markdown blocks if the LLM outputted code fence block wrappers
                    cleaned = response_text
                    if cleaned.startswith("```json"):
                        cleaned = cleaned[7:]
                    elif cleaned.startswith("```"):
                        cleaned = cleaned[3:]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]
                    
                    return json.loads(cleaned.strip())

        except httpx.ConnectError as exc:
            raise Exception(
                f"Failed to connect to local Ollama server at '{self.host}'. "
                "Ensure Ollama is running and accessible."
            ) from exc
        except httpx.ReadTimeout as exc:
            raise Exception(f"Ollama inference timed out after {self.timeout} seconds.") from exc
        except ValueError as exc:
            raise Exception(f"Failed to parse JSON response from local Ollama: {str(exc)}") from exc
