# config.py
# Application Configurations using Pydantic Settings
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Model config
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemma2:2b")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    
    # OCR settings
    OCR_ENGINE: str = os.getenv("OCR_ENGINE", "tesseract")  # tesseract or easyocr
    
    # System settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "180"))
    
    # LLM generation parameters
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4096"))

    class Config:
        case_sensitive = True

settings = Settings()
