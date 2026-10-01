# config.py
# Application Configurations using Pydantic Settings
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Model config (defaults to gemma2:2b, matching Ollama standard)
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemma2:2b")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    
    # Cache SQLite database file path
    DB_PATH: str = os.getenv("DB_PATH", "chef_agent.db")
    
    # System settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "240"))
    TEMPERATURE: float = 0.2

    class Config:
        case_sensitive = True

settings = Settings()
