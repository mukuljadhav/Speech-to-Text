"""
Centralized configuration for the Campaign Management AI Service.

Everything that might change between machines (Ollama URL, model name,
output folder, banner size) lives here and is loaded from the .env file.
This means later, when this service is deployed elsewhere, nobody has to
touch the Python code -- they just edit .env.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Ollama server that hosts Gemma 4 locally
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    GEMMA_MODEL_NAME: str = "gemma4:e2b"

    # Where generated banners are saved
    OUTPUT_DIR: str = "output"
    BANNER_WIDTH: int = 1080
    BANNER_HEIGHT: int = 1080
    BANNER_FORMAT: str = "PNG"  # PNG or JPEG

    # --- Voice input settings (Phase 2) ---
    # tiny / base / small / medium / large-v3 -- bigger = more accurate, slower on CPU
    WHISPER_MODEL_SIZE: str = "base"
    TEMP_AUDIO_DIR: str = "temp_audio"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Single shared instance imported everywhere else in the app
settings = Settings()