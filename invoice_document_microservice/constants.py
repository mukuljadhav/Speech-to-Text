# constants.py
# Global application constants

API_PREFIX = "/api/v1"
APP_VERSION = "1.0.0"

# File size and validation limits
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit for uploaded documents
SUPPORTED_FILE_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/tiff",
    "text/plain"
}

# Inference defaults
DEFAULT_TIMEOUT_SECS = 180
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4096

# Custom headers
HEADER_REQUEST_ID = "X-Request-ID"
HEADER_PROCESS_TIME = "X-Process-Time"
