# llm/exceptions.py
# Custom exception hierarchy for LLM operations

class LLMBaseError(Exception):
    """Base exception class for all LLM service failures."""
    pass

class LLMConnectionError(LLMBaseError):
    """Raised when the local LLM daemon is unreachable."""
    pass

class LLMTimeoutError(LLMBaseError):
    """Raised when the local LLM inference exceeds timeout boundaries."""
    pass

class LLMModelNotFoundError(LLMBaseError):
    """Raised when the configured model name is missing/not pulled."""
    pass

class LLMServerError(LLMBaseError):
    """Raised when the local LLM server encounters a critical internal error."""
    pass
