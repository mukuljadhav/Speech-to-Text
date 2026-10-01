# llm/base_llm.py
# Abstract LLM Interface base class
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        Query the local LLM and return the raw output string.
        
        Args:
            prompt (str): User instruction and raw document text.
            system_prompt (str, optional): Custom system behavior constraints.
            
        Returns:
            str: Raw JSON string response from the model.
        """
        pass
