# services/llm_service.py
# Layered service to build prompts and execute queries using BaseLLM interface
from llm.base_llm import BaseLLM
from services.prompt_manager import PromptManager

class LLMService:
    def __init__(self, llm_client: BaseLLM, prompt_manager: PromptManager):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager

    async def extract_json_from_text(self, text: str) -> str:
        """
        Formats text using prompt manager and queries the abstracted LLM.
        
        Args:
            text (str): Document text payload.
            
        Returns:
            str: Raw JSON string returned from local LLM.
        """
        # Compile templates using PromptManager
        user_prompt = self.prompt_manager.build_user_prompt(text)
        system_prompt = self.prompt_manager.get_system_prompt()
        
        # Execute generate call asynchronously
        raw_json_str = await self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt
        )
        return raw_json_str
