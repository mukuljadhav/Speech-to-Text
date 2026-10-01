# services/prompt_manager.py
# Loading, caching, and formatting prompt template text files
import os

class PromptManager:
    def __init__(self, prompts_dir: str):
        self.prompts_dir = prompts_dir
        self.system_prompt = ""
        self.extraction_template = ""
        self.load_prompts()

    def load_prompts(self) -> None:
        """
        Loads the prompt text templates from file.
        Uses fallback defaults if the target files do not exist.
        """
        system_prompt_path = os.path.join(self.prompts_dir, "system_prompt.txt")
        extraction_prompt_path = os.path.join(self.prompts_dir, "extraction_v1.txt")

        # 1. Read System Prompt Template
        if os.path.exists(system_prompt_path):
            with open(system_prompt_path, "r", encoding="utf-8") as f:
                self.system_prompt = f.read().strip()
        else:
            # Fallback guideline
            self.system_prompt = (
                "You are an expert document data extraction system. "
                "Analyze the document text and return all extracted information structured logically as JSON."
            )

        # 2. Read User Instruction/Extraction Template
        if os.path.exists(extraction_prompt_path):
            with open(extraction_prompt_path, "r", encoding="utf-8") as f:
                self.extraction_template = f.read().strip()
        else:
            # Fallback template with a placeholder
            self.extraction_template = (
                "Perform data extraction on the following document text.\n\n"
                "Document Text:\n"
                "===\n"
                "{document_text}\n"
                "===\n"
                "Return the dynamic extracted fields as a flat or nested JSON structure."
            )

    def build_user_prompt(self, document_text: str) -> str:
        """
        Formats the extraction template with the provided document text.
        """
        # Ensure template formatting doesn't break if document text has raw curly braces
        # We can handle formatting by replacing '{document_text}' explicitly
        # to prevent KeyError in documents containing arbitrary text braces.
        if "{document_text}" in self.extraction_template:
            return self.extraction_template.replace("{document_text}", document_text)
        return f"{self.extraction_template}\n\nDocument Text:\n{document_text}"

    def get_system_prompt(self) -> str:
        """
        Returns the loaded system prompt instructions.
        """
        return self.system_prompt
