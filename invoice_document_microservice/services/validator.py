# services/validator.py
# Cleans and validates raw LLM output strings into Python dictionaries
import json
from typing import Dict, Any

class JSONValidationError(Exception):
    """Raised when the LLM output cannot be parsed as valid JSON."""
    pass

class SchemaValidator:
    def __init__(self):
        pass

    def parse_and_clean_json(self, raw_text: str) -> Dict[str, Any]:
        """
        Cleans Markdown fences, extracts JSON substring, and validates JSON structure.
        
        Args:
            raw_text (str): Raw string output from the LLM.
            
        Returns:
            Dict[str, Any]: Clean parsed dictionary.
            
        Raises:
            JSONValidationError: If parsing fails.
        """
        if not raw_text:
            raise JSONValidationError("Empty response received from LLM.")

        cleaned = raw_text.strip()

        # 1. Strip Markdown Code Block Fences if present (e.g. ```json ... ```)
        if cleaned.startswith("```"):
            # Remove opening fence
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove closing fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        # 2. Heuristic: Locate actual JSON block if there's conversational prefix/suffix text
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        
        if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
            raise JSONValidationError(
                f"Could not locate JSON block brackets in LLM response: {raw_text[:100]}..."
            )
            
        json_str = cleaned[start_idx:end_idx + 1]

        # 3. Parse JSON
        try:
            parsed_data = json.loads(json_str)
            if not isinstance(parsed_data, dict):
                raise JSONValidationError("JSON output is not a dictionary object.")
            return parsed_data
        except json.JSONDecodeError as exc:
            raise JSONValidationError(f"Invalid JSON format: {str(exc)}. Raw snippet: {json_str[:150]}") from exc
