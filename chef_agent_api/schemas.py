# schemas.py
# Pydantic Request and Response Schemas
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class Ingredient(BaseModel):
    name: str = Field(..., description="Name of the ingredient (e.g. paneer, heavy cream)")
    quantity: str = Field(..., description="Quantity needed (e.g. 200, 2, 1.5, 1/2)")
    unit: str = Field(..., description="Unit of measurement (e.g. grams, pieces, tsp, cup, or empty string)")

class RecipeBase(BaseModel):
    dish_name: str = Field(..., description="Capitalized name of the dish")
    ingredients: List[Ingredient] = Field(..., description="List of ingredients needed for the recipe")
    instructions: List[str] = Field(..., description="Step-by-step instructions to prepare the dish")

class RecipeResponse(RecipeBase):
    source: str = Field(..., description="The source of this recipe: 'database' (cached hit), 'gemma_generated', 'gemma_adapted', or 'gemma_restructured'")
    is_customized: bool = Field(default=False, description="True if the recipe has been modified from the standard base version")
    message: Optional[str] = Field(None, description="Informational message about the recipe origin or modifications")

class RecipeGenerateRequest(BaseModel):
    dish_name: str = Field(..., example="Paneer Butter Masala", description="The name of the dish the user wants to cook")
    exclusions: List[str] = Field(default_factory=list, example=["onion", "garlic"], description="Ingredients the user wants to omit")
    substitutions: Dict[str, str] = Field(default_factory=dict, example={"paneer": "tofu"}, description="Key-value pairs of ingredients and their desired substitutes")
    feedback: Optional[str] = Field(None, example="Make it extra spicy and creamy", description="Additional custom preference or natural language constraints")

class RecipeRestructureRequest(BaseModel):
    recipe: RecipeBase = Field(..., description="The current recipe that needs to be restructured")
    exclusions: List[str] = Field(default_factory=list, description="Updated list of excluded ingredients")
    substitutions: Dict[str, str] = Field(default_factory=dict, description="Updated dictionary of ingredient substitutions")
    feedback: str = Field(..., description="Feedback or adjustments to apply to the recipe (e.g. 'Make it less sweet', 'Scale it up for 6 people')")
