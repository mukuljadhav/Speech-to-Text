# test_phase1.py
# Verification script for Phase 1 (Config & Schemas)
import sys
from config import settings
from schemas import Ingredient, RecipeBase, RecipeGenerateRequest, RecipeResponse
from pydantic import ValidationError

print("--- Testing config.py ---")
print(f"Ollama URL: {settings.OLLAMA_URL}")
print(f"Model Name: {settings.MODEL_NAME}")
print(f"Database Path: {settings.DB_PATH}")
print("[OK] config.py loaded successfully!\n")

print("--- Testing schemas.py ---")
# 1. Test correct ingredient schema
try:
    ing = Ingredient(name="Paneer", quantity="200", unit="grams")
    print(f"[OK] Ingredient schema validated: {ing}")
except ValidationError as e:
    print(f"[FAIL] Ingredient schema validation failed: {e}")
    sys.exit(1)

# 2. Test correct recipe schema
try:
    recipe = RecipeBase(
        dish_name="Paneer Butter Masala",
        ingredients=[
            Ingredient(name="Paneer", quantity="200", unit="grams"),
            Ingredient(name="Butter", quantity="2", unit="tbsp")
        ],
        instructions=[
            "Melt butter in a pan.",
            "Add paneer cubes and sauté."
        ]
    )
    print(f"[OK] RecipeBase schema validated: {recipe.dish_name} with {len(recipe.ingredients)} ingredients.")
except ValidationError as e:
    print(f"[FAIL] RecipeBase schema validation failed: {e}")
    sys.exit(1)

# 3. Test invalid schema inputs (to ensure validation is active and working)
try:
    # quantity is missing (required)
    Ingredient(name="Butter", unit="tbsp")
    print("[FAIL] Failed: Schema accepted missing required fields!")
    sys.exit(1)
except ValidationError as e:
    print("[OK] Schema successfully rejected missing required fields (expected behaviour).")

try:
    # dish_name is missing
    RecipeGenerateRequest(exclusions=["onion"])
    print("[FAIL] Failed: RecipeGenerateRequest accepted missing dish_name!")
    sys.exit(1)
except ValidationError as e:
    print("[OK] Schema successfully rejected missing dish_name in request (expected behaviour).")

print("\n=== ALL PHASE 1 TESTS PASSED SUCCESSFULLY ===")
