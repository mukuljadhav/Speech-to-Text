# test_phase2.py
# Verification script for Phase 2 (Async Database Cache)
import asyncio
import os
import sys
from config import settings
from db import init_db, get_recipe_from_db, save_recipe_to_db

# Override the database file for testing to avoid polluting production cache
settings.DB_PATH = "test_chef_agent.db"

async def run_tests():
    print("--- Testing Database Layer ---")
    
    # 1. Initialize DB
    print("Initializing test database...")
    await init_db()
    if not os.path.exists("test_chef_agent.db"):
        print("❌ FAIL: Database file 'test_chef_agent.db' was not created!")
        sys.exit(1)
    print("[OK] Database file created successfully.")
    
    # 2. Test fetching non-existent recipe
    print("Querying non-existent recipe...")
    missing = await get_recipe_from_db("non existent dish")
    if missing is not None:
        print(f"❌ FAIL: Returned {missing} instead of None for missing recipe!")
        sys.exit(1)
    print("[OK] Querying non-existent recipe returned None.")
    
    # 3. Test saving a recipe
    print("Saving mock recipe...")
    mock_ingredients = [
        {"name": "test_paneer", "quantity": "100", "unit": "g"},
        {"name": "test_water", "quantity": "1", "unit": "cup"}
    ]
    mock_instructions = [
        "Step 1: Sauté test_paneer.",
        "Step 2: Add water."
    ]
    await save_recipe_to_db("Test Dish", mock_ingredients, mock_instructions)
    print("[OK] Recipe saved successfully.")
    
    # 4. Test fetching the saved recipe (case-insensitivity test)
    print("Querying saved recipe (testing case-insensitivity)...")
    fetched = await get_recipe_from_db("tEsT dIsH")
    if fetched is None:
        print("❌ FAIL: Could not retrieve saved recipe!")
        sys.exit(1)
    
    # Validate fields
    if fetched["dish_name"] != "test dish":
        print(f"❌ FAIL: Dish name mismatch. Expected 'test dish', got '{fetched['dish_name']}'")
        sys.exit(1)
        
    if fetched["ingredients"] != mock_ingredients:
        print(f"❌ FAIL: Ingredients mismatch. Got: {fetched['ingredients']}")
        sys.exit(1)
        
    if fetched["instructions"] != mock_instructions:
        print(f"❌ FAIL: Instructions mismatch. Got: {fetched['instructions']}")
        sys.exit(1)
        
    print("[OK] Successfully retrieved and verified mock recipe data.")
    
    # Clean up test database file
    print("Cleaning up test database file...")
    try:
        os.remove("test_chef_agent.db")
        print("[OK] Cleaned up test database.")
    except Exception as e:
        print(f"⚠️ Warning: Could not remove test database file: {e}")
        
    print("\n=== ALL DATABASE TESTS PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    asyncio.run(run_tests())
