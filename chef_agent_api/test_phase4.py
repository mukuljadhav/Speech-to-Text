# test_phase4.py
# Verification script for Phase 4 (FastAPI Endpoints)
import os
import sys
import shutil
import asyncio

# Configure settings BEFORE importing app to override the database path
from config import settings
settings.DB_PATH = "test_chef_agent.db"

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def run_tests():
    print("--- Testing API Endpoints ---")
    
    # Pre-clean test database
    if os.path.exists("test_chef_agent.db"):
        os.remove("test_chef_agent.db")

    # 1. Trigger Startup (creates tables)
    with TestClient(app) as test_client:
        print("Database initialized via lifespan.")
        
        # Test Case A: First Generation (Cache Miss)
        print("Testing Case A: Standard Generation (Cache Miss)...")
        payload = {
            "dish_name": "Boiled Water",
            "exclusions": [],
            "substitutions": {},
            "feedback": None
        }
        res = test_client.post("/api/v1/recipes/generate", json=payload)
        
        if res.status_code != 200:
            print(f"❌ FAIL: Generate returned {res.status_code}: {res.text}")
            sys.exit(1)
            
        data = res.json()
        print(f"First Gen Response: Source={data.get('source')}, Customized={data.get('is_customized')}")
        
        if data.get("source") != "gemma_generated":
            print(f"❌ FAIL: Expected source 'gemma_generated', got '{data.get('source')}'")
            sys.exit(1)
            
        # Give background task a small fraction of a second if needed, though TestClient handles synchronous background tasks automatically
        print("[OK] Cache miss generation successful.")

        # Test Case B: Second Generation (Cache Hit)
        print("Testing Case B: Cache Hit...")
        res2 = test_client.post("/api/v1/recipes/generate", json=payload)
        
        if res2.status_code != 200:
            print(f"❌ FAIL: Cache hit check returned {res2.status_code}: {res2.text}")
            sys.exit(1)
            
        data2 = res2.json()
        print(f"Second Gen Response: Source={data2.get('source')}, Customized={data2.get('is_customized')}")
        
        if data2.get("source") != "database":
            print(f"❌ FAIL: Expected source 'database', got '{data2.get('source')}'")
            sys.exit(1)
            
        print("[OK] Cache hit retrieval successful.")

        # Test Case C: Adaptation of DB Cache
        print("Testing Case C: Adapting DB recipe (Exclusions)...")
        payload_custom = {
            "dish_name": "Boiled Water",
            "exclusions": ["water"],
            "substitutions": {"water": "rosewater"},
            "feedback": "make it sweet"
        }
        res3 = test_client.post("/api/v1/recipes/generate", json=payload_custom)
        
        if res3.status_code != 200:
            print(f"❌ FAIL: Custom adaptation returned {res3.status_code}: {res3.text}")
            sys.exit(1)
            
        data3 = res3.json()
        print(f"Custom Gen Response: Source={data3.get('source')}, Customized={data3.get('is_customized')}")
        
        if data3.get("source") != "gemma_adapted":
            print(f"❌ FAIL: Expected source 'gemma_adapted', got '{data3.get('source')}'")
            sys.exit(1)
            
        print("[OK] Cache adaptation successful.")

        # Test Case D: Restructuring
        print("Testing Case D: Restructuring recipe...")
        payload_restruct = {
            "recipe": {
                "dish_name": "Boiled Water",
                "ingredients": [{"name": "Water", "quantity": "2", "unit": "cups"}],
                "instructions": ["Boil water in a pot."]
            },
            "exclusions": [],
            "substitutions": {},
            "feedback": "Scale ingredients for 10 people"
        }
        res4 = test_client.post("/api/v1/recipes/restructure", json=payload_restruct)
        
        if res4.status_code != 200:
            print(f"❌ FAIL: Restructure returned {res4.status_code}: {res4.text}")
            sys.exit(1)
            
        data4 = res4.json()
        print(f"Restructure Response: Source={data4.get('source')}, Customized={data4.get('is_customized')}")
        
        if data4.get("source") != "gemma_restructured":
            print(f"❌ FAIL: Expected source 'gemma_restructured', got '{data4.get('source')}'")
            sys.exit(1)
            
        print("[OK] Recipe restructuring successful.")

    # Clean up test database file
    print("Cleaning up test database file...")
    try:
        if os.path.exists("test_chef_agent.db"):
            os.remove("test_chef_agent.db")
        print("[OK] Cleaned up test database.")
    except Exception as e:
        print(f"⚠️ Warning: Could not remove test database file: {e}")

    print("\n=== ALL API ENDPOINT TESTS PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_tests()
