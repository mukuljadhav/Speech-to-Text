# test_restructure_endpoint.py
import json
import sys
from config import settings
settings.DB_PATH = "test_chef_agent.db"

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Standard Pasta Carbonara recipe to be restructured
base_recipe = {
    "dish_name": "Pasta Carbonara",
    "ingredients": [
        {"name": "Spaghetti", "quantity": "200", "unit": "grams"},
        {"name": "Bacon", "quantity": "100", "unit": "grams"},
        {"name": "Egg Yolks", "quantity": "3", "unit": "pieces"},
        {"name": "Heavy Cream", "quantity": "50", "unit": "ml"},
        {"name": "Garlic Clove", "quantity": "1", "unit": "piece"},
        {"name": "Parmesan Cheese", "quantity": "50", "unit": "grams"}
    ],
    "instructions": [
        "Boil spaghetti in salted water until al dente.",
        "Sauté chopped bacon and minced garlic in a pan until crispy.",
        "Whisk egg yolks, heavy cream, and grated parmesan in a bowl.",
        "Drain pasta and toss it in the pan with bacon.",
        "Remove from heat and pour in the egg-cream mixture quickly to create a creamy sauce."
    ]
}

# Payload with exclusions, substitutions, and custom feedback
payload = {
    "recipe": base_recipe,
    "exclusions": ["garlic clove", "egg yolks"],
    "substitutions": {
        "bacon": "smoked tofu",
        "heavy cream": "cashew cream",
        "parmesan cheese": "nutritional yeast",
        "spaghetti": "gluten-free spaghetti"
    },
    "feedback": "Make it fully vegan, gluten-free, and simplify the instructions into fewer steps."
}

print("Sending restructure request to local Gemma server (Ollama)...")
try:
    # Use TestClient to invoke the route in-memory
    res = client.post("/api/v1/recipes/restructure", json=payload)
    print(f"Status Code: {res.status_code}")
    if res.status_code == 200:
        print("\n=== SUCCESS: Full JSON Response ===")
        print(json.dumps(res.json(), indent=2))
    else:
        print(f"❌ FAIL: API returned error {res.status_code}: {res.text}")
except Exception as e:
    print(f"❌ FAIL: Connection or execution error: {e}")
