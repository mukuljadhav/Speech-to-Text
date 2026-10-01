# Chef Agent API - Backend Developer Guide

A production-ready asynchronous FastAPI REST microservice that generates, caches, and restructures recipes using a local Ollama instance (running Gemma) and a non-blocking SQLite database cache.

## 🚀 Setup & Execution

### 1. Prerequisites
* **Python**: Python 3.9 or higher.
* **Ollama**: Ensure Ollama is running locally and has the gemma model pulled:
  ```bash
  ollama serve
  ollama pull gemma2:2b
  ```

### 2. Install Dependencies
Install the required packages using pip:
```bash
pip install fastapi uvicorn pydantic pydantic-settings httpx
```

### 3. Running the Server Locally
Start the Uvicorn dev server from the `chef_agent_api` directory:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Running with Docker Compose
If you prefer running the API inside a Docker container:
```bash
# Build and run the container in the background
docker compose up --build -d

# View real-time logs
docker compose logs -f
```
Once started, the API is accessible at `http://localhost:8000`. The SQLite cache database file will be persistently stored on your host machine in the `chef_agent_api/data/` folder.

Once running:
* **Interactive API Documentation (Swagger)**: `http://localhost:8000/docs`
* **Alternative Documentation (Redoc)**: `http://localhost:8000/redoc`

---

## 🔌 API Endpoints for Frontend Integration

Our API consists of two core endpoints designed for stateless communication.

### 1. Generate / Adapt Recipe
* **Endpoint**: `POST /api/v1/recipes/generate`
* **Description**: Queries the SQLite database cache. If found and no customizations are requested, it returns the recipe instantly. If customizations (exclusions, substitutions, feedback) are requested, it uses Gemma to modify the recipe. If the dish is not in the cache, it triggers background caching of the standard recipe and returns the generated customized recipe.

#### Request Body Schema
```json
{
  "dish_name": "string (Required)",
  "exclusions": ["string (Optional) - list of ingredients to omit entirely"],
  "substitutions": {
    "ingredient_to_replace": "replacement_ingredient (Optional)"
  },
  "feedback": "string (Optional) - e.g., 'Make it extra spicy', 'Cook under 20 mins'"
}
```

#### Response Body Schema
```json
{
  "dish_name": "string",
  "ingredients": [
    {
      "name": "string",
      "quantity": "string",
      "unit": "string"
    }
  ],
  "instructions": [
    "string"
  ],
  "source": "string ('database', 'gemma_generated', or 'gemma_adapted')",
  "is_customized": "boolean",
  "message": "string"
}
```

#### Example cURL Request (Jain Diet Customization)
```bash
curl -X POST http://localhost:8000/api/v1/recipes/generate \
  -H "Content-Type: application/json" \
  -d '{
    "dish_name": "Paneer Butter Masala",
    "exclusions": ["onion", "garlic"],
    "substitutions": {
      "paneer": "tofu"
    },
    "feedback": "Make it extra creamy"
  }'
```

#### Example JS Integration
```javascript
async function getRecipe(dishName, exclusions = [], substitutions = {}, feedback = null) {
  try {
    const response = await fetch("http://localhost:8000/api/v1/recipes/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dish_name: dishName,
        exclusions: exclusions,
        substitutions: substitutions,
        feedback: feedback
      })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to generate recipe");
    }
    
    const recipeData = await response.json();
    console.log("Source of Recipe:", recipeData.source); // 'database' (cache) vs 'gemma_adapted'
    console.log("Ingredients:", recipeData.ingredients);
    console.log("Instructions:", recipeData.instructions);
    return recipeData;
  } catch (error) {
    console.error("Error fetching recipe:", error);
  }
}
```

---

### 2. Restructure Recipe
* **Endpoint**: `POST /api/v1/recipes/restructure`
* **Description**: Takes the *existing* recipe and applies natural-language feedback (e.g., *"scale it for 8 people"*, *"make it dairy-free"*) to return a modified recipe. This is fully stateless and requires no database session tracking.

#### Request Body Schema
```json
{
  "recipe": {
    "dish_name": "string (Required)",
    "ingredients": [
      {
        "name": "string",
        "quantity": "string",
        "unit": "string"
      }
    ],
    "instructions": ["string"]
  },
  "exclusions": ["string (Optional)"],
  "substitutions": {
    "ingredient": "substitute"
  },
  "feedback": "string (Required) - e.g., 'Make it dairy free'"
}
```

#### Example cURL Request (Tweak / Restructure)
```bash
curl -X POST http://localhost:8000/api/v1/recipes/restructure \
  -H "Content-Type: application/json" \
  -d '{
    "recipe": {
      "dish_name": "Paneer Butter Masala",
      "ingredients": [
        {"name": "Paneer", "quantity": "200", "unit": "grams"},
        {"name": "Tomato Puree", "quantity": "1", "unit": "cup"},
        {"name": "Heavy Cream", "quantity": "2", "unit": "tablespoons"}
      ],
      "instructions": [
        "Cut paneer into cubes.",
        "Sauté tomatoes and stir in cream and paneer."
      ]
    },
    "exclusions": [],
    "substitutions": {},
    "feedback": "Make it dairy free by swapping butter/cream/paneer with vegan options"
  }'
```

#### Example JS Integration
```javascript
async function restructureRecipe(currentRecipe, feedback, exclusions = [], substitutions = {}) {
  try {
    const response = await fetch("http://localhost:8000/api/v1/recipes/restructure", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        recipe: currentRecipe, // Send the full recipe object you received earlier
        exclusions: exclusions,
        substitutions: substitutions,
        feedback: feedback
      })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to restructure recipe");
    }
    
    const restructuredRecipe = await response.json();
    return restructuredRecipe;
  } catch (error) {
    console.error("Error restructuring recipe:", error);
  }
}
```

---

## 🛠️ Configuration Details

You can customize setting values using environment variables. FastAPI reads these automatically on startup:

* `MODEL_NAME`: Set custom Ollama model name (default: `gemma2:2b`).
* `OLLAMA_URL`: Local or remote Ollama server URL (default: `http://localhost:11434`).
* `DB_PATH`: SQLite database file location (default: `chef_agent.db`).
