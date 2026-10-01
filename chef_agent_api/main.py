# main.py
# FastAPI application routing, cache integration, and background tasks
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import json

from config import settings
from db import init_db, get_recipe_from_db, save_recipe_to_db
from schemas import RecipeGenerateRequest, RecipeRestructureRequest, RecipeResponse
from llm_client import LLMClient
import prompts

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the SQLite cache on application startup
    await init_db()
    yield

app = FastAPI(
    title="Chef Agent API",
    description="Asynchronous Recipe Generation, Caching and Restructuring API using Gemma & Ollama",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize the LLM client
llm = LLMClient()

async def background_cache_standard_recipe(dish_name: str):
    """
    Asynchronously generates a standard version of a recipe and caches it in SQLite in the background.
    """
    try:
        # Re-check database cache to avoid redundant generations
        exists = await get_recipe_from_db(dish_name)
        if exists:
            return
        
        prompt = prompts.USER_PROMPT_GENERATION.format(dish_name=dish_name)
        data = await llm.generate_json(prompt, prompts.SYSTEM_PROMPT_GENERATION)
        
        if "ingredients" in data and "instructions" in data:
            await save_recipe_to_db(dish_name, data["ingredients"], data["instructions"])
            print(f"[CACHE] Successfully cached standard recipe for: '{dish_name}' in background.")
    except Exception as e:
        print(f"[CACHE ERROR] Failed to cache standard recipe for '{dish_name}': {e}")

@app.post("/api/v1/recipes/generate", response_model=RecipeResponse)
async def generate_recipe(request: RecipeGenerateRequest):
    dish_cleaned = request.dish_name.lower().strip()
    
    # 1. Query the cache asynchronously
    cached_recipe = await get_recipe_from_db(dish_cleaned)
    
    # Determine if the user has requested any customizations
    has_customizations = len(request.exclusions) > 0 or len(request.substitutions) > 0 or request.feedback
    
    if cached_recipe:
        # --- Database Cache Hit ---
        if not has_customizations:
            # Case A: Return the cached standard recipe instantly (0 LLM overhead)
            return RecipeResponse(
                dish_name=cached_recipe["dish_name"],
                ingredients=cached_recipe["ingredients"],
                instructions=cached_recipe["instructions"],
                source="database",
                is_customized=False,
                message="Retrieved directly from database cache."
            )
        else:
            # Case B: Standard recipe found, but user wants it customized.
            # Use Gemma to adapt the standard recipe to the user's constraints.
            try:
                base_recipe_json = json.dumps(cached_recipe)
                prompt = prompts.USER_PROMPT_ADAPTATION.format(
                    base_recipe_json=base_recipe_json,
                    exclusions=", ".join(request.exclusions),
                    substitutions=json.dumps(request.substitutions),
                    feedback=request.feedback or "None"
                )
                adapted_data = await llm.generate_json(prompt, prompts.SYSTEM_PROMPT_ADAPTATION)
                return RecipeResponse(
                    dish_name=adapted_data.get("dish_name", cached_recipe["dish_name"]),
                    ingredients=adapted_data.get("ingredients", []),
                    instructions=adapted_data.get("instructions", []),
                    source="gemma_adapted",
                    is_customized=True,
                    message="Standard recipe retrieved from database and customized via Gemma."
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to adapt recipe: {str(e)}")
    else:
        # --- Database Cache Miss ---
        # Process the current user request
        try:
            if not has_customizations:
                # User wants a standard recipe. Fetch it now and save it synchronously to cache.
                prompt = prompts.USER_PROMPT_GENERATION.format(dish_name=request.dish_name)
                data = await llm.generate_json(prompt, prompts.SYSTEM_PROMPT_GENERATION)
                
                await save_recipe_to_db(request.dish_name, data.get("ingredients", []), data.get("instructions", []))
                
                return RecipeResponse(
                    dish_name=data.get("dish_name", request.dish_name),
                    ingredients=data.get("ingredients", []),
                    instructions=data.get("instructions", []),
                    source="gemma_generated",
                    is_customized=False,
                    message="Recipe generated by Gemma and cached in database."
                )
            else:
                # User wants a customized recipe. Generate it from scratch directly.
                prompt = prompts.USER_PROMPT_CUSTOM_GENERATION.format(
                    dish_name=request.dish_name,
                    exclusions=", ".join(request.exclusions),
                    substitutions=json.dumps(request.substitutions),
                    feedback=request.feedback or "None"
                )
                data = await llm.generate_json(prompt, prompts.SYSTEM_PROMPT_CUSTOM_GENERATION)
                
                return RecipeResponse(
                    dish_name=data.get("dish_name", request.dish_name),
                    ingredients=data.get("ingredients", []),
                    instructions=data.get("instructions", []),
                    source="gemma_adapted",
                    is_customized=True,
                    message="Customized recipe generated from scratch via Gemma."
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM Generation failed: {str(e)}")

@app.post("/api/v1/recipes/restructure", response_model=RecipeResponse)
async def restructure_recipe(request: RecipeRestructureRequest):
    """
    Accepts an existing recipe and user feedback, and restructures it asynchronously.
    """
    try:
        base_recipe_json = json.dumps(request.recipe.model_dump())
        prompt = prompts.USER_PROMPT_ADAPTATION.format(
            base_recipe_json=base_recipe_json,
            exclusions=", ".join(request.exclusions),
            substitutions=json.dumps(request.substitutions),
            feedback=request.feedback
        )
        restructured_data = await llm.generate_json(prompt, prompts.SYSTEM_PROMPT_ADAPTATION)
        return RecipeResponse(
            dish_name=restructured_data.get("dish_name", request.recipe.dish_name),
            ingredients=restructured_data.get("ingredients", []),
            instructions=restructured_data.get("instructions", []),
            source="gemma_restructured",
            is_customized=True,
            message="Recipe restructured based on user feedback."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Restructuring failed: {str(e)}")

# Serve Static files. We will build the frontend assets in static/
import os
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
