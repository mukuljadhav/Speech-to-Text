# db.py
# Asynchronous Database Operations using SQLite and Thread Pools
import sqlite3
import json
import asyncio
from typing import Optional, Dict, Any, List
from config import settings

async def init_db():
    """
    Initializes the SQLite database and creates the recipes table if it doesn't exist.
    Runs synchronously in a separate thread to prevent blocking the event loop.
    """
    def _init():
        conn = sqlite3.connect(settings.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recipes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dish_name TEXT UNIQUE,
                ingredients TEXT,
                instructions TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    await asyncio.to_thread(_init)

async def get_recipe_from_db(dish_name: str) -> Optional[Dict[str, Any]]:
    """
    Asynchronously queries the database for a recipe by dish name (case-insensitive).
    Returns the parsed recipe structure or None if not found.
    """
    def _get():
        conn = sqlite3.connect(settings.DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # lowercase search for uniform matching
        cleaned_name = dish_name.lower().strip()
        cursor.execute("SELECT * FROM recipes WHERE LOWER(dish_name) = ?", (cleaned_name,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "dish_name": row["dish_name"],
                "ingredients": json.loads(row["ingredients"]),
                "instructions": json.loads(row["instructions"])
            }
        return None
    return await asyncio.to_thread(_get)

async def save_recipe_to_db(dish_name: str, ingredients: List[Dict[str, Any]], instructions: List[str]):
    """
    Asynchronously saves or replaces a recipe in the database.
    Serializes ingredients and instructions lists to JSON strings.
    """
    def _save():
        conn = sqlite3.connect(settings.DB_PATH)
        cursor = conn.cursor()
        # lowercase storage for consistent lookups
        cleaned_name = dish_name.lower().strip()
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO recipes (dish_name, ingredients, instructions) VALUES (?, ?, ?)",
                (cleaned_name, json.dumps(ingredients), json.dumps(instructions))
            )
            conn.commit()
        except Exception as e:
            print(f"Database write error: {e}")
        finally:
            conn.close()
    await asyncio.to_thread(_save)
