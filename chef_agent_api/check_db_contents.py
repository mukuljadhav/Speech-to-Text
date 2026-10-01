# check_db_contents.py
import sqlite3
import json

conn = sqlite3.connect("chef_agent.db")
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

try:
    cursor.execute("SELECT id, dish_name, created_at FROM recipes")
    rows = cursor.fetchall()
    
    print("=== Saved Cache Recipes in Database ===")
    if not rows:
        print("Database is currently empty. Run a standard recipe search to populate the cache!")
    else:
        print(f"Total recipes cached: {len(rows)}\n")
        print(f"{'ID':<5} | {'Dish Name':<25} | {'Created At':<20}")
        print("-" * 56)
        for row in rows:
            print(f"{row['id']:<5} | {row['dish_name']:<25} | {row['created_at']:<20}")
except Exception as e:
    print(f"Error querying database: {e}")
finally:
    conn.close()
