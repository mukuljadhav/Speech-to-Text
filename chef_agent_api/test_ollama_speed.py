# test_ollama_speed.py
import httpx
import time
import asyncio

async def test():
    start = time.time()
    print("Sending standard request to local Ollama...")
    async with httpx.AsyncClient(timeout=180) as client:
        try:
            res = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "gemma2:2b",
                    "prompt": "Generate a short recipe for aloo Samosa",
                    "stream": False
                }
            )
            print(f"Status Code: {res.status_code}")
            print(f"Time Taken: {time.time() - start:.2f} seconds")
            if res.status_code == 200:
                print(f"Response preview: {res.json().get('response', '')[:300]}...")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test())
