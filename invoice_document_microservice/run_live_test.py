# run_live_test.py
# End-to-end integration script to verify live document extraction using local Ollama (Gemma 2 2B)

import httpx
import json
import time
import subprocess
import os
import sys

base_url = "http://127.0.0.1:8000"
ollama_url = "http://localhost:11434"

def verify_ollama():
    print("Step 1: Probing local Ollama server status...")
    try:
        response = httpx.get(ollama_url, timeout=2.0)
        if response.status_code == 200:
            print("  [SUCCESS] Ollama is running locally.")
            return True
    except Exception as e:
        print(f"  [ERROR] Cannot connect to Ollama. Make sure the Ollama application is running. Details: {e}")
        return False

def verify_gemma_pulled():
    print("Step 2: Checking if 'gemma2:2b' model is available in Ollama...")
    try:
        # Query Ollama's tags endpoint
        response = httpx.get(f"{ollama_url}/api/tags", timeout=3.0)
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            print(f"  Available models: {models}")
            if any("gemma2:2b" in m for m in models):
                print("  [SUCCESS] 'gemma2:2b' is pulled and ready.")
                return True
            else:
                print("  [WARNING] 'gemma2:2b' was not found in your pulled models list.")
                print("  Attempting to auto-pull 'gemma2:2b'. This might take a couple of minutes...")
                # Try pulling model
                pull_payload = {"name": "gemma2:2b", "stream": False}
                pull_resp = httpx.post(f"{ollama_url}/api/pull", json=pull_payload, timeout=600.0)
                if pull_resp.status_code == 200:
                    print("  [SUCCESS] 'gemma2:2b' successfully pulled.")
                    return True
        return False
    except Exception as e:
        print(f"  [ERROR] Failed to query model tags: {e}")
        return False

def create_sample_document():
    print("Step 3: Creating a sample text document...")
    sample_text = """
    MEDICAL HEALTH REPORT
    Date: June 30, 2026
    Patient Name: Johnathan Doe
    Age: 34
    Symptoms: Mild headache, slight fatigue, and sore throat.
    Diagnosis: Seasonal viral rhinopharyngitis (common cold).
    Prescribed Medicine:
    1. Paracetamol 500mg (1 tablet every 6 hours for fever/pain)
    2. Vitamin C 1000mg (1 tablet daily for immunity)
    Recommendation: Drink plenty of warm fluids and take 3 days of complete bed rest.
    """
    sample_path = "sample_medical_report.txt"
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(sample_text.strip())
    print(f"  [SUCCESS] Created sample file: '{sample_path}'")
    return sample_path

def run_extraction_test(file_path):
    print("Step 4: Sending live extraction request to FastAPI microservice...")
    url = f"{base_url}/api/v1/extract"
    
    # We must measure processing latency
    start_time = time.time()
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "text/plain")}
            print("  Processing document... (Running on CPU, please wait 10-30 seconds)")
            response = httpx.post(url, files=files, timeout=180.0)
            
        duration = time.time() - start_time
        print(f"  Completed in: {duration:.2f} seconds")
        print(f"  API Response Status: {response.status_code}")
        
        # Output headers
        print("  Headers:")
        for h in ["X-Request-ID", "X-Process-Time"]:
            if h in response.headers:
                print(f"    {h}: {response.headers[h]}")
                
        print("\n=== Live Extracted JSON Output ===")
        print(json.dumps(response.json(), indent=2))
        
    except Exception as e:
        print(f"  [ERROR] HTTP post request failed: {e}")

def main():
    print("====================================================")
    print("   Starting Live Document Extraction Integration Test")
    print("====================================================\n")
    
    if not verify_ollama():
        sys.exit(1)
        
    if not verify_gemma_pulled():
        sys.exit(1)
        
    sample_file = create_sample_document()
    
    # Start the server in the background
    print("\nStep 4: Booting up the FastAPI microservice server...")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--port", "8000", "--host", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Poll health endpoint to check when the server is fully ready (max 15 seconds)
    print("  Waiting for API server to initialize...")
    for _ in range(15):
        try:
            with httpx.Client() as client:
                resp = client.get(f"{base_url}/api/v1/health", timeout=1.0)
                if resp.status_code == 200:
                    print("  [SUCCESS] API server is online and responding.")
                    break
        except Exception:
            time.sleep(1.0)
    else:
        print("  [ERROR] API server failed to start within 15 seconds.")
        server_process.terminate()
        sys.exit(1)
        
    try:
        run_extraction_test(sample_file)
    finally:
        # Shutdown server
        print("\nStep 5: Cleaning up resources and shutting down API server...")
        server_process.terminate()
        server_process.wait()
        # Delete temporary sample file
        if os.path.exists(sample_file):
            os.remove(sample_file)
        print("  [SUCCESS] API server stopped. Diagnostic complete.")

if __name__ == "__main__":
    main()
