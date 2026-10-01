"""
Simple manual test for Phase 1.
Make sure the API is running first: uvicorn app.main:app --reload
Then run:  python test_phase1.py
"""

import json

import requests

BASE_URL = "http://localhost:8000"


def test_health():
    resp = requests.get(f"{BASE_URL}/health")
    print("Health check:", resp.status_code, resp.json())
    assert resp.status_code == 200


def test_generate_from_text():
    payload = {
        "text": "Summer sale on running shoes, 40% off, energetic and bold vibe",
        "brand_name": "SprintCo",
    }
    resp = requests.post(f"{BASE_URL}/api/v1/campaign/generate-from-text", json=payload)
    print("Status code:", resp.status_code)
    print(json.dumps(resp.json(), indent=2))
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["banner_path"]
    print(f"\nBanner saved at: {data['banner_path']}")


if __name__ == "__main__":
    test_health()
    test_generate_from_text()
    print("\nAll Phase 1 tests passed.")
