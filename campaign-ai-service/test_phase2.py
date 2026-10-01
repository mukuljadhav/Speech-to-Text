"""
Manual test for Phase 2 (unified endpoint: text + voice).

Make sure the API is running first: uvicorn app.main:app --reload
Then run: python test_phase2.py

Before running the voice test, record a short voice memo (a few seconds)
saying something like "Diwali sale on electronics, 30% off, festive vibe"
and save it as voice_sample.wav / .m4a / .mp3 in this same folder. Update
VOICE_FILE_PATH below to match.
"""

import json
import os

import requests

BASE_URL = "http://localhost:8000"
VOICE_FILE_PATH = "voice_sample.wav"  # <-- put your own recording here


def test_health():
    resp = requests.get(f"{BASE_URL}/health")
    print("Health check:", resp.status_code, resp.json())
    assert resp.status_code == 200


def test_text_only():
    print("\n--- Test: text only ---")
    resp = requests.post(
        f"{BASE_URL}/api/v1/campaign/generate",
        data={"text": "Summer sale on running shoes, 40% off, energetic vibe", "brand_name": "SprintCo"},
    )
    print("Status code:", resp.status_code)
    print(json.dumps(resp.json(), indent=2))
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_voice_only():
    print("\n--- Test: voice only ---")
    if not os.path.exists(VOICE_FILE_PATH):
        print(f"SKIPPED: {VOICE_FILE_PATH} not found. Record a short voice memo to test this.")
        return

    with open(VOICE_FILE_PATH, "rb") as f:
        files = {"voice": (VOICE_FILE_PATH, f, "audio/wav")}
        resp = requests.post(f"{BASE_URL}/api/v1/campaign/generate", files=files)

    print("Status code:", resp.status_code)
    print(json.dumps(resp.json(), indent=2))
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_text_and_voice_combined():
    print("\n--- Test: text + voice combined ---")
    if not os.path.exists(VOICE_FILE_PATH):
        print(f"SKIPPED: {VOICE_FILE_PATH} not found.")
        return

    with open(VOICE_FILE_PATH, "rb") as f:
        files = {"voice": (VOICE_FILE_PATH, f, "audio/wav")}
        data = {"text": "Make it look premium and minimal", "brand_name": "TechBazaar"}
        resp = requests.post(f"{BASE_URL}/api/v1/campaign/generate", data=data, files=files)

    print("Status code:", resp.status_code)
    print(json.dumps(resp.json(), indent=2))
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_no_input_returns_400():
    print("\n--- Test: no input (should fail with 400) ---")
    resp = requests.post(f"{BASE_URL}/api/v1/campaign/generate", data={})
    print("Status code:", resp.status_code, resp.json())
    assert resp.status_code == 400


if __name__ == "__main__":
    test_health()
    test_text_only()
    test_voice_only()
    test_text_and_voice_combined()
    test_no_input_returns_400()
    print("\nAll Phase 2 tests completed.")