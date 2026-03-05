# test_api_direct.py
import requests
import json

print("Testing UrRight API...")
url = "http://localhost:8516/chat"

payload = {
    "message": "What are my rights if arrested?",
    "session_id": "test123",
    "language": "en"
}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")