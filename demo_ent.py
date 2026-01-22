import requests

response = requests.post(
    "http://127.0.0.1:8000/predict", json={"city": "Los Angeles", "date": "2026-01-21"}
)

print("Status Code:", response.status_code)
print("Response:", response.json())
