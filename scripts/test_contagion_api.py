import requests
import json

API_URL = "http://localhost:8000/api/v1/analyze/contagion"

REQUEST_PAYLOAD = {
    "institutions": [
        {
            "institution_id": "bank1",
            "name": "Global Bank",
            "type": "bank",
            "country": "US",
            "assets": 1000000000,
            "capital": 100000000,
            "risk_weight": 1.0
        },
        {
            "institution_id": "bank2",
            "name": "International Bank",
            "type": "bank",
            "country": "UK",
            "assets": 800000000,
            "capital": 80000000,
            "risk_weight": 1.2
        }
    ],
    "exposures": {
        "matrix": [
            [0, 5000000],
            [3000000, 0]
        ],
        "institution_ids": ["bank1", "bank2"]
    },
    "shock_scenarios": [
        {
            "name": "liquidity_shock",
            "description": "10% liquidity shock to bank1",
            "node_attribute": "capital",
            "node_indices": [0],
            "value": -0.1
        }
    ]
}

def test_contagion_endpoint():
    """Sends a POST request to the contagion analysis endpoint and prints the response."""
    try:
        response = requests.post(API_URL, json=REQUEST_PAYLOAD, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        print(f"Status Code: {response.status_code}")
        try:
            response_data = response.json()
            print("Response JSON:")
            print(json.dumps(response_data, indent=2))
        except json.JSONDecodeError:
            print("Response content is not valid JSON:")
            print(response.text)
            
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response content: {response.text if 'response' in locals() else 'No response content'}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
        print("Please ensure the API server is running at http://localhost:8000")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An unexpected error occurred: {req_err}")

if __name__ == "__main__":
    print(f"Sending request to {API_URL}...")
    test_contagion_endpoint()
