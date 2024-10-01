import requests
import json

# Define the API endpoint
api_endpoint = "http://127.0.0.1:9090/search"

# Define the model name
model_name = "LaBSE"

# Function to perform API requests and print results
def perform_search_case(case_data, case_name):
    headers = {
        "Content-Type": "application/json"
    }
    # Include the model_name in the case data
    print("Querying", case_data)
    response = requests.post(api_endpoint, headers=headers, data=json.dumps(case_data))
    if response.status_code == 200:
        print(f"{case_name}:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"{case_name} failed with status code {response.status_code}:")
        print(response.text)

case_4 = {
    "model_name": "LaBSE",
    "term": "Date",
    "context": {
        "domain": "person",
    },
    "filters": {
        "datatype":  "something"
    }
}
perform_search_case(case_4, "CASE 4")

case_5 = {
    "model_name": "LaBSE",
    "term": "Date",
    "context": {
        "domain": "creative work",
    },
    "filters": {
        "datatype": "object_property"
    }
}
perform_search_case(case_5, "CASE 5")

case_6 = {
    "model_name": "LaBSE",
    "term": "Date",
}
perform_search_case(case_6, "CASE 6")
