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
    response = requests.post(api_endpoint, headers=headers, data=json.dumps(case_data))
    if response.status_code == 200:
        print(f"{case_name}:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"{case_name} failed with status code {response.status_code}:")
        print(response.text)


case_name = "Simple fuzzy filtering"

data = {
    "fuzzy_filters": {"label": "date"},
    "fuzzy_filters_config": {"model_name": "LaBSE"},
}
perform_search_case(data, case_name)

case_name = "Simple exact filtering"
data = {

    "exact_filters": {"termtype": "ObjectProperty"}
}
perform_search_case(data, case_name)

case_name = "Complex exact filtering"
data = {
    "exact_filters": {"termtype": "ObjectProperty", "domain": 'http://www.demcare.eu/ontologies/demlab.owl#Protocol'}
}
perform_search_case(data, case_name)


case_name = "Complex fuzzy filtering"

data = {
    "fuzzy_filters": {"label": "date", "domain": "creative work"},
    "fuzzy_filters_config": {"model_name": "LaBSE"},
    "exact_filters": {"termtype": "ObjectProperty"}
}
perform_search_case(data, case_name)


