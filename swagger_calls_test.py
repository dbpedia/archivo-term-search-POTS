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
    case_data["model_name"] = model_name
    print("Querying", case_data)
    response = requests.post(api_endpoint, headers=headers, data=json.dumps(case_data))
    if response.status_code == 200:
        print(f"{case_name}:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"{case_name} failed with status code {response.status_code}:")
        print(response.text)

# Define the cases
# print("CASE 1: ONTOLOGY FILTER")
# case_1 = {
#     "term": "birth date",
#     "filters": {
        
#         "datatype": "object_property",
#         "ontology": "artist",
#         "domain": ["https://cenguix.github.io/Text2KGBench/ont_17_artist/concepts#Artist"]

#     },


# }
# perform_search_case(case_1, "CASE 1")


# print("\nCASE 2: DOMAIN + RANGE FILTERS")
# case_2 = {
#     "term": "date",
#     "filters": {
        
#         "datatype": "object_property"
#     },
#     "context": {
#         "domain": "creative work"
#     }

# }
# perform_search_case(case_2, "CASE 2")

# print("\nCASE 3: MULTI COLLECTION QUERY")
# case_3 = {
#     "term": "Company",
# }
# perform_search_case(case_3, "CASE 3")

print("\nCASE 4: SIMPLE INJECTION QUERY 1")
case_4 = {
    "term": "Date",
    "context": {
        "domain": "person",
    },
    "filters": {
        "datatype": "object_property"
    }
}
perform_search_case(case_4, "CASE 4")

print("\nCASE 5: SIMPLE INJECTION QUERY 2")
case_5 = {
    "term": "Date",
    "context": {
        "domain": "creative work",
    },
    "filters": {
        "datatype": "object_property"
    }
}
perform_search_case(case_5, "CASE 5")

print("\nCASE 6: SIMPLE INJECTION QUERY 2")
case_6 = {
    "term": "Date",
}
perform_search_case(case_6, "CASE 6")

# print("\nCASE 6: SIMPLE INJECTION QUERY 3")
# case_6 = {
#     "term": "area",

#     "filters": {
#         "datatype": "object_property"
#     }
# }
# perform_search_case(case_6, "CASE 6")

# print("\nCASE 7: SIMPLE INJECTION QUERY 3")
# case_7 = {
#     "term": "area",
#     "context": {
#         "label": "surface",
#     },
#     "filters": {
#         "datatype": "object_property"
#     }
# }
# perform_search_case(case_7, "CASE 7")

# print("\nCASE 7: SIMPLE INJECTION QUERY 3")
# case_7 = {
#     "term": "area",
#     "context": {
#         "label": "land",
#     },
#     "filters": {
#         "datatype": "object_property"
#     }
# }
# perform_search_case(case_7, "CASE 7")

# print("\nCASE 8")
# case_8 = {
#     "term": "holder",

#     "filters": {
#         "datatype": "object_property"
#     },
#     "language": "English"
# }
# perform_search_case(case_8, "CASE 8")

# case_9 = {
#     "term": "ISO",

#     "filters": {
#         "datatype": "RDF_type"
#     },
#     "context": {
#         "label": "land",
#     },
    
#     "language": "English"
# }
# perform_search_case(case_9, "CASE 9")
