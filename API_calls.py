import requests
import json

# Define the API endpoint
api_endpoint = "http://127.0.0.1:8014/search"
#"http://127.0.0.1:9090/search"

# Define the model name
model_name = "LaBSE"
import json
# Function to perform API requests and print results
import requests
import json
import csv
import requests
import json
import tabulate
def perform_search_case(case_data, case_name, output_filename="search_results.txt"):
    print()
    print("--------------------------- QUERY ---------------------------")
    
    # Print the query parameters as a small table
    query_data = [[k, v] for k, v in case_data.items()]  # Create a list of query parameters
    query_table = tabulate.tabulate(query_data, headers=["Query Parameter", "Value"], tablefmt="fancy_grid")
    
    # Print query table above results
    print(query_table)
    print()  # Extra newline for better formatting
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Include the model_name in the case data
    response = requests.post(api_endpoint, headers=headers, data=json.dumps(case_data))
    
    if response.status_code == 200:
        
        response_data = json.loads(response.text)
        print("Results retrieved successfully!")

        # Prepare results list for tabulate
        table_data = []
        result_count = 0
        for collection in response_data:
            for obj in response_data[collection]:
                result_count += 1
                obj_data = obj["object"]
                distance = obj["distance"]
                
                # Prepare the row for each result
                row = [
                    result_count,
                    obj_data.get("label", "N/A"),
                    obj_data.get("termIRI", "N/A"),
                    obj_data.get("description", "N/A"),
                    ', '.join(obj_data.get("domain", [])),
                    ', '.join(obj_data.get("range", [])),
                    distance
                ]
                table_data.append(row)

        # Define the headers for the results table
        result_headers = ["Result #", "Label", "Term IRI", "Description", "Domain", "Range", "Distance"]

        # Generate the result table using tabulate
        result_table_str = tabulate.tabulate(table_data, headers=result_headers, tablefmt="grid")
        
        # Print the result table
        print(result_table_str)

        # Save the query and result tables to a text file
        with open(output_filename, "a", encoding="utf-8") as f:
            f.write("\n\n")
            f.write(query_table)  # Save the query table
            f.write("\n\n")  # Add space between query and results
            f.write(result_table_str)  # Save the result table
            f.write("\n\n\n")
        print(f"\nResults have been saved to {output_filename}")

    else:
        print(f"{case_name} failed with status code {response.status_code}:")
        print(response.text)
case_name = "Simple fuzzy filtering"

data = {
    "fuzzy_filters": {"label": "parent", "range": "male"},
    "fuzzy_filters_config": {"model_name": "LaBSE", "lang": "en"},
    "exact_filters": {"termtype": "ObjectProperty"},
    "limit": 3
}
perform_search_case(data, case_name)

data = {
    "fuzzy_filters": {"label": "parent", "range": "female"},
    "fuzzy_filters_config": {"model_name": "LaBSE", "lang": "en"},
    "exact_filters": {"termtype": "ObjectProperty"},
    "limit": 3
}
perform_search_case(data, case_name)

data = {
    "fuzzy_filters": {"label": "parent", "range": "man"},
    "fuzzy_filters_config": {"model_name": "LaBSE", "lang": "en"},
    "exact_filters": {"termtype": "ObjectProperty"},
    "limit": 3
}
perform_search_case(data, case_name)

data = {
    "fuzzy_filters": {"label": "parent", "range": "woman"},
    "fuzzy_filters_config": {"model_name": "LaBSE", "lang": "en"},
    "exact_filters": {"termtype": "ObjectProperty"},
    "limit": 3
}
perform_search_case(data, case_name)

data = {
    "fuzzy_filters": {"label": "parent", "range": "masculine"},
    "fuzzy_filters_config": {"model_name": "LaBSE", "lang": "en"},
    "exact_filters": {"termtype": "ObjectProperty"},
    "limit": 3
}
perform_search_case(data, case_name)

data = {
    "fuzzy_filters": {"label": "parent", "range": "feminine"},
    "fuzzy_filters_config": {"model_name": "LaBSE", "lang": "en"},
    "exact_filters": {"termtype": "ObjectProperty"},
    "limit": 3
}
perform_search_case(data, case_name)



# case_name = "Simple exact filtering"
# data = {

#     "exact_filters": {"termtype": "ObjectProperty"}
# }
# perform_search_case(data, case_name)

# case_name = "Complex exact filtering"
# data = {
#     "exact_filters": {"termtype": "ObjectProperty", "domain": 'http://www.example.lirb.com/Person'}
# }
# perform_search_case(data, case_name)


# case_name = "Complex fuzzy filtering"

# data = {
#     "fuzzy_filters": {"label": "Date"},
#     "fuzzy_filters_config": {"model_name": "LaBSE"},#, "hybrid_search_field": "label"},
#     "exact_filters": {"termtype": "Class"}
# }
# perform_search_case(data, case_name)


