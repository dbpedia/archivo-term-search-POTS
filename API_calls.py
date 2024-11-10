import json
import csv
import os
from dotenv import load_dotenv
import requests
import tabulate

# Define the API endpoint
load_dotenv()
api_hostname = os.getenv("SEARCH_API_BASE_URI","127.0.0.1:8014")

api_endpoint = f"http://{api_hostname}/search"
#"http://127.0.0.1:9090/search"

# Define the model name
model_name = "LaBSE"
def perform_search_case(case_data, case_info, output_filename="search_results.txt"):
    print()
    print("--------------------------- QUERY ---------------------------")
    
    # Print the query parameters as a small table
    query_data = [[k, v] for k, v in case_data.items()]  # Create a list of query parameters
    query_table = tabulate.tabulate(query_data, headers=["Query Parameter", "Value"], tablefmt="fancy_grid")
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Include the model_name in the case data
    response = requests.post(api_endpoint, headers=headers, data=json.dumps(case_data))
    
    if response.status_code == 200:
        
        response_data = json.loads(response.text)
        print("Results retrieved successfully!")

        # Prepare results list for tabulate
        all_tables = []
        
        errors = []
        for collection in response_data:
            acc = []
            row_format = []
            table_data = []
            result_count = 0
            for obj in response_data[collection]:
                
                
                if type(obj) == dict:
                    result_count += 1
                    obj_data = obj["object"]
                    distance = obj["distance"]
                    row_format = list(obj_data.keys())
                    # Prepare the row for each result
                    row = [
                        collection,
                        result_count,
                    ]
                    row_format = sorted(row_format, key=custom_sort_key)
                    row.extend([obj_data.get(val) for val in row_format])
                    row.append(distance)
                    table_data.append(row)
                else:
                    acc.append(obj)
            all_tables.append([table_data, row_format])
            if acc:
                errors.append("".join(acc))
            
        with open(output_filename, "a", encoding="utf-8") as f:
            f.write("\n\n")
            f.write("CASE: "+ case_info+"\n")
            f.write(query_table)  # Save the query table
        
        for table_data, row_format in all_tables:
            if row_format:
                row_format = [x[0].upper()+x[1:] for x in row_format]
                
                # Define the headers for the results table
                result_headers = ["Collection", "Result #"]+row_format+["Distance"]

                # Generate the result table using tabulate
                result_table_str = tabulate.tabulate(table_data, headers=result_headers, tablefmt="grid")
                
                
                # Save the query and result tables to a text file
                with open(output_filename, "a", encoding="utf-8") as f:
                    f.write("\n")  # Add space between query and results
                    f.write(result_table_str)  # Save the result table
                    f.write("\n")
            
        
        error_headers = ["Notes"]

        # Generate the result table using tabulate
        error_table_str = tabulate.tabulate([[error] for error in errors], headers=error_headers, tablefmt="grid")

        
        # Save the query and result tables to a text file
        with open(output_filename, "a", encoding="utf-8") as f:
            f.write("\n")
            f.write(error_table_str)  # Save the result table
            f.write("\n\n\n")
    

    else:
        

        error_headers = ["! ! ! ERROR ! ! !"]
        print(f"{case_info} failed with status code {response.status_code}:")
        
        error = json.loads(response.text)["error"]

        # Generate the result table using tabulate
        error_table_str = tabulate.tabulate([[error]], headers=error_headers, tablefmt="grid", stralign='center')

        
        with open(output_filename, "a", encoding="utf-8") as f:
            f.write("\n")
            f.write("CASE: "+ case_info+"\n")
            f.write(query_table)
            f.write("\n")
            f.write(error_table_str)  # Save the result table
            f.write("\n\n\n")
        

def custom_sort_key(element):
    # Assign a "priority" for sorting.
    # If the element is "Domain", it gets priority 1, "Range" gets priority 2.
    # Similarly, "Superclass" gets priority 3, "Subclass" gets priority 4.
    # All other elements get a higher priority value (for general sorting).
    
    priority = {
        "label": 1,
        "termIRI": 2,
        "description": 3,
        "domain": 4,
        "range": 5,
        "superclass": 6,
        "subclass": 7,
    }
    
    # For any element that isn't specifically listed, we assign it a higher priority
    return priority.get(element, 100)  # Return 100 for elements that are not specifically listed



case_info = "INVALID KEYS 1 (missing everything)"

data = {

}
perform_search_case(data, case_info)

case_info = "INVALID KEYS 2 (missing fuzzy_config)"

data = {
    "fuzzy_filters": {"label": "parent"}
}
perform_search_case(data, case_info)


case_info = "INVALID KEYS 3 (searching for invalid property in collection)"

data = {
    "fuzzy_filters": {"label": "parent", "subclass": "male"},
    "fuzzy_filters_config": {"model_name": "LaBSE", "lang": "en"},
    "exact_filters": {"termtype": "ObjectProperty"},
}
perform_search_case(data, case_info)

case_info = "VALID 1 (simple fuzzy search)"
data = {
    "fuzzy_filters": {"label": "parent"},
    "fuzzy_filters_config": {"model_name": "LaBSE", "lang": "en"},
    "limit": 3
}
perform_search_case(data, case_info)

case_info = "VALID 2 (complex fuzzy search)"

data = {
    "fuzzy_filters": {"label": "parent", "range": "male"},
    "fuzzy_filters_config": {"model_name": "LaBSE", "lang": "en"},
    "limit": 3
}
perform_search_case(data, case_info)

case_info = "VALID 3 (simple exact search - Collection)"

data = {
    "exact_filters": {"termtype": "ObjectProperty"},
    "limit": 3
}
perform_search_case(data, case_info)

case_info = "VALID 4 (fuzzy + exact search)"

data = {
    "fuzzy_filters": {"label": "parent"},
    "fuzzy_filters_config": {"model_name": "LaBSE", "lang": "en"},
    "exact_filters": {"termtype": "ObjectProperty"},
    "limit": 3
}
perform_search_case(data, case_info)


