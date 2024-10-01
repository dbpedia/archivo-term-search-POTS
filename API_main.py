from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import Filter, TargetVectors
from weaviate.util import generate_uuid5
from langchain_community.embeddings import SentenceTransformerEmbeddings
from VectorDB_creation_aux import *
from dotenv import load_dotenv
import os
from datetime import datetime


app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

load_dotenv()
wcd_url = os.getenv("WCD_URL")
wcd_api_key = os.getenv("WCD_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
hf_key = os.getenv("HF_KEY")
url_endpoint = "http://95.217.207.179:8995/sparql/"
# Create a client instance
headers = {
    "X-HuggingFace-Api-Key": hf_key,
}
client = weaviate.connect_to_local(
    
    port=8085,
    grpc_port=50051,
    headers=headers

)
 
create_new = False

models = ["LaBSE","all-MiniLM-L6-v2","all-MiniLM-L12-v2","all-distilroberta-v1","paraphrase-multilingual-MiniLM-L12-v2","multi-qa-mpnet-base-cos-v1"]
model_name = models[0]

# Mappings between model names (formatted to _ format) and model instances
models = {x.replace("-", "_"): SentenceTransformerEmbeddings(model_name=x) for x in models}

def build_filters(filtersDict):
    combined_filter = None

    # Loop through the dictionary and create filters
    for key, value in filtersDict.items():
        if key != "datatype" and key != "language":
            # Depending on your key names, choose the correct filter type
            current_filter = Filter.by_property(key).equal(value)
            
            # Combine filters using logical AND (&)
            if combined_filter is None:
                combined_filter = current_filter
            else:
                combined_filter = combined_filter & current_filter
    return combined_filter

# Main search function logic
def search(model_name, query_dict):
    collections = {
        "data_property": "DataProperties",
        "object_property": "ObjectProperties",
        "individual": "Individuals",
        "class": "Classes",
        "RDF_type": "RDF_types"
    }


    # Extract filters and injections from the query_dict
    filters = query_dict.get("filters", {})
    injections = query_dict.get("context", {})
    
    language = filters.get("language", "None")
    
    
    print("LANG:", language)
    
    # Build filters (this function should be defined based on your actual filter structure)
    filters_built = build_filters(filters)

    try:
        # Embed query term or use injections
        if injections:
            vectorized_input = {model_name + "___" + i.capitalize() + f"___{language}": models[model_name].embed_query(injections[i]) for i in injections}
            #print("Query vector models:", vectorized_input.keys())
            vectorized_input = {k: vectorized_input[k] for k in vectorized_input if k.endswith("___"+language)}
            
            #print("VI after filter:", vectorized_input)
        else:
            vectorized_input = models[model_name].embed_query(query_dict["term"])
            #print("Query vector models: only one")
        
        results = {}

        # Check if filters include a specific datatype
        if "datatype" in filters:
            collection_name = collections.get(filters["datatype"])
            
            if injections:
                collection = client.collections.get(name=collection_name)
                named_vectors = collection.config.get().vector_config.keys()
                for named_vector in named_vectors:
                    if not named_vector in vectorized_input:
                        vectorized_input[named_vector] = models[model_name].embed_query(query_dict["term"])
                vectorized_input = {k: vectorized_input[k] for k in vectorized_input if k in named_vectors and k.split("___")[0] == model_name and k.split("___")[-1] == language}
                    
            if collection_name:
                results[collection_name] = search_collection(collection_name, model_name, query_dict, vectorized_input, filters_built, injections)
                
            else:
                return {"error": f"Unknown datatype: {filters['datatype']}"}, 400
        else:
            # No datatype specified, search all collections
            for collection_name in collections.values():
                
                collection = client.collections.get(name=collection_name)
                named_vectors = collection.config.get().vector_config.keys()
                
                #print(collection_name, named_vectors)
                if injections:
                    collection = client.collections.get(name=collection_name)
                    named_vectors = collection.config.get().vector_config.keys()
                    for named_vector in named_vectors:
                        if not named_vector in vectorized_input:
                            vectorized_input[named_vector] = models[model_name].embed_query(query_dict["term"])
                    vectorized_input = {k: vectorized_input[k] for k in vectorized_input if k in named_vectors and k.split("___")[0] == model_name and k.split("___")[-1] == language}
                results[collection_name] = search_collection(collection_name, model_name, query_dict, vectorized_input, filters_built, injections)
            
        return {"results": results}, 200

    except Exception as e:
        # Handle and return detailed error
        return {"error": str(e), "traceback": traceback.format_exc()}, 500

def log_search_params(collection_name, model_name, query_dict, vectorized_input, filters_built, injections, filename="search_log.txt"):
    """
    Logs search parameters to a file, checking if any parameter is a dictionary before logging.
    """
    with open(filename, "a") as log_file:
        log_file.write(f"Search timestamp: {datetime.now()}\n")
        log_file.write(f"Collection Name: {collection_name}\n")
        log_file.write(f"Model Name: {model_name}\n")
        

        log_file.write(f"Query Dict: {query_dict}\n")
        
        # Check if vectorized_input is a dict and log appropriately
        if isinstance(vectorized_input, dict):
            log_file.write(f"Vectorized Input Keys: {list(vectorized_input.keys())}\n")
        else:
            log_file.write(f"Vectorized Input: Single vector\n")
        
        # Log filters and injections
        log_file.write(f"Filters: {filters_built}\n")
        log_file.write(f"Injections: {injections}\n")
        log_file.write("-" * 50 + "\n")

def search_collection(collection_name, model_name, query_dict, vectorized_input, filters_built, injections, log_file="search_log.txt"):
    """
    Helper function to search within a specific collection and log the search parameters.
    """
    results = []
    try:
        # Log search parameters to file
        log_search_params(collection_name, model_name, query_dict, vectorized_input, filters_built, injections, log_file)

        # Get the collection object from the client (Assuming `client` is properly initialized)
        collection = client.collections.get(name=collection_name)
        named_vectors = collection.config.get().vector_config.keys()

        # "term":  "abc"
        # "model": "model1"
        # "strategy"
        
        # Handle vectorization and decide which vectors to use based on injections
        if injections:
            
            # TODO: references["domain"].namedVectors["label"]
            # reference name and namedvector name of the target reference
            
            target_vectors = list(vectorized_input.keys())
            
            # context: {"domain": "something"}
            # model1_domain, model1_domainplus_range
            inj_vectors = []
            search_term_vectors = []
            for vec in target_vectors:
                for inj in injections:
                    if inj.capitalize() in vec.split("___")[1]:
                        inj_vectors.append(vec)
                    else:
                        search_term_vectors.append(vec)
            
            total = 1000
            weighted_target_vector_dict = {}
            for inj_vec in inj_vectors:
                weighted_target_vector_dict[inj_vec] = int(total / 2)
            for search_vec in search_term_vectors:
                weighted_target_vector_dict[search_vec] = int(total / 2 )
            print(weighted_target_vector_dict)
            
            target_vectors = TargetVectors.manual_weights(weighted_target_vector_dict)
            
        else:
            target_vectors = [x for x in named_vectors if model_name in x]

        # Perform the query on the collection using the vector embeddings and filters
        if query_dict.get("hybrid", None) == "True":
            search_results = collection.query.hybrid(
                vector=vectorized_input,
                target_vector=target_vectors,
                filters=filters_built,
                limit=3
            ).objects
        else:
            search_results = collection.query.near_vector(
                near_vector=vectorized_input,
                target_vector=target_vectors, # ref
                filters=filters_built,
                limit=3
            ).objects

        # Collect the results
        for result in search_results:
            results.append(result.properties["term"])

    except Exception as e:
        print(f"Error querying {collection_name}: {e}")

    return results

def validate_filters_context(datatype, data):
    # Extract filters and context from data
    filters = data.get("filters", {})
    context = data.get("context", {})
    collections = {
        "data_property": "DataProperties",
        "object_property": "ObjectProperties",
        "individual": "Individuals",
        "class": "Classes",
        "RDF_type": "RDF_types"
    }
    VALID_FILTERS_CONTEXT = {
        "Classes": ["ontology", "label", "description", "subclass", "superclass", "language"],
        "RDF_types": ["ontology", "label", "description", "superclass", "language"],
        "Individuals": ["ontology", "label", "description", "domain", "range", "language"],
        "ObjectProperties": ["ontology", "label", "description", "domain", "range", "language"],
        "DataProperties": ["ontology", "label", "description", "domain", "range", "language"],
        "Language": ["en", "fr", "None"]
    }
    # Get the valid keys for the given datatype
    try:
        valid_keys = VALID_FILTERS_CONTEXT.get(collections[datatype])
    except:
        return False, f"Invalid datatype '{datatype}'. Available datatypes: {list(collections.keys())}"
    # Check if the datatype is valid and the provided filters/context are valid
    if valid_keys:
        # Check filters
        for key in filters:
            if key != "datatype":
                if key not in valid_keys:
                    return False, f"Invalid filter '{key}' for datatype '{datatype}. Valid filters: {valid_keys}"

        # Check context
        for key in context:
            if key not in valid_keys:
                return False, f"Invalid context '{key}' for datatype '{datatype}. Valid filters: {valid_keys}"
    else:
        return False, f"Invalid datatype '{datatype}'"

    return True, ""

@app.route('/search', methods=['POST'])
def search_endpoint():
    data = request.json
    model_name = data.get("model_name")
    term = data.get("term")

    # Validate required parameters
    if not model_name or not term:
        return jsonify({"error": "Missing required parameters: 'model_name' or 'term'"}), 400

    # Get datatype from filters if provided
    datatype = data.get("filters", {}).get("datatype")

    # If datatype is provided, validate filters and context
    if datatype:
        is_valid, error_message = validate_filters_context(datatype, data)
        if not is_valid:
            return jsonify({"error": error_message}), 400

    # Proceed with search operation
    results, status_code = search(model_name, data)
    return jsonify(results), status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090)

