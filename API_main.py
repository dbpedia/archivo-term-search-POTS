from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import Filter, TargetVectors, MetadataQuery
from weaviate.util import generate_uuid5
from langchain_community.embeddings import SentenceTransformerEmbeddings
from VectorDB_creation_aux import *
from dotenv import load_dotenv
import os
from datetime import datetime
from weaviate_client import get_weaviate_client
import logging
from VectorDB_creation import get_properties_from_collection

# Initialize Flask app
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS) for all routes
CORS(app)

# Load environment variables from a .env file
load_dotenv()

# Load various API keys and configurations from environment variables
wcd_url = os.getenv("WCD_URL")
wcd_api_key = os.getenv("WCD_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
hf_key = os.getenv("HF_KEY")
weaviate_address = os.getenv("WEAVIATE_ADDRESS")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE")
DEFAULT_LIMIT = int(os.getenv("DEFAULT_NO_OF_SEARCH_RESULTS"))

# Configure logging for error tracking
logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'{__name__}_ERRORS.log', encoding='utf-8', level=logging.ERROR)

# Create a client instance for Weaviate (used for querying)
with get_weaviate_client() as client:
    create_new = False

    # List of model names to use for embedding
    models = ["LaBSE"]

    # Mappings between model names (formatted to snake_case) and their corresponding SentenceTransformerEmbedding instances
    models = {x.replace("-", "_"): SentenceTransformerEmbeddings(model_name=x) for x in models}

    # Collection names
    collections = ['DataProperties', 'ObjectProperties', 'Classes', 'Individuals', 'RDFtypes']

    # Translation map for term types to collection name
    collections_translation = {
        'DataProperty': 'DataProperties',
        'ObjectProperty': 'ObjectProperties',
        'Class': 'Classes',
        'Individual': 'Individuals',
        'RDFtype': 'RDFtypes'
    }

    # Function to validate the provided filters in the search request
    def validate_filters(data):
        fuzzy_filters = data.get("fuzzy_filters")
        fuzzy_filters_config = data.get("fuzzy_filters_config")
        exact_filters = data.get("exact_filters")

        # Check if either fuzzy or exact filters are present
        if not fuzzy_filters and not exact_filters:
            return False, "At least one of 'fuzzy_filters' or 'exact_filters' must be provided."

        # Validate fuzzy filters
        if fuzzy_filters:
            if not isinstance(fuzzy_filters, dict):
                return False, "'fuzzy_filters' must be a dictionary."

            if fuzzy_filters_config:
                valid_fuzzy_config_keys = ["model_name", "lang", "hybrid_search_field"]
                for key in fuzzy_filters_config:
                    if key not in valid_fuzzy_config_keys:
                        return False, f"Invalid key in 'fuzzy_filters_config': '{key}'. Valid keys: {valid_fuzzy_config_keys}"
            else:
                return False, "'fuzzy_filters_config' must be passed when passing 'fuzzy_filters'."

        # Validate exact filters
        if exact_filters:
            if not isinstance(exact_filters, dict):
                return False, "'exact_filters' must be a dictionary."

            # Check datatype value and validate key names against the collection's properties
            datatype_value = exact_filters.get("termtype")
            if datatype_value not in collections_translation:
                return False, f"Invalid value for exact_filter['datatype']: '{datatype_value}'. Valid values: {list(collections_translation.keys())}"
            else:
                translation = collections_translation[datatype_value]
                valid_exact_keys = [x.name for x in client.collections.get(name=translation).config.get().properties]

                for key in exact_filters:
                    if key not in valid_exact_keys and key != "termtype":
                        return False, f"Invalid key in 'exact_filters': '{key}'. Valid keys for selected term type '{datatype_value}' are: {valid_exact_keys}"
                    
                # Validate fuzzy filters against valid exact properties
                if fuzzy_filters:
                    valid_fuzzy_keys = [x.name for x in client.collections.get(name=translation).config.get().properties]
                    for key in fuzzy_filters:
                        if key not in valid_fuzzy_keys:
                            return False, f"Invalid key in 'fuzzy_filters': '{key}'. Valid keys for selected term type '{datatype_value}' are: {valid_fuzzy_keys}"

        return True, ""  # Return True if all checks pass

    # Function to build the filters to query Weaviate
    def build_filters(filtersDict):
        combined_filter = None

        # Loop through the dictionary and create filters
        for key, value in filtersDict.items():
            if key != "termtype" and key != "language":
                # Create a filter for the property
                current_filter = Filter.by_property(key).equal(value)

                # Combine filters using logical AND (&)
                if combined_filter is None:
                    combined_filter = current_filter
                else:
                    combined_filter = combined_filter & current_filter
        return combined_filter

    # Function to query a collection with fuzzy filters and embeddings
    def query_collection(model_name, target_collection, signature_properties_to_consider, reference_properties_to_consider, hybrid_property, built_filters, desired_language, limit):
        
        # Embed the query properties using the specified model
        signature_property_embeddings = {x: models[model_name].embed_query(signature_properties_to_consider[x]) for x in signature_properties_to_consider}
        reference_property_embeddings = {x: models[model_name].embed_query(reference_properties_to_consider[x]) for x in reference_properties_to_consider}

        translation = collections_translation[target_collection]
        collection_properties = [x.name[0].upper() + x.name[1:] for x in client.collections.get(name=translation).config.get().properties]

        # Check if the properties are available in the collection
        for signature_property in signature_properties_to_consider:
            if signature_property not in collection_properties:
                raise Exception(f"{translation} collection not searched -> Property '{signature_property}' not found in collection '{translation}'")

        for reference_property in reference_properties_to_consider:
            if reference_property not in collection_properties:
                raise Exception(f"{translation} collection not searched -> Property '{reference_property}' not found in collection '{translation}'")

        hybrid_property_str = hybrid_property

        # Get the collection object
        collection = client.collections.get(name=translation)
        named_vectors = collection.config.get().vector_config.keys()

        named_vectors_to_search = {}

        # Check which vectors need to be searched based on property and language
        for vector_name in named_vectors:
            if "___CP_SEPARATOR___" in vector_name:
                if reference_properties_to_consider:
                    original_vector, copy_vector_info = vector_name.split("___CP_SEPARATOR___")
                    property_to_find, target_collection, index = copy_vector_info.split("___")
                    vectorizer, prop, language = original_vector.split("___")

                    # Add vectors that match the model and language to the search
                    if vectorizer == model_name and language == desired_language and property_to_find in reference_properties_to_consider:
                        if signature_properties_to_consider:
                            if prop in signature_properties_to_consider:
                                named_vectors_to_search[vector_name] = reference_property_embeddings[property_to_find]
                        else:
                            named_vectors_to_search[vector_name] = reference_property_embeddings[property_to_find]
            else:
                vectorizer, prop, language = vector_name.split("___")
                if vectorizer == model_name and language == desired_language and prop in signature_properties_to_consider:
                    named_vectors_to_search[vector_name] = signature_property_embeddings[prop]

        target_vectors = list(named_vectors_to_search.keys())

        # Adjust vector scores based on the query type (signature vs reference properties)
        if signature_properties_to_consider and reference_properties_to_consider:
            target_vectors = TargetVectors.relative_score({x: (0.5 if "___CP_SEPARATOR___" not in x else 0.33 ) for x in target_vectors})
        else:
            target_vectors = TargetVectors.relative_score({x: 1/len(target_vectors) for x in target_vectors})

        # Perform the query without hybrid property
        if not hybrid_property:
            results = collection.query.near_vector(
                    near_vector=named_vectors_to_search,
                    target_vector=target_vectors,
                    filters=built_filters,
                    limit=limit,
                    return_metadata=MetadataQuery(distance=True)
                ).objects
        else:
            # Perform hybrid search (if applicable)
            results = collection.query.hybrid(
                    query=hybrid_property_str,
                    vector=named_vectors_to_search,
                    target_vector=target_vectors,
                    filters=built_filters,
                    limit=limit,
                    return_metadata=MetadataQuery(distance=True)
                ).objects

        return [{"object": x.properties, "distance": x.metadata.distance} for x in results]

    # Function to perform exact search based on provided filters
    def pure_exact_search(exact_filters, limit):
        
        filters = build_filters(exact_filters)

        target_collections = collections
        if "termtype" in exact_filters:
            target_collections = [exact_filters["termtype"]]

        results = {}

        for collection_name in target_collections:
            results[collection_name] = [{"object": x.properties, "distance": "N/A"} for x in client.collections.get(name=collections_translation[collection_name]).query.fetch_objects(filters=filters, limit=limit).objects]

        return results

    # Function to perform fuzzy search with hybrid property
    def fuzzy_search(fuzzy_filters, fuzzy_filters_config, exact_filters, hybrid_property, language, limit):
        print("Doing fuzzy search")
        target_collection = None
        built_filters = None

        # Handle exact filters if present
        if exact_filters:
            target_collection = exact_filters.get("termtype")
            built_filters = build_filters(exact_filters)

        # Define which properties are considered signature and reference properties
        signature_properties = ["Label", "Description"]
        reference_properties = ["Domain", "Range", "Subclass", "Superclass"]

        # Filter properties to consider for search
        signature_properties_to_consider = {x.capitalize(): fuzzy_filters[x] for x in fuzzy_filters if x.capitalize() in signature_properties}
        reference_properties_to_consider = {x.capitalize(): fuzzy_filters[x] for x in fuzzy_filters if x.capitalize() in reference_properties}

        model_name = fuzzy_filters_config.get("model_name")

        # Perform query for the given target collection
        results = {}
        if target_collection:
            results[target_collection] = query_collection(model_name, target_collection, signature_properties_to_consider, reference_properties_to_consider, hybrid_property, built_filters, language, limit)
        else:
            # Query all collections if no target collection is specified
            for collection_name in collections_translation:
                try:
                    results[collections_translation[collection_name]] = query_collection(model_name, collection_name, signature_properties_to_consider, reference_properties_to_consider, hybrid_property, built_filters, language, limit)
                except Exception as e:
                    logging.error("Failed to fetch data from %s: %s\n%s", collection_name, e, traceback.format_exc())
                    results[collections_translation[collection_name]] = "".join(str(e))
        return results

    # Main search function that decides between exact and fuzzy search
    def search(data):
        fuzzy_filters = data.get("fuzzy_filters")
        fuzzy_filters_config = data.get("fuzzy_filters_config")
        exact_filters = data.get("exact_filters")
        limit = data.get("limit", DEFAULT_LIMIT)

        if fuzzy_filters_config:
            language = fuzzy_filters_config.get("language")
            if not "language" in fuzzy_filters_config:
                language = fuzzy_filters_config.get("lang", DEFAULT_LANGUAGE)
        else:
            language = DEFAULT_LANGUAGE
            

        # Decide whether to perform fuzzy or exact search
        if fuzzy_filters:
            hybrid_search_field = fuzzy_filters_config.get("hybrid_search_field")
            results = fuzzy_search(fuzzy_filters, fuzzy_filters_config, exact_filters, hybrid_search_field, language, limit)
        else:
            results = pure_exact_search(exact_filters, limit)

        return results, 200

    # Flask route to handle the search request
    @app.route('/search', methods=['POST'])
    def search_endpoint():
        try:
            data = request.json

            # Validate filters in the request data
            is_valid, error_message = validate_filters(data)
            if not is_valid:
                return jsonify({"error": error_message}), 400
            
            # Perform the search and return the results
            results, status_code = search(data)
            return jsonify(results), status_code

        except Exception as e:
            logging.error(e, traceback.format_exc())
            return jsonify("Internal server error"), 400
    
    # Run the Flask app
    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=8014, debug=True)
