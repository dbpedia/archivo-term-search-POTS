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

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

load_dotenv()
wcd_url = os.getenv("WCD_URL")
wcd_api_key = os.getenv("WCD_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
hf_key = os.getenv("HF_KEY")
# weaviate_port = int(os.getenv("WEAVIATE_PORT"))
# weaviate_port_grpc = int(os.getenv("WEAVIATE_PORT_GRPC"))
weaviate_address = os.getenv("WEAVIATE_ADDRESS")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE")
DEFAULT_LIMIT = int(os.getenv("DEFAULT_NO_OF_SEARCH_RESULTS"))

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'{__name__}_ERRORS.log', encoding='utf-8', level=logging.ERROR)

# Create a client instance
with get_weaviate_client() as client:
    create_new = False

    # models = ["LaBSE","all-MiniLM-L6-v2","all-MiniLM-L12-v2","all-distilroberta-v1","paraphrase-multilingual-MiniLM-L12-v2","multi-qa-mpnet-base-cos-v1"]
    models = ["LaBSE"]

    # # Mappings between model names (formatted to _ format) and model instances
    models = {x.replace("-", "_"): SentenceTransformerEmbeddings(model_name=x) for x in models}

    collections = ['DataProperties', 'ObjectProperties', 'Classes', 'Individuals', 'RDFtypes'] # TODO: Add "Ontologies"

    collections_translation = {
        'DataProperty': 'DataProperties',
        'ObjectProperty': 'ObjectProperties',
        'Class': 'Classes',
        'Individual': 'Individuals',
        'RDFtype': 'RDFtypes'
    }

    def validate_filters(data):
        # Check for fuzzy filters
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

            valid_fuzzy_keys = ["label", "description", "superclass", "sublcass", "domain", "range"] # TODO: This should be collection-sensitive
            for key in fuzzy_filters:
                if key not in valid_fuzzy_keys:
                    return False, f"Invalid key in 'fuzzy_filters': '{key}'. Valid keys: {valid_fuzzy_keys}"

        # Validate exact filters
        if exact_filters:
            if not isinstance(exact_filters, dict):
                return False, "'exact_filters' must be a dictionary."

            # Check datatype value
            datatype_value = exact_filters.get("termtype")
            if not datatype_value in collections_translation:
                return False, f"Invalid value for exact_filter['datatype']: '{datatype_value}'. Valid values: {list(collections_translation.keys())}"
            else:
                translation = collections_translation[datatype_value]
                valid_exact_keys = [x.name for x in client.collections.get(name=translation).config.get().properties]

                for key in exact_filters:
                    if key not in valid_exact_keys and key != "termtype":
                        return False, f"Invalid key in 'exact_filters': '{key}'. Valid keys: {valid_exact_keys}"

            # TODO: check which keys are passed and see if they make sense


        return True, ""

    def build_filters(filtersDict):
        combined_filter = None

        # Loop through the dictionary and create filters
        for key, value in filtersDict.items():
            if key != "termtype" and key != "language":
                # Depending on your key names, choose the correct filter type
                current_filter = Filter.by_property(key).equal(value)

                # Combine filters using logical AND (&)
                if combined_filter is None:
                    combined_filter = current_filter
                else:
                    combined_filter = combined_filter & current_filter
        return combined_filter

    def query_collection(model_name, target_collection, signature_properties_to_consider, reference_properties_to_consider, hybrid_property, built_filters, desired_language, limit):
        signature_property_embeddings = {x: models[model_name].embed_query(signature_properties_to_consider[x]) for x in signature_properties_to_consider}
        reference_property_embeddings = {x: models[model_name].embed_query(reference_properties_to_consider[x]) for x in reference_properties_to_consider}

        hybrid_property_str = hybrid_property

        # Get the collection object from the client (Assuming `client` is properly initialized)
        collection = client.collections.get(name=collections_translation[target_collection])
        named_vectors = collection.config.get().vector_config.keys()

        #print(named_vectors)
        named_vectors_to_search = {}
        for vector_name in named_vectors:
            #print(vector_name)
            if "___CP_SEPARATOR___" in vector_name:
                if reference_properties_to_consider:
                    original_vector, copy_vector_info = vector_name.split("___CP_SEPARATOR___")
                    property_to_find, target_collection, index = copy_vector_info.split("___")
                    vectorizer, prop, language = original_vector.split("___")

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

        
        if signature_properties_to_consider and reference_properties_to_consider:
            target_vectors = TargetVectors.relative_score({x: (0.5 if "___CP_SEPARATOR___" not in x else 0.33 ) for x in target_vectors})
        else:
            target_vectors = TargetVectors.relative_score({x: 1/len(target_vectors) for x in target_vectors})


        if not hybrid_property:

            results = collection.query.near_vector(
                    near_vector=named_vectors_to_search,
                    target_vector=target_vectors,
                    filters=built_filters,
                    limit=limit,
                    return_metadata=MetadataQuery(distance=True)
                ).objects
        else:

            results = collection.query.hybrid(
                    query=hybrid_property_str,
                    vector=named_vectors_to_search,
                    target_vector=target_vectors,
                    filters=built_filters,
                    limit=limit,
                    return_metadata=MetadataQuery(distance=True)
                ).objects


        return [{"object": x.properties, "distance": x.metadata.distance} for x in results]

        # except Exception as e:
        #     print(e)

    def pure_exact_search(exact_filters):
        filters = build_filters(exact_filters)

        target_collections = collections
        if "termtype" in exact_filters:
            target_collections = [exact_filters["termtype"]]

        results = {}

        for collection_name in target_collections:

            results[collection_name] = [{"object": x.properties, "distance": "N/A"} for x in client.collections.get(name=collections_translation[collection_name]).query.fetch_objects(filters=filters, limit=DEFAULT_LIMIT).objects]

        return results

    def fuzzy_search(fuzzy_filters, fuzzy_filters_config, exact_filters, hybrid_property, language, limit):
        print("Doing fuzzy search")
        target_collection = None
        built_filters = None

        if exact_filters:
            target_collection = exact_filters.get("termtype")

            built_filters = build_filters(exact_filters)

        signature_properties = ["Label", "Description"]

        reference_properties = ["Domain", "Range", "Subclass", "Superclass"]

        signature_properties_to_consider = {x.capitalize(): fuzzy_filters[x] for x in fuzzy_filters if x.capitalize() in signature_properties}

        reference_properties_to_consider = {x.capitalize(): fuzzy_filters[x] for x in fuzzy_filters if x.capitalize() in reference_properties}

        # Ex:
        model_name = fuzzy_filters_config.get("model_name")

        results = {}
        if target_collection:
            results[target_collection] = query_collection(model_name, target_collection, signature_properties_to_consider, reference_properties_to_consider, hybrid_property, built_filters, language, limit)

        else:
            for collection_name in collections_translation:
                try:
                    results[collections_translation[collection_name]] = query_collection(model_name, collection_name, signature_properties_to_consider, reference_properties_to_consider, hybrid_property, built_filters, language, limit)
                except Exception as e:
                    logging.error("Failed to fetch data from %s: %s\n%s", collection_name, e, traceback.format_exc())
                    results[collections_translation[collection_name]] = "Error: Failed to fetch data from the collection"
        return results

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
            limit = DEFAULT_LIMIT

        # PURE FUZZY SEARCH
        if fuzzy_filters:

            hybrid_search_field = fuzzy_filters_config.get("hybrid_search_field")


            results = fuzzy_search(fuzzy_filters, fuzzy_filters_config, exact_filters, hybrid_search_field, language, limit)


        else:

            # PURE EXACT SEARCH
            results = pure_exact_search(exact_filters)

        return results, 200

    @app.route('/search', methods=['POST'])
    def search_endpoint():

        try:
            data = request.json

            # Validate filters
            is_valid, error_message = validate_filters(data)
            if not is_valid:

                return jsonify({"error": error_message}), 400
            
            # Proceed with search operation
            results, status_code = search(data)

            return jsonify(results), status_code
        except Exception as e:
            logging.error(e, traceback.format_exc())
            return jsonify("Internal server error"), 400
    
    if __name__ == '__main__':
        app.run(host='127.0.0.1', port=8014, debug=True)