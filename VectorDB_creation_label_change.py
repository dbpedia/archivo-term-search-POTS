
# This seems to be a modified version of VectorDB_creation.py, however, it is not clear what the purpose of this script is.

import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import Filter
from weaviate.util import generate_uuid5
from langchain_community.embeddings import SentenceTransformerEmbeddings
from VectorDB_creation_aux_label_change import *
from dotenv import load_dotenv
import os
import sys
import numpy as np
import traceback
from dataclasses import dataclass, field
from typing import List
from weaviate_client import get_weaviate_client
import logging
import csv

load_dotenv()
UNKNOWN_LABEL_REPLACEMENT_EMBEDDING_STRATEGY = "infer_on_iri"
# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the root logger to DEBUG to capture all log levels

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('error_logs.log', mode='a', encoding='utf-8')

# Create formatters
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Set the log level for each handler
console_handler.setLevel(logging.INFO)  # Info level and above for console
file_handler.setLevel(logging.ERROR)    # Error level and above for file

# Set the formatter for each handler
console_handler.setFormatter(console_formatter)
file_handler.setFormatter(file_formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Global variables
wcd_url = os.getenv("WCD_URL")
wcd_api_key = os.getenv("WCD_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
hf_key = os.getenv("HF_KEY")
url_endpoint =  os.getenv("SPARQL_ENDPOINT")
weaviate_address = os.getenv("WEAVIATE_ADDRESS")
create_new = os.getenv("DELETE_OLD_INDEX", False)
empty_property_embedding_strategy = os.getenv("EMPTY_ARRAY_PROPERTY_SLOT_FILLING_STRATEGY")

# Flag to designate indexer success / failure on exit
global exception_happened
exception_happened = False

# Available models
model_names = ["LaBSE"] # GET EMBEDDED MODELS
# TODO: WHEN NAMING THE VECTORS ALSO ACCOUNT FOR NON EMBEDDED

models = {x.replace("-", "_"): SentenceTransformerEmbeddings(model_name=x) for x in model_names}

languages = ["en", "de", "ru", "fr", "es", "pt", "None"]

# Generic Classes and Functions
@dataclass
class VirtualNamedVectorEmbedding:
    # On __init__ vectors (AKA named___vector___strings)
    field_name: str = field(default_factory=str)
    language: str = field(default_factory=str)
    vectorizer: str = field(default_factory=str)
    embed_strategy: str = field(default_factory=str)
    
    # CopiedVectors
    copy_relationship: str = field(default_factory=str)
    copy_relationship_index: str = field(default_factory=str)

    def __post_init__(self):
        object.__setattr__(self, 'name', f"{self.vectorizer}___{self.field_name}___{self.language}")

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, VirtualNamedVectorEmbedding):
            return self.name == other.name
        return False

def has_no_labels(result_doc):
    return not any([getattr(result_doc, f"label_{lang.lower()}", None) for lang in languages])
def create_named_vectors(item):
    # Unpack the input tuple into meaningful variable names
    field_name, language, vectorizer, embed_strategy, copy_relationship, copy_relationship_index = item
    
    # Create the original named vector embedding
    original_embedding = VirtualNamedVectorEmbedding(
        field_name=field_name,
        language=language,
        vectorizer=vectorizer,
        embed_strategy=embed_strategy,
        copy_relationship=copy_relationship,
        copy_relationship_index=copy_relationship_index,
    )

    # Initialize a list to hold the original and copied embeddings
    embeddings = [original_embedding]
    for n in range(1, 4):  # Create 3 copies of the original embedding
        copy_embedding = VirtualNamedVectorEmbedding(
            field_name=original_embedding.field_name,
            language=original_embedding.language,
            vectorizer=original_embedding.vectorizer,
            embed_strategy=original_embedding.embed_strategy,
            copy_relationship=original_embedding.copy_relationship,
            copy_relationship_index=original_embedding.copy_relationship_index,
        )
        # Name the copied embedding to reflect its relationship
        copy_embedding.name = f"{original_embedding.name}___CP_SEPARATOR___{copy_relationship}___{copy_relationship_index}___{n}"
        embeddings.append(copy_embedding)

    return embeddings  # Return the list of embeddings

def get_copied_named_vectors(all_objects, all_named_vectors):
    # Initialize dictionaries to hold embeddings and empty embeddings
    embeddings = {}
    empty_embeddings = {}
    logging.info("Filling copied named vectors mark")
    for formatted_object in all_objects:
        # Ensure a dictionary exists for each object's UUID
        if formatted_object.uuid not in embeddings:
            embeddings[formatted_object.uuid] = {}
        logging.info("Object is %s", formatted_object)
        for vector in all_named_vectors:
            if "___CP_SEPARATOR___" in vector:
                
                logger.info("vector is %s", vector)
                # Split the vector name to extract original and copy info
                original_vector_info, copy_vector_info = vector.split("___CP_SEPARATOR___")
                property_to_find, target_collection, index = copy_vector_info.split("___")
                vectorizer = vector.split("___")[0]
                prop = vector.split("___")[1]
                # Initialize empty embeddings if they haven't been created yet
                if vectorizer not in empty_embeddings:
                    empty_embeddings[vectorizer] = generate_empty_embedding(models[vectorizer])

                property_to_find = property_to_find.lower()

                # Check if the property exists in the object's properties
                if len(formatted_object.properties[property_to_find]) >= int(index):

                    target_uri = formatted_object.properties[property_to_find][int(index)-1]

                    
                    # print("property to find is", property_to_find)
                    # print("target collection is", target_collection)
                    # print("index is", index)
                    # print("vectorizer is", vectorizer)
                    # print("prop is", prop)
                    
                    if target_uri != "":
                        # Query the collection for the named vector embedding
                        result = query_collection_for_NV_embedding(target_collection, target_uri, original_vector_info)
                        logger.info("Looking for vector: %s", vector)
                        logger.info("Current object properties: %s", formatted_object.properties)
                        logger.info("Fetched the %s embedding from %s (%s collection)", original_vector_info, target_uri, target_collection)
                        if result:
                            # Store the result in the embeddings dictionary
                            embeddings[formatted_object.uuid][vector] = result
    

                        else:
                            # If the result is not found, use an empty embedding
                            embeddings[formatted_object.uuid][vector] = empty_embeddings[vectorizer]

                else:
                    if empty_property_embedding_strategy == "empty":
                        # If the property index is out of range, use an empty embedding
                        embeddings[formatted_object.uuid][vector] = empty_embeddings[vectorizer]
                    elif empty_property_embedding_strategy == "average":
                        if int(index) > 1:
                            # If the property index is out of range, fall back to average of previous vectors
                            previous_vectors = []
                            for i in range(1, int(index)):
                                # Find vectors with the same property, vectorizer, and lower index
                                previous_vector_name = vector[:-1]+str(i)

                                if previous_vector_name in embeddings[formatted_object.uuid]:
                                    previous_vectors.append(embeddings[formatted_object.uuid][previous_vector_name])

                            if previous_vectors:
                                # Compute the average of the previous vectors
                                embeddings[formatted_object.uuid][vector] = np.mean(previous_vectors, axis=0)
                            else:
        
                                logger.error("No previous vectors found for %s, this should not happen", vector)

                                raise Exception

                        
    return embeddings  # Return the mappings of UUIDs to named vector embeddings

def fill_copied_named_vectors(uuid_to_nv_mappings, target_collection):

    logger.info("Filling copied named vectors")
    collection = client.collections.get(name=target_collection)
    
    # Calculate the progress interval for logging
    tp = max(len(uuid_to_nv_mappings) / 10, 1)
    for i, uuid in enumerate(uuid_to_nv_mappings):
        if i % int(tp) == 0:
            logger.info("%d / %d", i, len(uuid_to_nv_mappings))
            logger.info(f"UUID IS {uuid}")
            #logger.info(f"mappings are {uuid_to_nv_mappings[uuid]}")
        if uuid_to_nv_mappings[uuid]:

            # Update the collection with the vector mappings
            collection.data.update(uuid, vector=uuid_to_nv_mappings[uuid])

def query_collection_for_NV_embedding(target_collection, target_uri, target_named_vector):
    # Fetch the specified collection
    collection = client.collections.get(name=target_collection)

    # Attempt to fetch the object by its UUID
    data_object = collection.query.fetch_object_by_id(
        uuid=generate_uuid5(target_uri),  # Generate the UUID for the target URI
        include_vector=True  # Include vector data in the response
    )
    
    if data_object:
        try:
            # Return the specific named vector if it exists
            return data_object.vector[target_named_vector]
        except KeyError:
            return None    
    return None  # Return None if the object is not found

def fetch_all_objects(collection):
    # Fetch all objects from the specified collection
    return [x for x in client.collections.get(collection).iterator()]

def fetch_all_named_vectors(collection):
    # Fetch all named vectors from the specified collection
    res = list(client.collections.get(collection).config.get().vector_config.keys())
    if collection == "RDFtypes":
        return [r for r in res if not "Domain" in r and not "Range" in r and not "Subclass" in r]
    return res

# Object Property Collection Functions
def get_object_property_collection_mappings():

    methodologies = {
        "Label": embed_using_label,
        "Description": embed_using_desc,
        "Domain": embed_using_domain,
        "Range": embed_using_range
    }
    return methodologies  # Return both models and methodologies

def format_object_property_query_results(endpoint_query_results, methodologies, models, all_named_vectors):
    formatted_objects = {}
    for i, result_doc in enumerate(endpoint_query_results):
        
        # Format each result document into a structured object
        formatted_object = {
            "TermIRI": result_doc.termIRI,
          
            "Ontology": result_doc.ontology,

            "Domain": result_doc.domain,
            "Range": result_doc.range,
        }
            # Add "label_BLANK" keys for each language in the languages list
        for lang in languages:
            label_key = f"label_{lang.lower()}"

            formatted_object[label_key] = getattr(result_doc, label_key, None)   
            desc_key = f"description_{lang.lower()}"
            formatted_object[desc_key] = getattr(result_doc, desc_key, "")
        # Infer the label from TermIRI if it's not provided
        if has_no_labels(result_doc):
            
            if UNKNOWN_LABEL_REPLACEMENT_EMBEDDING_STRATEGY == "infer_on_iri":
                if "#" in result_doc.termIRI:
                    formatted_object["label_none"] = [result_doc.termIRI.split("#")[1]]
                elif "/" in result_doc.termIRI:
                    formatted_object["label_none"] = [result_doc.termIRI.split("/")[1]]

        embeddings = {}  # Dictionary to hold embeddings for the object
        for vector in all_named_vectors:
            # Skip vectors with copy separator as they are handled elsewhere
            if "___CP_SEPARATOR___" not in vector.name:
                model = vector.vectorizer
                field_name = vector.field_name
                case_lang = vector.language

               
                
                # Generate embeddings only if the language matches
                if field_name in ["Label", "Description"]:
                    doc_has_field_in_wanted_language = hasattr(result_doc, f"{field_name.lower()}_{case_lang.lower()}")
                    if doc_has_field_in_wanted_language:
                        
                        embeddings[vector.name] = methodologies[field_name](result_doc, models[model], case_lang)
                    
                else:
                    
                    embeddings[vector.name] = methodologies[field_name](result_doc, models[model])
          

        # Generate a UUID for the object
        uuid = generate_uuid5(result_doc.termIRI)
        if uuid in formatted_objects:
            old_object = formatted_objects[uuid]
            formatted_objects[uuid] = [old_object[0].update_info(result_doc), old_object[1] | embeddings, uuid]
        else:
            formatted_objects[uuid] = [formatted_object, embeddings, uuid]  # Append formatted data

    
    return list(formatted_objects.values())  # Return the list of formatted objects

def create_object_property_collection():
    logger.info("Creating ObjectProperties collection")
    logger.info("Fetching SPARQL endpoint query results")
    # Fetch data from the endpoint for ObjectProperties
    endpoint_query_results =  fetch_data_from_endpoint(url_endpoint, type="ObjectProperties")

    # Get the models and methodologies for embedding
    methodologies = get_object_property_collection_mappings()

    # Create combinations for different fields, languages, and models
    all_combos = []
    for field_name in ["Label", "Description"]:
        for lang in languages:
            for model in models:
                for copy_relationship, collection in [("Domain", "Classes"), ("Range", "Classes")]:
                    all_combos.append([field_name, lang, model, "default", copy_relationship, collection])

    # Create named vectors for all combinations
    all_named_vectors = []
    for item in all_combos:
        all_named_vectors.extend(create_named_vectors(item))  # Use extend to add all vectors to the list

    # Remove duplicates by converting the list to a set
    all_named_vectors = set(all_named_vectors)  

    print("Formatting results for upload")
    # Format objects for upload
    formatted_objects_for_upload = format_object_property_query_results(endpoint_query_results, methodologies, models, all_named_vectors)

    # Configure vectorizer for the named vectors
    vectorizer_config = [wvc.config.Configure.NamedVectors.none(name=x.name) for x in all_named_vectors]

    logger.info("Creating collection")
    # Create a new collection with the specified properties and vectorizer config
    collection = client.collections.create(
        name="ObjectProperties",
        description="Text2kg benchmark object properties",
        vectorizer_config=vectorizer_config,
        properties=[
            wvc.config.Property(name="TermIRI", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="RDF_type", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="Label", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="Description", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="Domain", data_type=wvc.config.DataType.TEXT_ARRAY),
            wvc.config.Property(name="Range", data_type=wvc.config.DataType.TEXT_ARRAY),
            wvc.config.Property(name="Language", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="Ontology", data_type=wvc.config.DataType.TEXT),
        ],
    )

    # Upload the formatted object data
    logger.info("Uploading data")
    objects_to_upload = [wvc.data.DataObject(properties=d[0], vector=d[1], uuid=d[2]) for d in formatted_objects_for_upload]

    # Split objects into batches for uploading
    batches = split_list(objects_to_upload, 4)
    try:
        successes = 0
        for i, batch in enumerate(batches):
            logger.info("Uploading batch %d to %s", i + 1, collection.name)

            result = collection.data.insert_many(batch)  # Insert batch
            inserted_docs = batch  # The original batch contains the inserted documents
            successes += len(inserted_docs)  # Count successfully inserted documents

            logger.info("Successfully inserted %d documents in batch %d: %s", len(inserted_docs), i + 1, [Vectorless(x) for x in inserted_docs])

        logger.info("Total successfully inserted documents: %d", successes)

        
        return {"uploaded": successes}
    except Exception as e:
        logger.error("Error during insert_many: %s", traceback.format_exc(), exc_info=e)
        exception_happened = True
        
        return {"error": True}
        
def fill_object_property_copied_named_vectors():
    
    # Fetch all objects and named vectors to fill the copied named vectors
    all_objects = fetch_all_objects(collection="ObjectProperties")
    all_named_vectors = fetch_all_named_vectors(collection="ObjectProperties")
    
    # Get the mappings for the copied named vectors
    uuid_to_nv_mappings = get_copied_named_vectors(all_objects, all_named_vectors)
    
    # Fill the copied named vectors in the collection
    fill_copied_named_vectors(uuid_to_nv_mappings, "ObjectProperties")


# Class Collection Functions

def get_class_collection_mappings():

    
    # Define methodologies for embedding based on different fields
    methodologies = {
        "Label": embed_using_label,
        "Description": embed_using_desc,
        "Subclass": embed_using_subclass, 
        "Superclass": embed_using_superclass
    }
    
    return methodologies  # Return both models and methodologies

def format_class_query_results(endpoint_query_results, methodologies, models, all_named_vectors):
    logger.info("Formatting objects")  # Indicate the start of formatting objects
    formatted_objects = {}  # List to hold the formatted objects

    for i, result_doc in enumerate(endpoint_query_results):
        # Create a dictionary to store properties of the class
        formatted_object = {
            "TermIRI": result_doc.termIRI,
          
            "Ontology": result_doc.ontology,
            
            "Subclass": result_doc.subclass,
            "Superclass": result_doc.superclass,
            
        }
        

        # Add "label_BLANK" keys for each language in the languages list
        for lang in languages:
            label_key = f"label_{lang.lower()}"

            formatted_object[label_key] = getattr(result_doc, label_key, [])
            
            desc_key = f"description_{lang.lower()}"
            formatted_object[desc_key] = getattr(result_doc, desc_key, "")
            
        # Infer the label from TermIRI if it's not provided
        if has_no_labels(result_doc):
            
            if UNKNOWN_LABEL_REPLACEMENT_EMBEDDING_STRATEGY == "infer_on_iri":
                if "#" in result_doc.termIRI:
                    formatted_object["label_none"] = [result_doc.termIRI.split("#")[1]]
                elif "/" in result_doc.termIRI:
                    formatted_object["label_none"] = [result_doc.termIRI.split("/")[1]]
        print("Line from query is", result_doc)
        embeddings = {}  # Dictionary to hold embeddings for the object
        for vector in all_named_vectors:
            # Skip vectors with copy separator as they are handled elsewhere
            if "___CP_SEPARATOR___" not in vector.name:
                model = vector.vectorizer
                field_name = vector.field_name
                case_lang = vector.language
                
                # Generate embeddings only if the language matches
                if field_name in ["Label", "Description"]:
                    doc_has_field_in_wanted_language = hasattr(result_doc, f"{field_name.lower()}_{case_lang.lower()}")
                    if doc_has_field_in_wanted_language:
                        
                        if case_lang != "None":
                            print("Putting", field_name, "in",vector.name) 
                            embeddings[vector.name] = methodologies[field_name](result_doc, models[model], case_lang)
                        else:
                            if len(getattr(result_doc, f"{field_name.lower()}_none")) > 0:
                                print("Putting", field_name, "in",vector.name) 
                                embeddings[vector.name] = methodologies[field_name](result_doc, models[model], "none")
                    
                else:
                    print("Putting", field_name, "in",vector.name) 
                    embeddings[vector.name] = methodologies[field_name](result_doc, models[model])

        # Generate a UUID for the object based on its TermIRI
        uuid = generate_uuid5(result_doc.termIRI)
        if uuid in formatted_objects:
            old_object = formatted_objects[uuid]
            formatted_objects[uuid] = [old_object[0].update_info(result_doc), old_object[1] | embeddings, uuid]
        else:
            formatted_objects[uuid] = [formatted_object, embeddings, uuid]  # Append formatted data

    return list(formatted_objects.values())  # Return the list of formatted objects

def create_class_collection():
    logger.info("Creating class collection")  # Indicate the start of collection creation
    logger.info("Fetching SPARQL endpoint query results")
    # Fetch data from the endpoint for Classes
    endpoint_query_results = fetch_data_from_endpoint(url_endpoint, type="Classes")
    print("ENDPOINT QUERY RESULTS:")
    for result_doc in endpoint_query_results:
        for attr in dir(result_doc):
                if attr.startswith("label"):
                    print("FOUND attr", attr, "in", result_doc)
    # Get the models and methodologies for embedding
    methodologies = get_class_collection_mappings()

    # Prepare combinations for various fields and relationships
    all_combos = []
    for field_name in ["Label", "Description"]:
        for lang in languages:
            for model in models:
                for copy_relationship, collection in [("Subclass", "Classes"), ("Superclass", "Classes")]:
                    all_combos.append([field_name, lang, model, "default", copy_relationship, collection])
    logger.info(f"all combos are {all_combos}")  # Indicate the start of named vector creation
    # Create named vectors for all combinations
    all_named_vectors = []
    for item in all_combos:
        all_named_vectors.extend(create_named_vectors(item))  # Use extend to add all vectors to the list

    all_named_vectors = set(all_named_vectors)  # Remove duplicates by converting the list to a set
    print("Formatting results for upload")
    # Format objects for upload
    formatted_objects_for_upload = format_class_query_results(endpoint_query_results, methodologies, models, all_named_vectors)
    print("outside", [x[0] for x in formatted_objects_for_upload])
    # Configure vectorizer for the named vectors
    vectorizer_config = [wvc.config.Configure.NamedVectors.none(name=x.name) for x in all_named_vectors]

    logger.info("Creating collection")  # Indicate that the collection is being created
    # Create a new collection with the specified properties and vectorizer config
    collection = client.collections.create(
        name="Classes",
        description="Text2kg benchmark classes",
        vectorizer_config=vectorizer_config,
        properties=[
            wvc.config.Property(name="TermIRI", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="RDF_type", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="Label", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="Description", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="Subclass", data_type=wvc.config.DataType.TEXT_ARRAY),
            wvc.config.Property(name="Superclass", data_type=wvc.config.DataType.TEXT_ARRAY),
            wvc.config.Property(name="Language", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="Ontology", data_type=wvc.config.DataType.TEXT),
        ],
    )

    # Upload the formatted object data
    logger.info("Uploading data")  # Indicate the start of data upload
    objects_to_upload = [wvc.data.DataObject(properties=d[0], vector=d[1], uuid=d[2]) for d in formatted_objects_for_upload]

    # Split objects into batches for uploading
    batches = split_list(objects_to_upload, 4)
    try:
        successes = 0
        for i, batch in enumerate(batches):
            logger.info("Uploading batch %d to %s", i + 1, collection.name)

            result = collection.data.insert_many(batch)  # Insert batch
            inserted_docs = batch  # The original batch contains the inserted documents
            successes += len(inserted_docs)  # Count successfully inserted documents

            logger.info("Successfully inserted %d documents in batch %d: %s", len(inserted_docs), i + 1, [Vectorless(x) for x in inserted_docs])

        logger.info("Total successfully inserted documents: %d", successes)

        
        return {"uploaded": successes}
    except Exception as e:
        logger.error("Error during insert_many: %s", traceback.format_exc(), exc_info=e)
        exception_happened = True
        
        return {"error": True}

def fill_class_copied_named_vectors():
    # Fetch all objects and named vectors to fill the copied named vectors
    all_objects = fetch_all_objects(collection="Classes")
    all_named_vectors = fetch_all_named_vectors(collection="Classes")
    
    # Get the mappings for the copied named vectors
    uuid_to_nv_mappings = get_copied_named_vectors(all_objects, all_named_vectors)
    
    # Fill the copied named vectors in the collection
    fill_copied_named_vectors(uuid_to_nv_mappings, "Classes")

# Individual Collection Functions

def get_individuals_collection_mappings():

    # Define methodologies for embedding based on different fields
    methodologies = {
        "Label": embed_using_label,
        "Description": embed_using_desc,
        "Domain": embed_using_domain,
        "Range": embed_using_range
    }
    return methodologies  # Return both models and methodologies

def format_individuals_query_results(endpoint_query_results, methodologies, models, all_named_vectors):
    formatted_objects = []  # List to hold formatted individual objects
    for i, result_doc in enumerate(endpoint_query_results):
        # Create a dictionary to store properties of the individual
        formatted_object = {
            "TermIRI": result_doc.termIRI,

            "Ontology": result_doc.ontology,
            "Label": result_doc.label,

            "Domain": result_doc.domain,
            "Range": result_doc.range,
            "Language": result_doc.language
        }
        
        # Infer the label from TermIRI if it's not provided
        if not result_doc.label:
            if "#" in result_doc.termIRI:
                formatted_object["Label"] = result_doc.termIRI.split("#")[1]
            elif "/" in result_doc.termIRI:
                formatted_object["Label"] = result_doc.termIRI.split("/")[1]

        embeddings = {}  # Dictionary to hold embeddings for the individual
        for vector in all_named_vectors:
            # Skip vectors with copy separator as they are handled elsewhere
            if "___CP_SEPARATOR___" not in vector.name:
                model = vector.vectorizer
                field_name = vector.field_name
                case_lang = vector.language
            
                # Generate embeddings only if the language matches
                if case_lang == result_doc.language:
                    embeddings[vector.name] = methodologies[field_name](result_doc, models[model])

        # Generate a UUID for the individual based on its TermIRI
        uuid = generate_uuid5(result_doc.termIRI)
        formatted_objects.append([formatted_object, embeddings, uuid])  # Append formatted data
    
    return formatted_objects  # Return the list of formatted objects

def create_individuals_collection():
    logger.info("Creating Individuals collection")  # Indicate the start of collection creation
    logger.info("Fetching SPARQL endpoint query results")
    # Fetch data from the endpoint for Individuals
    endpoint_query_results = fetch_data_from_endpoint(url_endpoint, type="Individuals")

    # Get the models and methodologies for embedding
    methodologies = get_individuals_collection_mappings()

    # Prepare combinations for various fields and relationships
    all_combos = []
    for field_name in ["Label", "Description"]:
        for lang in languages:
            for model in models:
                for copy_relationship, collection in [("Domain", "Classes"), ("Range", "Classes")]:
                    all_combos.append([field_name, lang, model, "default", copy_relationship, collection])

    # Create named vectors for all combinations
    all_named_vectors = []
    for item in all_combos:
        all_named_vectors.extend(create_named_vectors(item))  # Use extend to add all vectors to the list

    all_named_vectors = set(all_named_vectors)  # Remove duplicates by converting the list to a set
    print("Formatting results for upload")
    # Format objects for upload
    formatted_objects_for_upload = format_individuals_query_results(endpoint_query_results, methodologies, models, all_named_vectors)

    # Configure vectorizer for the named vectors
    vectorizer_config = [wvc.config.Configure.NamedVectors.none(name=x.name) for x in all_named_vectors]

    logger.info("Creating collection")  # Indicate that the collection is being created
    
    # Create a new collection with the specified properties and vectorizer config
    collection = client.collections.create(
        name="Individuals",
        description="Text2kg benchmark properties",
        vectorizer_config=vectorizer_config,
        properties=[
            wvc.config.Property(name="TermIRI", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="RDF_type", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="Label", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="Description", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="Domain", data_type=wvc.config.DataType.TEXT_ARRAY),
            wvc.config.Property(name="Range", data_type=wvc.config.DataType.TEXT_ARRAY),
            wvc.config.Property(name="Language", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="Ontology", data_type=wvc.config.DataType.TEXT),
        ],
    )

    # Upload the formatted object data
    logger.info("Uploading data")  # Indicate the start of data upload
    objects_to_upload = [wvc.data.DataObject(properties=d[0], vector=d[1], uuid=d[2]) for d in formatted_objects_for_upload]
 
    # Split objects into batches for uploading
    batches = split_list(objects_to_upload, 4)
    try:
        successes = 0
        for i, batch in enumerate(batches):
            logger.info("Uploading batch %d to %s", i + 1, collection.name)

            result = collection.data.insert_many(batch)  # Insert batch
            inserted_docs = batch  # The original batch contains the inserted documents
            successes += len(inserted_docs)  # Count successfully inserted documents

            logger.info("Successfully inserted %d documents in batch %d: %s", len(inserted_docs), i + 1, [Vectorless(x) for x in inserted_docs])

        logger.info("Total successfully inserted documents: %d", successes)

        
        return {"uploaded": successes}
    except Exception as e:
        logger.error("Error during insert_many: %s", traceback.format_exc(), exc_info=e)
        exception_happened = True
            
        return {"error": True}
    
def fill_individuals_copied_named_vectors():
    # Fetch all objects and named vectors to fill the copied named vectors
    all_objects = fetch_all_objects(collection="Individuals")
    all_named_vectors = fetch_all_named_vectors(collection="Individuals")
    
    # Get the mappings for the copied named vectors
    uuid_to_nv_mappings = get_copied_named_vectors(all_objects, all_named_vectors)
    
    # Fill the copied named vectors in the collection
    fill_copied_named_vectors(uuid_to_nv_mappings, "Individuals")

# Data Property Collection Functions
def get_data_property_collection_mappings():

    # Define methodologies for embedding based on different fields
    methodologies = {
        "Label": embed_using_label,
        "Description": embed_using_desc,
        "Domain": embed_using_domain,
        "Range": embed_using_range
    }
    return methodologies  # Return both models and methodologies

def format_data_property_query_results(endpoint_query_results, methodologies, models, all_named_vectors):
    formatted_objects = {}
    for i, result_doc in enumerate(endpoint_query_results):
        
        # Format each result document into a structured object
        formatted_object = {
            "TermIRI": result_doc.termIRI,

            "Ontology": result_doc.ontology,

            "Domain": result_doc.domain,
            "Range": result_doc.range,
        }
        
        
        # Add "label_BLANK" keys for each language in the languages list
        for lang in languages:
            label_key = f"label_{lang.lower()}"

            formatted_object[label_key] = getattr(result_doc, label_key, [])
            
            desc_key = f"description_{lang.lower()}"
            formatted_object[desc_key] = getattr(result_doc, desc_key, "")
        # Infer the label from TermIRI if it's not provided
        if has_no_labels(result_doc):
            
            if UNKNOWN_LABEL_REPLACEMENT_EMBEDDING_STRATEGY == "infer_on_iri":
                if "#" in result_doc.termIRI:
                    formatted_object["label_none"] = [result_doc.termIRI.split("#")[1]]
                elif "/" in result_doc.termIRI:
                    formatted_object["label_none"] = [result_doc.termIRI.split("/")[1]]


        embeddings = {}  # Dictionary to hold embeddings for the object
        for vector in all_named_vectors:
            # Skip vectors with copy separator as they are handled elsewhere
            if "___CP_SEPARATOR___" not in vector.name:
                model = vector.vectorizer
                field_name = vector.field_name
                case_lang = vector.language

               
                
                # Generate embeddings only if the language matches
                if field_name in ["Label", "Description"]:
                    doc_has_field_in_wanted_language = hasattr(result_doc, f"{field_name.lower()}_{case_lang.lower()}")
                    if doc_has_field_in_wanted_language:
                        
                        embeddings[vector.name] = methodologies[field_name](result_doc, models[model], case_lang)
                    
                else:
                    
                    embeddings[vector.name] = methodologies[field_name](result_doc, models[model])
          

        # Generate a UUID for the object
        uuid = generate_uuid5(result_doc.termIRI)
                
        if uuid in formatted_objects:
            old_object = formatted_objects[uuid]
            formatted_objects[uuid] = [old_object[0].update_info(result_doc), old_object[1] | embeddings, uuid]
        else:
            formatted_objects[uuid] = [formatted_object, embeddings, uuid]  # Append formatted data

    
    return list(formatted_objects.values())  # Return the list of formatted objects

def Vectorless(obj):
    # Get the object's properties and values as a dictionary
    properties = vars(obj)
    
    # Filter out the 'vector' property
    filtered_properties = {key: value for key, value in properties.items() if key != 'vector'}
    
    # Convert the filtered properties to a string
    return ", ".join(f"{key}: {value}" for key, value in filtered_properties.items())


def create_data_property_collection():
    logger.info("Creating DataProperties collection")  # Indicate the start of collection creation
    logger.info("Fetching SPARQL endpoint query results")
    # Fetch data from the endpoint for DataProperties
    endpoint_query_results = fetch_data_from_endpoint(url_endpoint, type="DataProperties")
    
    # Get the models and methodologies for embedding
    methodologies = get_data_property_collection_mappings()

    all_combos = []  # List to hold all combinations of fields, languages, and models
    for field_name in ["Label", "Description"]:
        for lang in languages:
            for model in models:
                # Create combinations for Domain and Range relationships
                for copy_relationship, collection in [("Domain", "Classes"), ("Range", "Classes")]:
                    all_combos.append([field_name, lang, model, "default", copy_relationship, collection])

    # Create named vectors for all combinations
    all_named_vectors = []
    for item in all_combos:
        all_named_vectors.extend(create_named_vectors(item))  # Use extend to add all vectors to the list

    all_named_vectors = set(all_named_vectors)  # Remove duplicates by converting the list to a set
    print("Formatting results for upload")
    # Format objects for upload
    formatted_objects_for_upload = format_data_property_query_results(endpoint_query_results, methodologies, models, all_named_vectors)

    # Configure vectorizer for the named vectors
    vectorizer_config = [wvc.config.Configure.NamedVectors.none(name=x.name) for x in all_named_vectors]

    logger.info("Creating collection")  # Indicate that the collection is being created
    # Create a new collection with the specified properties and vectorizer config
    collection = client.collections.create(
        name="DataProperties",
        description="Text2kg benchmark data properties",
        vectorizer_config=vectorizer_config,
        properties=[
            wvc.config.Property(name="TermIRI", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="RDF_type", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="Label", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="Description", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="Domain", data_type=wvc.config.DataType.TEXT_ARRAY),
            wvc.config.Property(name="Range", data_type=wvc.config.DataType.TEXT_ARRAY),
            wvc.config.Property(name="Language", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="Ontology", data_type=wvc.config.DataType.TEXT),
        ],
    )

    # Upload the formatted object data
    logger.info("Uploading data")  # Indicate the start of data upload
    objects_to_upload = [wvc.data.DataObject(properties=d[0], vector=d[1], uuid=d[2]) for d in formatted_objects_for_upload]

    # Split objects into batches for uploading
    batches = split_list(objects_to_upload, 4)
    try:
        successes = 0
        for i, batch in enumerate(batches):
            logger.info("Uploading batch %d to %s", i + 1, collection.name)

            result = collection.data.insert_many(batch)  # Insert batch
            inserted_docs = batch  # The original batch contains the inserted documents
            successes += len(inserted_docs)  # Count successfully inserted documents

            logger.info("Successfully inserted %d documents in batch %d: %s", len(inserted_docs), i + 1, [Vectorless(x) for x in inserted_docs])

        logger.info("Total successfully inserted documents: %d", successes)

        
        return {"uploaded": successes}
    except Exception as e:
        logger.error("Error during insert_many: %s", traceback.format_exc(), exc_info=e)
        exception_happened = True
        
        return {"error": True}
def fill_data_property_copied_named_vectors():
    # Fetch all objects and named vectors to fill the copied named vectors
    all_objects = fetch_all_objects(collection="DataProperties")

    all_named_vectors = fetch_all_named_vectors(collection="DataProperties")
    
    # Get the mappings for the copied named vectors
    uuid_to_nv_mappings = get_copied_named_vectors(all_objects, all_named_vectors)
    
    # Fill the copied named vectors in the collection
    fill_copied_named_vectors(uuid_to_nv_mappings, "DataProperties")

# RDFtype Collection Functions
def get_rdftype_collection_mappings():

    methodologies = {
        "Label": embed_using_label,
        "Description": embed_using_desc,
        "Superclass": embed_using_superclass
    }
    return methodologies

def format_rdftype_query_results(endpoint_query_results, methodologies, models, all_named_vectors):
    
    logger.info("Formatting objects")
    
    formatted_objects = []
    for i, result_doc in enumerate(endpoint_query_results):
        formatted_object = {
            "TermIRI": result_doc.termIRI,
            "Ontology": result_doc.ontology,

            "Superclass": result_doc.superclass,
            "Language": result_doc.language
        }
        if not result_doc.label:
            if "#" in result_doc.termIRI:
                formatted_object["Label"] = result_doc.termIRI.split("#")[1]
            elif "/" in result_doc.termIRI:
                formatted_object["Label"] = result_doc.termIRI.split("/")[1]

        embeddings = {}
        for vector in all_named_vectors:
            if "___CP_SEPARATOR___" not in vector.name:
                model = vector.vectorizer
                field_name = vector.field_name
                case_lang = vector.language

                if case_lang == result_doc.language:
                    embeddings[vector.name] = methodologies[field_name](result_doc, models[model])

        uuid = generate_uuid5(result_doc.termIRI)
        formatted_objects.append([formatted_object, embeddings, uuid])

    return formatted_objects

def create_rdftype_collection():
    
    logger.info("Creating RDF_types collection")
    logger.info("Fetching SPARQL endpoint query results")
    endpoint_query_results =  fetch_data_from_endpoint(url_endpoint, type="RDFtypes")

    methodologies = get_rdftype_collection_mappings()

    all_combos = []
    for field_name in ["Label", "Description"]:
        for lang in languages:
            for model in models:
                for copy_relationship, collection in [("Superclass", "RDFtypes")]:

                        all_combos.append([field_name, lang, model, "default", copy_relationship, collection])

    all_named_vectors = []
    for item in all_combos:
        all_named_vectors.extend(create_named_vectors(item))  # Use extend to add all vectors to the list

    all_named_vectors = set(all_named_vectors)  # Convert the list to a set to remove duplicates
    print("Formatting results for upload")
    formatted_objects_for_upload = format_rdftype_query_results(endpoint_query_results, methodologies, models, all_named_vectors)
    
    # Configure vectorizer
    vectorizer_config = [wvc.config.Configure.NamedVectors.none(name=x.name) for x in all_named_vectors]

    logger.info("Creating collection")
    collection = client.collections.create(
            
            name="RDFtypes",
            
            description="Text2kg benchmark RDF_types",
            
            vectorizer_config=vectorizer_config,

            properties = [
                wvc.config.Property(name="TermIRI", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="Label", data_type=wvc.config.DataType.TEXT), 
                wvc.config.Property(name="Description", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="Superclass", data_type=wvc.config.DataType.TEXT_ARRAY),
                wvc.config.Property(name="Language", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="Ontology", data_type=wvc.config.DataType.TEXT),
            ],
        )

    if formatted_objects_for_upload:
        # Upload the formatted object data
        logger.info("Uploading data")
        objects_to_upload = [wvc.data.DataObject(properties=d[0], vector=d[1], uuid=d[2]) for d in formatted_objects_for_upload]

        batches = split_list(objects_to_upload, 4)
        try:
            successes = 0
            for i, batch in enumerate(batches):
                logger.info("Uploading batch %d to %s", i + 1, collection.name)

                result = collection.data.insert_many(batch)  # Insert batch
                inserted_docs = batch  # The original batch contains the inserted documents
                successes += len(inserted_docs)  # Count successfully inserted documents

                logger.info("Successfully inserted %d documents in batch %d: %s", len(inserted_docs), i + 1, [Vectorless(x) for x in inserted_docs])

            logger.info("Total successfully inserted documents: %d", successes)

            
            return {"uploaded": successes}
        except Exception as e:
            logger.error("Error during insert_many: %s", traceback.format_exc(), exc_info=e)
            exception_happened = True
            
            return {"error": True}
    return {"uploaded": 0}

def fill_rdftype_copied_named_vectors():
    all_objects = fetch_all_objects(collection="RDFtypes")

    all_named_vectors = fetch_all_named_vectors(collection="RDFtypes")

    uuid_to_nv_mappings = get_copied_named_vectors(all_objects, all_named_vectors)
    
    fill_copied_named_vectors(uuid_to_nv_mappings, "RDFtypes")

# Ontology Collection Functions
def get_ontology_collection_mappings():
    pass

def format_ontology_query_results(endpoint_query_results, methodologies, models, all_named_vectors):
    pass

def create_ontology_collection():
    pass

def fill_ontology_copied_named_vectors():
    pass

def ontology_collection_creation():
    # Get ontology property data from endpoint
    logger.info("Loading data")
    endpoint_query_results = fetch_data_from_endpoint(url_endpoint, type="Ontologies")

    # Mappings between mapping methodology names and functions
    methodologies = {"OntologySpecial": embed_ontology_classes, "OntologyDataproperties": embed_ontology_dataproperties}  # Add your embedding methodologies here


    # All combinations between models, methodologies and languages
    all_combos = []
    for model in models:
        for method in methodologies:
            for lang in languages:
                all_combos.append({"case_name": f"{model}___{method}___{lang}", "model": model, "method": method, "lang": lang})


    formatted_objects_for_upload = []

    # Formatting of the ontology data to upload to the collection
    logger.info("Generating embeddings and formatting data")
    for i, result_doc in enumerate(endpoint_query_results):
        tp = max(len(endpoint_query_results) / 10, 1)
        if int(tp) > 1:
            if i % int(tp) == 0:
                logger.info("%d / %d", i, len(endpoint_query_results))

        formatted_object = {}

        uuid = generate_uuid5(result_doc.ontologyIRI)

        # Map the ontology attributes to the respective collection properties
        formatted_object["OntologyIRI"] = result_doc.ontologyIRI
        
        formatted_object["Classes"] = result_doc.classes
        formatted_object["Dataproperties"] = result_doc.dataproperties
        formatted_object["Language"] = result_doc.language

        embeddings = {}
        for case in all_combos:
            name = case["case_name"]
            model = case["model"]
            method = case["method"]
            case_lang = case["lang"]
            if case_lang == result_doc.language:

                if not name in embeddings:
                    embeddings[name] = []

                embeddings[name] = methodologies[method](result_doc, models[model])

        formatted_objects_for_upload.append([formatted_object, embeddings, uuid])

    # Configurations for custom vectorizers (one for every case)
    vectorizer_config = [wvc.config.Configure.NamedVectors.none(name=x["case_name"]) for x in all_combos]

    # Create a new collection with the vectorizer configs
    logger.info("Creating collection")
    collection = client.collections.create(
        name="Ontologies",
        description="Text2kg benchmark Ontologies",
        vectorizer_config=vectorizer_config,
        properties=[
            wvc.config.Property(name="OntologyIRI", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="Classes", data_type=wvc.config.DataType.TEXT_ARRAY),
            wvc.config.Property(name="Dataproperties", data_type=wvc.config.DataType.TEXT_ARRAY),
            wvc.config.Property(name="Language", data_type=wvc.config.DataType.TEXT),
        ],
    )


    # Upload the formatted_object data
    logger.info("Uploading data")
    objects_to_upload = []
    for d in formatted_objects_for_upload:
        objects_to_upload.append(wvc.data.DataObject(
            properties=d[0],
            vector=d[1],
            uuid=d[2]
        ))

    batches = split_list(objects_to_upload, 4)

    try:
        successes = 0
        for i, batch in enumerate(batches):
            logger.info("Uploading batch %d to %s", i + 1, collection.name)

            result = collection.data.insert_many(batch)  # Insert batch
            inserted_docs = batch  # The original batch contains the inserted documents
            successes += len(inserted_docs)  # Count successfully inserted documents

            logger.info("Successfully inserted %d documents in batch %d: %s", len(inserted_docs), i + 1, [Vectorless(x) for x in inserted_docs])

        logger.info("Total successfully inserted documents: %d", successes)

    except Exception as e:
        logger.error("Error during insert_many: %s", traceback.format_exc(), exc_info=e)
        exception_happened = True
        
        return {"error": True}

def get_properties_from_collection(client, collection_name):
    collection = client.collections.get(collection_name)
    
    all_properties = set()
    for item in collection.iterator():
        
        for k in item.properties:
            all_properties.add(k[0].upper()+k[1:])
    
    return all_properties

# OBJECT PROPERTIES: {DOMAIN: CLASSES, RANGE: CLASSES}
# INDIVIDUALS: {DOMAIN: CLASSES?, RANGE: CLASSES?}
# CLASSES: {SUBCLASS: CLASSES, SUPERCLASS: CLASSES}
# RDF_TYPES: {SUPERCLASS: RDF_TYPES}
# ONTOLOGIES: {}
# RDF_TYPES: {}
# Define the get_collection_status function
def get_collection_status(collection_name, stats):
    if "uploaded" in stats:
        return [collection_name, "Successfully created", stats["uploaded"]]
    elif "error" in stats and stats["error"]:
        return [collection_name, "Error", "N/A"]
    else:
        return [collection_name, "Unknown", "N/A"]

# Main program
if __name__ == "__main__":
    exception_happened = False
    
    with get_weaviate_client() as client:
        
        collections_at_beginning = [collection_name for collection_name in client.collections.list_all(simple=True)]

        # Define the CSV file name
        csv_file_name = "collection_creation_status.csv"
        
        with open(csv_file_name, mode='w', newline='', encoding='utf-8') as file:
            
            writer = csv.writer(file)
            writer.writerow(["START OF COLLECTION CREATION SCRIPT"])
            # Write the header
            writer.writerow(["Exsiting collections"])
            # Write the collection statuses
            writer.writerow(collections_at_beginning)
        
        if create_new:
            client.collections.delete_all()
            collections_after_delete = [collection_name for collection_name in client.collections.list_all(simple=True)]
            
            with open(csv_file_name, mode='a', newline='', encoding='utf-8') as file:
                
                writer = csv.writer(file)
                writer.writerow([])
                writer.writerow(["AFTER DELETION ISSUED"])
                # Write the header
                writer.writerow(["Exsiting collections"])
                # Write the collection statuses
                writer.writerow(collections_after_delete)

            try:
                
                object_property_stats = "N/A"
                data_property_stats = "N/A"
                class_stats = "N/A"
                rdftype_stats = "N/A"
                individuals_stats = "N/A"
                ontology_stats = "N/A"
                
                # # Create stats for each collection
                object_property_stats = create_object_property_collection()
                data_property_stats = create_data_property_collection()
                class_stats = create_class_collection()
                # rdftype_stats = create_rdftype_collection()
                #individuals_stats = create_individuals_collection()

                # # Fill in copied named vectors
                fill_object_property_copied_named_vectors()
                fill_data_property_copied_named_vectors()
                fill_class_copied_named_vectors()
                # fill_rdftype_copied_named_vectors()
                # fill_individuals_copied_named_vectors()

                # Collect collection statuses
                status_data = [
                    get_collection_status("Object Properties", object_property_stats),
                    get_collection_status("Data Properties", data_property_stats),
                    get_collection_status("Classes", class_stats),
                    get_collection_status("RDFTypes", rdftype_stats),
                    get_collection_status("Individuals", individuals_stats)
                ]

                # Write the data to the CSV file
                with open(csv_file_name, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    # Write the header
                    writer.writerow([])
                    writer.writerow(["END OF COLLECTION CREATION SCRIPT"])
                    writer.writerow(["Collection Name", "Status", "Items Processed"])
                    # Write the collection statuses
                    writer.writerows(status_data)

                print(f"Collection statuses saved to {csv_file_name}")

            except Exception as e:
                logger.error(e, traceback.format_exc())
                exception_happened = True

            finally:
                if exception_happened:
                    # TODO: Error message to look at logs
                    sys.exit(1)

                sys.exit(0)