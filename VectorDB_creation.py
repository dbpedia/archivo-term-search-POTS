import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import Filter
from weaviate.util import generate_uuid5
from langchain_community.embeddings import SentenceTransformerEmbeddings
from VectorDB_creation_aux import *
from dotenv import load_dotenv
import os
import traceback
from dataclasses import dataclass, field
from typing import List
load_dotenv()

# Global variables
wcd_url = os.getenv("WCD_URL")
wcd_api_key = os.getenv("WCD_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
hf_key = os.getenv("HF_KEY")
url_endpoint =  os.getenv("SPARQL_ENDPOINT")
local_weaviate_port = int(os.getenv("WEAVIATE_PORT"))
local_weaviate_port_grpc = int(os.getenv("WEAVIATE_PORT_GRPC"))
create_new = os.getenv("DELETE_OLD_INDEX")

# Available models
model_names = ["LaBSE","all-MiniLM-L6-v2","all-MiniLM-L12-v2","all-distilroberta-v1","paraphrase-multilingual-MiniLM-L12-v2","multi-qa-mpnet-base-cos-v1"]
models = {x.replace("-", "_"): SentenceTransformerEmbeddings(model_name=x) for x in model_names}

languages = ["en", "fr", "None"]


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
    target_property: str = field(default_factory=str)

    def __post_init__(self):
        object.__setattr__(self, 'name', f"{self.vectorizer}___{self.field_name}___{self.language}")

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, VirtualNamedVectorEmbedding):
            return self.name == other.name
        return False

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
        target_property=""
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
            target_property=""
        )
        # Name the copied embedding to reflect its relationship
        copy_embedding.name = f"{original_embedding.name}___CP_SEPARATOR___{copy_relationship}___{copy_relationship_index}___{n}"
        embeddings.append(copy_embedding)

    return embeddings  # Return the list of embeddings

def get_copied_named_vectors(all_objects, all_named_vectors):
    # Initialize dictionaries to hold embeddings and empty embeddings
    embeddings = {}
    empty_embeddings = {}
    
    for formatted_object in all_objects:
        print("\n---------------------------------------------------")
        print("New Object")
        print("Object IRI:", formatted_object.properties["termIRI"])
        print("Object label:", formatted_object.properties["label"])

        # Ensure a dictionary exists for each object's UUID
        if not formatted_object.uuid in embeddings:
            embeddings[formatted_object.uuid] = {}
            
        for vector in all_named_vectors:
            print("")
            print("Search in vector", vector)
            if "___CP_SEPARATOR___" in vector:
                # Split the vector name to extract original and copy info
                original_vector_info, copy_vector_info = vector.split("___CP_SEPARATOR___")
                property_to_find, target_collection, index = copy_vector_info.split("___")
                vectorizer = vector.split("___")[0]
                prop = vector.split("___")[1]
                
                # Initialize empty embeddings if they haven't been created yet
                if not vectorizer in empty_embeddings:
                    empty_embeddings[vectorizer] = generate_empty_embedding(models[vectorizer])
                
                property_to_find = property_to_find.lower()
                
                # Check if the property exists in the object's properties
                if len(formatted_object.properties[property_to_find]) >= int(index):
                    print("Looking to find", property_to_find)
                    target_uri = formatted_object.properties[property_to_find][int(index)-1]

                    # Query the collection for the named vector embedding
                    result = query_collection_for_NV_embedding(target_collection, target_uri, original_vector_info)
                    
                    if result:
                        # Store the result in the embeddings dictionary
                        embeddings[formatted_object.uuid][vector] = result
                        print("Set", formatted_object.properties["termIRI"], "to", target_uri,"'s", prop) 
                    else:
                        # If no result found, use an empty embedding
                        embeddings[formatted_object.uuid][vector] = empty_embeddings[vectorizer]
                else:
                    # If the index is out of bounds, use an empty embedding
                    embeddings[formatted_object.uuid][vector] = empty_embeddings[vectorizer]
                        
    return embeddings  # Return the mappings of UUIDs to named vector embeddings

def fill_copied_named_vectors(uuid_to_nv_mappings, target_collection):
    print("Filling copied named vectors")
    collection = client.collections.get(name=target_collection)

    # Calculate the progress interval for logging
    tp = len(uuid_to_nv_mappings) / 10
    for i, uuid in enumerate(uuid_to_nv_mappings):
        if i % int(tp) == 0:
            print(i, "/", len(uuid_to_nv_mappings))
        if uuid_to_nv_mappings[uuid]:
            # Update the collection with the vector mappings
            collection.data.update(uuid, vector=uuid_to_nv_mappings[uuid])

def query_collection_for_NV_embedding(target_collection, target_uri, target_named_vector):
    # Fetch the specified collection
    collection = client.collections.get(name=target_collection)
    print("Looking for", target_uri)

    # Attempt to fetch the object by its UUID
    data_object = collection.query.fetch_object_by_id(
        uuid=generate_uuid5(target_uri),  # Generate the UUID for the target URI
        include_vector=True  # Include vector data in the response
    )
    
    if data_object:
        # Return the specific named vector if it exists
        return data_object.vector[target_named_vector]
    return None  # Return None if the object is not found

def fetch_all_objects(collection):
    # Fetch all objects from the specified collection
    return [x for x in client.collections.get(collection).iterator()]

def fetch_all_named_vectors(collection):
    # Fetch all named vectors from the specified collection
    return list(client.collections.get(collection).config.get().vector_config.keys())

# Object Property Collection Functions
def get_object_property_collection_mappings():
    # Initialize models for different vectorization strategies
    models = {x.replace("-", "_"): SentenceTransformerEmbeddings(model_name=x) for x in model_names}
    methodologies = {
        "Label": embed_using_label,
        "Description": embed_using_desc,
        "Domain": embed_using_domain,
        "Range": embed_using_range
    }
    return models, methodologies  # Return both models and methodologies

def format_object_property_query_results(endpoint_query_results, methodologies, models, all_named_vectors):
    formatted_objects = []
    for i, result_doc in enumerate(endpoint_query_results):
        # Format each result document into a structured object
        formatted_object = {
            "TermIRI": result_doc.termIRI,
            "RDF_type": result_doc.rdfType,
            "Ontology": result_doc.ontology,
            "Label": result_doc.label,
            "Description": result_doc.description,
            "Domain": result_doc.domain,
            "Range": result_doc.range,
            "Language": result_doc.language
        }
        
        # Assign a label based on the TermIRI if no label is present
        if not result_doc.label:
            if "#" in result_doc.termIRI:
                formatted_object["Label"] = result_doc.termIRI.split("#")[1]
            elif "/" in result_doc.termIRI:
                formatted_object["Label"] = result_doc.termIRI.split("/")[1]

        # Initialize a dictionary to hold embeddings for the object
        embeddings = {}
        for vector in all_named_vectors:
            if "___CP_SEPARATOR___" not in vector.name:
                model = vector.vectorizer
                field_name = vector.field_name
                case_lang = vector.language
            
                # Only embed if the language matches
                if case_lang == result_doc.language:
                    embeddings[vector.name] = methodologies[field_name](result_doc, models[model])

        # Generate a UUID for the object
        uuid = generate_uuid5(result_doc.termIRI)
        formatted_objects.append([formatted_object, embeddings, uuid])
    
    return formatted_objects  # Return the list of formatted objects

def create_object_property_collection():
    print("Creating ObjectProperties collection")
    # Fetch data from the endpoint for ObjectProperties
    endpoint_query_results =  fetch_data_from_endpoint(url_endpoint, type="ObjectProperties")

    # Get the models and methodologies for embedding
    models, methodologies = get_object_property_collection_mappings()

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
    print([x.name for x in all_named_vectors])
    
    # Format objects for upload
    formatted_objects_for_upload = format_object_property_query_results(endpoint_query_results, methodologies, models, all_named_vectors)

    # Configure vectorizer for the named vectors
    vectorizer_config = [wvc.config.Configure.NamedVectors.none(name=x.name) for x in all_named_vectors]

    # Delete the existing collection if it exists
    client.collections.delete("ObjectProperties")

    print("Creating collection")
    # Create a new collection with the specified properties and vectorizer config
    collection = client.collections.create(
        name="ObjectProperties",
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
    print("Uploading data")
    objects_to_upload = [wvc.data.DataObject(properties=d[0], vector=d[1], uuid=d[2]) for d in formatted_objects_for_upload]

    # Split objects into batches for uploading
    batches = split_list(objects_to_upload, 4)
    try:
        for i, batch in enumerate(batches):
            print(f"Uploading batch {i + 1}")
            collection.data.insert_many(batch)
    except Exception as e:
        print(f"Error during insert_many: {e}")

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
    # Create mappings for model names to their corresponding SentenceTransformerEmbeddings instances
    models = {x.replace("-", "_"): SentenceTransformerEmbeddings(model_name=x) for x in model_names}
    
    # Define methodologies for embedding based on different fields
    methodologies = {
        "Label": embed_using_label,
        "Description": embed_using_desc,
        "Subclass": embed_using_subclass, 
        "Superclass": embed_using_superclass
    }
    return models, methodologies  # Return both models and methodologies

def format_class_query_results(endpoint_query_results, methodologies, models, all_named_vectors):
    print("Formatting objects")  # Indicate the start of formatting objects
    formatted_objects = []  # List to hold the formatted objects
    for i, result_doc in enumerate(endpoint_query_results):
        # Create a dictionary to store properties of the class
        formatted_object = {
            "TermIRI": result_doc.termIRI,
            "RDF_type": result_doc.rdfType,
            "Ontology": result_doc.ontology,
            "Label": result_doc.label,
            "Description": result_doc.description,
            "Subclass": result_doc.subclass,
            "Superclass": result_doc.superclass,
            "Language": result_doc.language
        }
        
        # Infer the label from TermIRI if it's not provided
        if not result_doc.label:
            if "#" in result_doc.termIRI:
                formatted_object["Label"] = result_doc.termIRI.split("#")[1]
            elif "/" in result_doc.termIRI:
                formatted_object["Label"] = result_doc.termIRI.split("/")[1]

        embeddings = {}  # Dictionary to hold embeddings for the object
        for vector in all_named_vectors:
            # Skip vectors with copy separator as they are handled elsewhere
            if "___CP_SEPARATOR___" not in vector.name:
                model = vector.vectorizer
                field_name = vector.field_name
                case_lang = vector.language

                # Generate embeddings only if the language matches
                if case_lang == result_doc.language:
                    embeddings[vector.name] = methodologies[field_name](result_doc, models[model])

        # Generate a UUID for the object based on its TermIRI
        uuid = generate_uuid5(result_doc.termIRI)
        formatted_objects.append([formatted_object, embeddings, uuid])  # Append formatted data
    
    return formatted_objects  # Return the list of formatted objects

def create_class_collection():
    print("Creating class collection")  # Indicate the start of collection creation
    # Fetch data from the endpoint for Classes
    endpoint_query_results = fetch_data_from_endpoint(url_endpoint, type="Classes")

    # Get the models and methodologies for embedding
    models, methodologies = get_class_collection_mappings()

    # Prepare combinations for various fields and relationships
    all_combos = []
    for field_name in ["Label", "Description"]:
        for lang in languages:
            for model in models:
                for copy_relationship, collection in [("Subclass", "Classes"), ("Superclass", "Classes")]:
                    all_combos.append([field_name, lang, model, "default", copy_relationship, collection])

    # Create named vectors for all combinations
    all_named_vectors = []
    for item in all_combos:
        all_named_vectors.extend(create_named_vectors(item))  # Use extend to add all vectors to the list

    all_named_vectors = set(all_named_vectors)  # Remove duplicates by converting the list to a set

    # Format objects for upload
    formatted_objects_for_upload = format_class_query_results(endpoint_query_results, methodologies, models, all_named_vectors)

    # Configure vectorizer for the named vectors
    vectorizer_config = [wvc.config.Configure.NamedVectors.none(name=x.name) for x in all_named_vectors]

    # Delete the existing collection if it exists
    client.collections.delete("Classes")

    print("Creating collection")  # Indicate that the collection is being created
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
    print("Uploading data")  # Indicate the start of data upload
    objects_to_upload = [wvc.data.DataObject(properties=d[0], vector=d[1], uuid=d[2]) for d in formatted_objects_for_upload]

    # Split objects into batches for uploading
    batches = split_list(objects_to_upload, 4)
    try:
        for i, batch in enumerate(batches):
            print(f"Uploading batch {i + 1}")  # Indicate which batch is being uploaded
            collection.data.insert_many(batch)  # Insert the batch into the collection
    except Exception as e:
        print(f"Error during insert_many: {e}")  # Catch and print any errors that occur during upload

def fill_class_copied_named_vectors():
    # Fetch all objects and named vectors to fill the copied named vectors
    all_objects = fetch_all_objects(collection="Classes")
    all_named_vectors = fetch_all_named_vectors(collection="Classes")
    
    # Get the mappings for the copied named vectors
    uuid_to_nv_mappings = get_copied_named_vectors(all_objects, all_named_vectors)
    
    # Fill the copied named vectors in the collection
    fill_copied_named_vectors(uuid_to_nv_mappings, "Classes")

# OLD (TODO: rewrite soon)
def class_collection_creation_hf_integration():
    # Get ontology class data from endpoint
    print("Loading data")  # Indicate the start of data loading
    endpoint_query_results = fetch_data_from_endpoint(url_endpoint, type="classes")

    # Create mappings for model names to their corresponding SentenceTransformerEmbeddings instances
    models = {x.replace("-", "_"): SentenceTransformerEmbeddings(model_name=x) for x in model_names}

    # Define methodologies for embedding based on different fields
    methodologies = {
        "Label": embed_using_label,
        "Description": embed_using_desc,
        "Subclass": embed_using_subclass,
        "Superclass": embed_using_superclass
    }
    
    # Define a dataclass to represent named vector embeddings
    @dataclass
    class VirtualNamedVectorEmbedding: 
        field_name: str = field(default_factory=str)  # Field name (e.g., "Label")
        language: str = field(default_factory=str)  # Language of the embedding
        vectorizer: str = field(default_factory=str)  # Name of the model used for embedding
        embed_strategy: str = field(default_factory=str)  # Strategy for embedding (e.g., "default")
        
        # CopiedVectors
        copy_relationship: str = field(default_factory=str)  # Relationship type (e.g., "Subclass")
        copy_relationship_index: str = field(default_factory=str)  # Index of the relationship
        
        # Construct the name for the embedding
        name = f"{vectorizer}___{field_name}___{language}___CP_SEPARATOR___{copy_relationship}___{copy_relationship_index}"
    
    all_named_vectors = []  # List to hold all named vector embeddings
    # Create embeddings for different fields, languages, and models
    for field_name in ["Label", "Description", "Subclass", "Superclass"]:
        for lang in languages:
            for model in models:
                embed_strategy = methodologies[field_name]
                for copy_relationship, collection in [("Domain", "Classes_hf")]:
                    all_named_vectors.append(VirtualNamedVectorEmbedding(field_name, lang, model, embed_strategy, copy_relationship, collection))

    formatted_objects_for_upload = []  # List to hold objects formatted for upload

    # Formatting of the ontology data to upload to the collection
    print("Generating embeddings and formatting data")
    for i, result_doc in enumerate(endpoint_query_results):
        tp = len(endpoint_query_results) / 10  # Progress tracker
        
        # Log progress every 10%
        if i % int(tp) == 0:
            print(i, "/", len(endpoint_query_results))
            
        formatted_object = {}  # Dictionary to hold formatted object properties

        # Generate a UUID for the object based on its TermIRI
        uuid = generate_uuid5(result_doc.termIRI)
        formatted_object["TermIRI"] = result_doc.termIRI
        formatted_object["RDF_type"] = result_doc.rdfType
        formatted_object["Ontology"] = result_doc.ontology
        formatted_object["Label"] = result_doc.label
        formatted_object["Description"] = result_doc.description
        formatted_object["Subclass"] = result_doc.subclass
        formatted_object["Superclass"] = result_doc.superclass
        formatted_object["Language"] = result_doc.language

        # Infer the label from TermIRI if it's not provided
        if not result_doc.label:
            if "#" in result_doc.termIRI:
                formatted_object["Label"] = result_doc.termIRI.split("#")[1]
            elif "/" in result_doc.termIRI:
                formatted_object["Label"] = result_doc.termIRI.split("/")[1]
                
        formatted_objects_for_upload.append([formatted_object, uuid])  # Append formatted data

    # Configure vectorizer for the named vectors
    vectorizer_config = [wvc.config.Configure.NamedVectors.text2vec_huggingface(name=x.name, model=x.vectorizer, source_properties=x.field_name) for x in all_named_vectors] 
    
    # Delete the existing collection for testing purposes
    client.collections.delete("Classes_hf")

    # Create a new collection with the vectorizer configs
    print("Creating collection")
    collection = client.collections.create(
        name="Classes_hf",
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
    print("Uploading data")
    objects_to_upload = []
    for d in formatted_objects_for_upload:
        objects_to_upload.append(wvc.data.DataObject(
            properties=d[0],
            uuid=d[1]
        ))

    # Split objects into batches for uploading
    batches = split_list(objects_to_upload, 4)

    try:
        for i, batch in enumerate(batches):
            print(f"Uploading batch {i+1}")  # Indicate which batch is being uploaded
            collection.data.insert_many(batch)  # Insert the batch into the collection

    except Exception as e:
        print(f"Error during insert_many: {e}")  # Catch and print any errors during upload

# Individual Collection Functions

def get_individuals_collection_mappings():
    # Create mappings for model names to their corresponding SentenceTransformerEmbeddings instances
    models = {x.replace("-", "_"): SentenceTransformerEmbeddings(model_name=x) for x in model_names}
    
    # Define methodologies for embedding based on different fields
    methodologies = {
        "Label": embed_using_label,
        "Description": embed_using_desc,
        "Domain": embed_using_domain,
        "Range": embed_using_range
    }
    return models, methodologies  # Return both models and methodologies

def format_individuals_query_results(endpoint_query_results, methodologies, models, all_named_vectors):
    formatted_objects = []  # List to hold formatted individual objects
    for i, result_doc in enumerate(endpoint_query_results):
        # Create a dictionary to store properties of the individual
        formatted_object = {
            "TermIRI": result_doc.termIRI,
            "RDF_type": result_doc.rdfType,
            "Ontology": result_doc.ontology,
            "Label": result_doc.label,
            "Description": result_doc.description,
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
    print("Creating Individuals collection")  # Indicate the start of collection creation
    # Fetch data from the endpoint for Individuals
    endpoint_query_results = fetch_data_from_endpoint(url_endpoint, type="Individuals")

    # Get the models and methodologies for embedding
    models, methodologies = get_individuals_collection_mappings()

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

    # Format objects for upload
    formatted_objects_for_upload = format_individuals_query_results(endpoint_query_results, methodologies, models, all_named_vectors)

    # Configure vectorizer for the named vectors
    vectorizer_config = [wvc.config.Configure.NamedVectors.none(name=x.name) for x in all_named_vectors]

    # Delete the existing collection if it exists
    client.collections.delete("Individuals")

    print("Creating collection")  # Indicate that the collection is being created
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
    print("Uploading data")  # Indicate the start of data upload
    objects_to_upload = [wvc.data.DataObject(properties=d[0], vector=d[1], uuid=d[2]) for d in formatted_objects_for_upload]

    # Split objects into batches for uploading
    batches = split_list(objects_to_upload, 4)
    try:
        for i, batch in enumerate(batches):
            print(f"Uploading batch {i + 1}")  # Indicate which batch is being uploaded
            collection.data.insert_many(batch)  # Insert the batch into the collection
    except Exception as e:
        print(f"Error during insert_many: {e}")  # Catch and print any errors that occur during upload

def fill_individuals_copied_named_vectors():
    # Fetch all objects and named vectors to fill the copied named vectors
    all_objects = fetch_all_objects(collection="Individuals")
    all_named_vectors = fetch_all_named_vectors(collection="Individuals")
    
    # Get the mappings for the copied named vectors
    uuid_to_nv_mappings = get_copied_named_vectors(all_objects, all_named_vectors)
    
    # Fill the copied named vectors in the collection
    fill_copied_named_vectors(uuid_to_nv_mappings, "Individuals")

def individual_collection_creation():
    # Get ontology property data from endpoint
    print("Loading data")  # Indicate the start of data loading
    endpoint_query_results = fetch_data_from_endpoint(url_endpoint, type="individuals")

    # Mappings between mapping methodology names and functions
    methodologies = {
        "Label": embed_using_label,
        "Description": embed_using_desc,
        "Domain": embed_using_domain,
        "Range": embed_using_range
    }

    # All combinations between models, methodologies and languages
    all_combos = []
    for model in models:
        for method in methodologies:
            for lang in languages:
                all_combos.append({
                    "case_name": f"{model}___{method}___{lang}",
                    "model": model,
                    "method": method,
                    "lang": lang
                })

    formatted_objects_for_upload = []  # List to hold formatted objects for upload

    # Formatting of the ontology data to upload to the collection
    print("Generating embeddings and formatting data")  # Indicate the start of data formatting
    for i, result_doc in enumerate(endpoint_query_results):
        tp = len(endpoint_query_results) / 10  # Progress tracker
        
        # Log progress every 10%
        if i % int(tp) == 0:
            print(i, "/", len(endpoint_query_results))
            
        formatted_object = {}  # Dictionary to hold formatted object properties
        
        # Generate a UUID for the object based on its TermIRI
        uuid = generate_uuid5(result_doc.termIRI)
        formatted_object["TermIRI"] = result_doc.termIRI
        formatted_object["RDF_type"] = result_doc.rdfType
        formatted_object["Ontology"] = result_doc.ontology
        formatted_object["Label"] = result_doc.label
        formatted_object["Description"] = result_doc.description
        formatted_object["Domain"] = result_doc.domain
        formatted_object["Range"] = result_doc.range
        formatted_object["Language"] = result_doc.language
        
        # Infer the label from TermIRI if it's not provided
        if not result_doc.label:
            if "#" in result_doc.termIRI:
                formatted_object["Label"] = result_doc.termIRI.split("#")[1]
            elif "/" in result_doc.termIRI:
                formatted_object["Label"] = result_doc.termIRI.split("/")[1]
        
        embeddings = {}  # Dictionary to hold embeddings for the individual
        for case in all_combos:
            name = case["case_name"]
            model = case["model"]
            method = case["method"]
            case_lang = case["lang"]
            # Generate embeddings only if the language matches
            if case_lang == result_doc.language:
                if not name in embeddings:
                    embeddings[name] = []
                
                embeddings[name] = methodologies[method](result_doc, models[model])  # Generate embeddings
            
        formatted_objects_for_upload.append([formatted_object, embeddings, uuid])  # Append formatted data

    # Configure vectorizer for the named vectors
    vectorizer_config = [wvc.config.Configure.NamedVectors.none(name=x["case_name"]) for x in all_combos]

    # Delete the existing collection if it exists
    client.collections.delete("Individuals")

    # Create a new collection with the vectorizer configs
    print("Creating collection")  # Indicate that the collection is being created
    collection = client.collections.create(
        name="Individuals",
        description="Text2kg benchmark individuals",
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
    print("Uploading data")  # Indicate the start of data upload
    objects_to_upload = []
    for d in formatted_objects_for_upload:
        objects_to_upload.append(wvc.data.DataObject(
            properties=d[0],
            vector=d[1],
            uuid=d[2]
        ))

    # Split objects into batches for uploading
    batches = split_list(objects_to_upload, 4)

    try:
        for i, batch in enumerate(batches):
            print(f"Uploading batch {i + 1}")  # Indicate which batch is being uploaded
            collection.data.insert_many(batch)  # Insert the batch into the collection
    except Exception as e:
        print(f"Error during insert_many: {e}")  # Catch and print any errors that occur during upload


# Data Property Collection Functions
def get_data_property_collection_mappings():
    # Create mappings of model names to their corresponding SentenceTransformerEmbeddings instances
    models = {x.replace("-", "_"): SentenceTransformerEmbeddings(model_name=x) for x in model_names}
    
    # Define methodologies for embedding based on different fields
    methodologies = {
        "Label": embed_using_label,
        "Description": embed_using_desc,
        "Domain": embed_using_domain,
        "Range": embed_using_range
    }
    return models, methodologies  # Return both models and methodologies

def format_data_property_query_results(endpoint_query_results, methodologies, models, all_named_vectors):
    formatted_objects = []  # List to hold formatted data property objects
    for i, result_doc in enumerate(endpoint_query_results):
        # Create a dictionary to store properties of the data property
        formatted_object = {
            "TermIRI": result_doc.termIRI,
            "RDF_type": result_doc.rdfType,
            "Ontology": result_doc.ontology,
            "Label": result_doc.label,
            "Description": result_doc.description,
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

        embeddings = {}  # Dictionary to hold embeddings for the data property
        for vector in all_named_vectors:
            # Skip vectors with copy separator as they are handled elsewhere
            if "___CP_SEPARATOR___" not in vector.name:
                model = vector.vectorizer
                field_name = vector.field_name
                case_lang = vector.language
            
                # Generate embeddings only if the language matches
                if case_lang == result_doc.language:
                    embeddings[vector.name] = methodologies[field_name](result_doc, models[model])

        # Generate a UUID for the data property based on its TermIRI
        uuid = generate_uuid5(result_doc.termIRI)
        formatted_objects.append([formatted_object, embeddings, uuid])  # Append formatted data
    
    return formatted_objects  # Return the list of formatted objects

def create_data_property_collection():
    print("Creating DataProperties collection")  # Indicate the start of collection creation
    # Fetch data from the endpoint for DataProperties
    endpoint_query_results = fetch_data_from_endpoint(url_endpoint, type="DataProperties")

    # Get the models and methodologies for embedding
    models, methodologies = get_data_property_collection_mappings()

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

    # Format objects for upload
    formatted_objects_for_upload = format_data_property_query_results(endpoint_query_results, methodologies, models, all_named_vectors)

    # Configure vectorizer for the named vectors
    vectorizer_config = [wvc.config.Configure.NamedVectors.none(name=x.name) for x in all_named_vectors]

    # Delete the existing collection if it exists
    client.collections.delete("DataProperties")

    print("Creating collection")  # Indicate that the collection is being created
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
    print("Uploading data")  # Indicate the start of data upload
    objects_to_upload = [wvc.data.DataObject(properties=d[0], vector=d[1], uuid=d[2]) for d in formatted_objects_for_upload]

    # Split objects into batches for uploading
    batches = split_list(objects_to_upload, 4)
    try:
        for i, batch in enumerate(batches):
            print(f"Uploading batch {i + 1}")  # Indicate which batch is being uploaded
            collection.data.insert_many(batch)  # Insert the batch into the collection
    except Exception as e:
        print(f"Error during insert_many: {e}")  # Catch and print any errors that occur during upload

def fill_data_property_copied_named_vectors():
    # Fetch all objects and named vectors to fill the copied named vectors
    all_objects = fetch_all_objects(collection="DataProperties")

    all_named_vectors = fetch_all_named_vectors(collection="DataProperties")
    
    # Get the mappings for the copied named vectors
    uuid_to_nv_mappings = get_copied_named_vectors(all_objects, all_named_vectors)
    
    # Fill the copied named vectors in the collection
    fill_copied_named_vectors(uuid_to_nv_mappings, "DataProperties")

def data_property_collection_creation():
    # Get ontology property data from endpoint
    print("Loading data")  # Indicate the start of data loading
    endpoint_query_results = fetch_data_from_endpoint(url_endpoint, type="data_properties")

    # Mappings between model names (formatted_object to _ format) and model instances
    models = {x.replace("-", "_"): SentenceTransformerEmbeddings(model_name=x) for x in model_names}

    # Mappings between mapping methodology names and functions
    methodologies = {"Label": embed_using_label, "Description": embed_using_desc, "Domain": embed_using_domain, "Range": embed_using_range}

    # All combinations between models, methodologies, and languages
    all_combos = []
    for model in models:
        for method in methodologies:
            for lang in languages:
                all_combos.append({"case_name": f"{model}___{method}___{lang}", "model": model, "method": method, "lang": lang})

    formatted_objects_for_upload = []  # List to hold formatted data property objects for upload

    # Formatting of the ontology data to upload to the collection
    print("Generating embeddings and formatting data")  # Indicate the start of data formatting
    for i, result_doc in enumerate(endpoint_query_results):
        tp = len(endpoint_query_results) / 10  # Progress tracker
        
        # Log progress every 10%
        if i % int(tp) == 0:
            print(i, "/", len(endpoint_query_results))
            
        formatted_object = {}  # Dictionary to hold formatted object properties

        # Generate a UUID for the object based on its TermIRI
        uuid = generate_uuid5(result_doc.termIRI)
        
        formatted_object["TermIRI"] = result_doc.termIRI
        formatted_object["RDF_type"] = result_doc.rdfType
        formatted_object["Ontology"] = result_doc.ontology
        formatted_object["Label"] = result_doc.label
        formatted_object["Description"] = result_doc.description
        formatted_object["Domain"] = result_doc.domain
        formatted_object["Range"] = result_doc.range
        formatted_object["Language"] = result_doc.language
        
        # Infer the label from TermIRI if it's not provided
        if not result_doc.label:
            if "#" in result_doc.termIRI:
                formatted_object["Label"] = result_doc.termIRI.split("#")[1]
            elif "/" in result_doc.termIRI:
                formatted_object["Label"] = result_doc.termIRI.split("/")[1]

        embeddings = {}  # Dictionary to hold embeddings for the data property
        for case in all_combos:
            name = case["case_name"]
            model = case["model"]
            method = case["method"]
            case_lang = case["lang"]
            # Generate embeddings only if the language matches
            if case_lang == result_doc.language:
                if name not in embeddings:
                    embeddings[name] = []  # Initialize the embeddings list if not already present
                
                embeddings[name] = methodologies[method](result_doc, models[model])  # Generate embeddings
            
        formatted_objects_for_upload.append([formatted_object, embeddings, uuid])  # Append formatted data

    # Configure vectorizer for the named vectors
    vectorizer_config = [wvc.config.Configure.NamedVectors.none(name=x["case_name"]) for x in all_combos]

    # Delete the existing collection if it exists
    client.collections.delete("DataProperties")

    # Create a new collection with the vectorizer configurations
    print("Creating collection")  # Indicate that the collection is being created
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
    print("Uploading data")  # Indicate the start of data upload
    objects_to_upload = []
    for d in formatted_objects_for_upload:
        objects_to_upload.append(wvc.data.DataObject(
            properties=d[0],
            vector=d[1], 
            uuid=d[2]
        ))
    
    # Log domains for each object
    for f in formatted_objects_for_upload:
        print(f[0]["Domain"])  # Print the domain of each object

    # Split objects into batches for uploading
    batches = split_list(objects_to_upload, 4)

    try:
        for i, batch in enumerate(batches):
            print(f"Uploading batch {i+1}")  # Indicate which batch is being uploaded
            collection.data.insert_many(batch)  # Insert the batch into the collection

    except Exception as e:
        print(f"Error during insert_many: {e}")  # Catch and print any errors that occur during upload


# RDFtype Collection Functions
def get_rdftype_collection_mappings():
    models = {x.replace("-", "_"): SentenceTransformerEmbeddings(model_name=x) for x in model_names}
    methodologies = {
        "Label": embed_using_label,
        "Description": embed_using_desc,
        "Superclass": embed_using_superclass
    }
    return models, methodologies

def format_rdftype_query_results(endpoint_query_results, methodologies, models, all_named_vectors):
    print("Formatting objects")
    formatted_objects = []
    for i, result_doc in enumerate(endpoint_query_results):
        formatted_object = {
            "TermIRI": result_doc.termIRI,
            "Ontology": result_doc.ontology,
            "Label": result_doc.label,
            "Description": result_doc.description,
            "Superclass": result_doc.superclass,
            "Language": result_doc.language
        }
        if not result_doc.label:
            if "#" in result_doc.termIRI:
                formatted_object["Label"] = result_doc.termIRI.split("#")[1]
            elif "/" in result_doc.termIRI:
                formatted_object["Label"] = result_doc.termIRI.split("/")[1]
        #print(result_doc.termIRI)
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
    print("Creating RDF_types collection")
    endpoint_query_results =  fetch_data_from_endpoint(url_endpoint, type="RDFtypes")

    models, methodologies = get_rdftype_collection_mappings()

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

    formatted_objects_for_upload = format_rdftype_query_results(endpoint_query_results, methodologies, models, all_named_vectors)
    print(formatted_objects_for_upload)
    # Configure vectorizer
    vectorizer_config = [wvc.config.Configure.NamedVectors.none(name=x.name) for x in all_named_vectors]

    client.collections.delete("RDFtypes")

    print("Creating collection")
    collection = client.collections.create(
            
            name="RDFtypes",
            
            description="Text2kg benchmark RDF_types",
            
            vectorizer_config=vectorizer_config,

            properties = [
                wvc.config.Property(name="TermIRI", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="Label", data_type=wvc.config.DataType.TEXT), # Some labels are inferred based on the IRI, TODO later
                wvc.config.Property(name="Description", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="Superclass", data_type=wvc.config.DataType.TEXT_ARRAY),
                wvc.config.Property(name="Language", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="Ontology", data_type=wvc.config.DataType.TEXT),
            ],
        )


    # Upload the formatted object data
    print("Uploading data")
    objects_to_upload = [wvc.data.DataObject(properties=d[0], vector=d[1], uuid=d[2]) for d in formatted_objects_for_upload]

    batches = split_list(objects_to_upload, 4)
    try:
        for i, batch in enumerate(batches):
            print(f"Uploading batch {i + 1}")
            collection.data.insert_many(batch)
    except Exception as e:
        print(f"Error during insert_many: {e}")

def fill_rdftype_copied_named_vectors():
    all_objects = fetch_all_objects(collection="RDFtypes")

    all_named_vectors = fetch_all_named_vectors(collection="RDFtypes")
    
    uuid_to_nv_mappings = get_copied_named_vectors(all_objects, all_named_vectors)
    
    fill_copied_named_vectors(uuid_to_nv_mappings, "RDFtypes")

def rdftype_collection_creation():
    # Get ontology property data from endpoint
    print("Loading data")
    endpoint_query_results = fetch_data_from_endpoint(url_endpoint, type="RDFtypes")

    ##print(ontologies)
    # Mappings between mapping methodology names and functions
    methodologies = {"Label": embed_using_label, "Description": embed_using_desc, "Superclass": embed_using_superclass}

    # All combinations between models, methodologies and languages
    all_combos = []
    for model in models:
        for method in methodologies:
            for lang in languages:
                all_combos.append({"case_name": f"{model}___{method}___{lang}","model": model, "method": method, "lang":lang})
    print([x["case_name"] for x in all_combos])

    formatted_objects_for_upload = []

    # Formatting of the ontology data to upload to the collection
    print("Generating embeddings and formatting data")
    for i, result_doc in enumerate(endpoint_query_results):
        tp = len(endpoint_query_results) / 10
        
        if i % int(tp) == 0:
            print(i, "/", len(endpoint_query_results))
            
        formatted_object = {}
        
        uuid = generate_uuid5(result_doc.termIRI)
        
        #{"termIRI": "http:termIRI1", "type": "http:class", 
        
        formatted_object["TermIRI"] = result_doc.termIRI
        formatted_object["Type"] = result_doc.rdfType
        formatted_object["Ontology"] = result_doc.ontology
        formatted_object["Label"] = result_doc.label
        formatted_object["Description"] = result_doc.description
        formatted_object["Superclass"] = result_doc.superclass
        formatted_object["Language"] = result_doc.language
        
        if not result_doc.label:
            if "#" in result_doc.termIRI:
                formatted_object["Label"] = result_doc.termIRI.split("#")[1]
            elif "/" in result_doc.termIRI:
                formatted_object["Label"] = result_doc.termIRI.split("/")[1]
        #print(formatted_object)
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
    
    # TODO: modify above so some uri for "special models" 

    # Delete the last iteration of the collection (for testing purposes)
    client.collections.delete("RDFtypes")

    # Create a new collection with the vectorizer configs
    print("Creating collection")
    collection = client.collections.create(
            name="RDFtypes",
            description="Text2kg benchmark RDF_types",
            
            vectorizer_config=vectorizer_config,

            properties = [
                wvc.config.Property(name="TermIRI", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="RDF_type", data_type=wvc.config.DataType.TEXT), # 
                wvc.config.Property(name="Label", data_type=wvc.config.DataType.TEXT), # Some labels are inferred based on the IRI, TODO later
                wvc.config.Property(name="Description", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="Superclass", data_type=wvc.config.DataType.TEXT_ARRAY),
                wvc.config.Property(name="Language", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="Ontology", data_type=wvc.config.DataType.TEXT),
            ],
        )

    
    # # Upload the formatted_object data
    print("Uploading data")
    objects_to_upload = []
    for d in formatted_objects_for_upload:
        objects_to_upload.append(wvc.data.DataObject(
        properties=d[0],
        vector=d[1],
        uuid=d[2]
        ))
        
    # for f in formatted_objects_for_upload:
    #     print(f[0]["Label"])


    batches = split_list(objects_to_upload, 4)

    try:
        for i, batch in enumerate(batches):
            print(f"Uploading batch {i+1}")
            # Perform the insert operation
            #collection.data.insert_many(objects_to_upload)
            collection.data.insert_many(batch)

    except Exception as e:
        print(f"Error during insert_many: {e}")



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
    print("Loading data")
    endpoint_query_results = fetch_data_from_endpoint(url_endpoint, type="Ontologies")

    # Mappings between mapping methodology names and functions
    methodologies = {"OntologySpecial": embed_ontology_classes, "OntologyDataproperties": embed_ontology_dataproperties}  # Add your embedding methodologies here


    # All combinations between models, methodologies and languages
    all_combos = []
    for model in models:
        for method in methodologies:
            for lang in languages:
                all_combos.append({"case_name": f"{model}___{method}___{lang}", "model": model, "method": method, "lang": lang})

    print([x["case_name"] for x in all_combos])

    formatted_objects_for_upload = []

    # Formatting of the ontology data to upload to the collection
    print("Generating embeddings and formatting data")
    for i, result_doc in enumerate(endpoint_query_results):
        tp = len(endpoint_query_results) / 10
        if int(tp) > 1:
            if i % int(tp) == 0:
                print(i, "/", len(endpoint_query_results))

        formatted_object = {}

        uuid = generate_uuid5(result_doc.ontologyIRI)

        # Map the ontology attributes to the respective collection properties
        formatted_object["OntologyIRI"] = result_doc.ontologyIRI
        
        formatted_object["Classes"] = result_doc.classes
        formatted_object["Dataproperties"] = result_doc.dataproperties
        formatted_object["Language"] = result_doc.language

        #print(formatted_object)
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

    # Delete the last iteration of the collection (for testing purposes)
    client.collections.delete("Ontologies")

    # Create a new collection with the vectorizer configs
    print("Creating collection")
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
    print("Uploading data")
    objects_to_upload = []
    for d in formatted_objects_for_upload:
        objects_to_upload.append(wvc.data.DataObject(
            properties=d[0],
            vector=d[1],
            uuid=d[2]
        ))


    batches = split_list(objects_to_upload, 4)

    try:
        for i, batch in enumerate(batches):
            print(f"Uploading batch {i+1}")
            collection.data.insert_many(batch)

    except Exception as e:
        print(f"Error during insert_many: {e}",  traceback.format_exc())
   
   

# OBJECT PROPERTIES: {DOMAIN: CLASSES, RANGE: CLASSES}
# INDIVIDUALS: {DOMAIN: CLASSES?, RANGE: CLASSES?}
# CLASSES: {SUBCLASS: CLASSES, SUPERCLASS: CLASSES}
# RDF_TYPES: {SUPERCLASS: RDF_TYPES}
# ONTOLOGIES: {}
# RDF_TYPES: {}


# Create a client instance
headers = {
    "X-HuggingFace-Api-Key": hf_key,
}



if __name__ == "__main__":
    
    client = weaviate.connect_to_local(
        
        port=8011,
        grpc_port=50051,
        headers=headers
        
    )

    client.collections.delete_all()

    try:
        
        create_object_property_collection()
        create_data_property_collection()
        create_class_collection()
        create_rdftype_collection()
        create_individuals_collection()
        #create_ontology_collection()
        print("END OF COLLECTION CREATION")
        for x in client.collections.list_all(simple=True):
            print(client.collections.get(name=x), "exists")
        fill_object_property_copied_named_vectors()
        fill_data_property_copied_named_vectors()
        fill_class_copied_named_vectors()
        fill_rdftype_copied_named_vectors()
        fill_individuals_copied_named_vectors()
        #fill_ontology_copied_named_vectors()

    except Exception as e:
        print("Error", e, traceback.format_exc())
        
    finally;
        client.close()