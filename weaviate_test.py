import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import Filter
from weaviate.util import generate_uuid5
from langchain_community.embeddings import SentenceTransformerEmbeddings
from weaviate_test_aux_functions import *
from dotenv import load_dotenv
import os
import csv
load_dotenv()
wcd_url = os.getenv("WCD_URL")
wcd_api_key = os.getenv("WCD_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
url_endpoint = "http://95.217.207.179:8995/sparql/"
# Create a client instance
print("Connecting to cloud")
client = weaviate.connect_to_local(
    
    port=8085,
    grpc_port=50051,
    
)

create_new = True

def class_collection_creation():
    # Get ontology property data from endpoint
    print("Loading data")
    all_data = get_data(url_endpoint, type="classes")

    # Names of models to test
    models = ["LaBSE","all-MiniLM-L6-v2","all-MiniLM-L12-v2","all-distilroberta-v1","paraphrase-multilingual-MiniLM-L12-v2","multi-qa-mpnet-base-cos-v1"]

    # Mappings between model names (formatted to _ format) and model instances
    models = {x.replace("-", "_"): SentenceTransformerEmbeddings(model_name=x) for x in models}

    #print(ontologies)
    # Mappings between mapping methodology names and functions
    methodologies = {"Label": embed_using_label, "Description": embed_using_desc, "SubclassPLUSSuperclass": embed_using_subclass_plus_superclass, "Subclass": embed_using_subclass, "Superclass": embed_using_superclass}

    # Languages to consider
    languages = ["en", "fr", "None"]

    # All combinations between models, methodologies and languages
    all_combos = []
    for model in models:
        for method in methodologies:
            for lang in languages:
                all_combos.append({"case_name": f"{model}___{method}___{lang}","model": model, "method": method, "lang":lang})

    ##print(set([c["case_name"] for c in all_combos]))

    formatted_for_upload = []
    if create_new:
        # Formatting of the ontology data to upload to the collection
        print("Generating embeddings and formatting data")
        for i, d in enumerate(all_data):
            tp = len(all_data) / 10
            
            if i % int(tp) == 0:
                print(i, "/", len(all_data))
                
            formatted = {}
            term, ontology, datatype, label, description, domain, rang, language = d
            uuid = generate_uuid5(term)
            formatted["Term"] = term
            formatted["RDF_type"] = datatype
            formatted["Ontology"] = ontology
            formatted["Label"] = label
            formatted["Description"] = description
            formatted["Subclass"] = domain
            formatted["Superclass"] = rang
            formatted["Language"] = language

            embeddings = {}
            for case in all_combos:
                name = case["case_name"]
                model = case["model"]
                method = case["method"]
                case_lang = case["lang"]
                if case_lang == language:
                    
                    if not name in embeddings:
                        embeddings[name] = []
                    
                    embeddings[name] = methodologies[method](d, models[model])
                    
            formatted_for_upload.append([formatted, embeddings, uuid])

    # Configurations for custom vectorizers (one for every case)
    vectorizer_config = [wvc.config.Configure.NamedVectors.none(name=x["case_name"]) for x in all_combos]

    if create_new:
        # Delete the last iteration of the collection (for testing purposes)
        client.collections.delete("Classes")

        # Create a new collection with the vectorizer configs
        print("Creating collection")
        collection = client.collections.create(
                name="Classes",
                description="Text2kg benchmark classes",
                
                vectorizer_config=vectorizer_config,

                properties = [
                    wvc.config.Property(name="Term", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="RDF_type", data_type=wvc.config.DataType.TEXT), # 
                    wvc.config.Property(name="Label", data_type=wvc.config.DataType.TEXT), # Some labels are inferred based on the IRI, TODO later
                    wvc.config.Property(name="Description", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="Subclass", data_type=wvc.config.DataType.TEXT_ARRAY),
                    wvc.config.Property(name="Superclass", data_type=wvc.config.DataType.TEXT_ARRAY),
                    wvc.config.Property(name="Language", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="Ontology", data_type=wvc.config.DataType.TEXT),
                ],
            )
    else:
        collection = client.collections.get(name="Classes")
    
    # # Upload the formatted data
    print("Uploading data")
    objects_to_upload = []
    for d in formatted_for_upload:
        objects_to_upload.append(wvc.data.DataObject(
        properties=d[0],
        vector=d[1],
        uuid=d[2]
        ))
        
    for f in formatted_for_upload:
        print(f[0]["Label"])
    if create_new:

        batches = split_list(objects_to_upload, 4)

        try:
            for i, batch in enumerate(batches):
                print(f"Uploading batch {i+1}")
                # Perform the insert operation
                #collection.data.insert_many(objects_to_upload)
                collection.data.insert_many(batch)

        except Exception as e:
            print(f"Error during insert_many: {e}")

def individual_collection_creation():
    # Get ontology property data from endpoint
    print("Loading data")
    all_data = get_data(url_endpoint, type="individuals")

    # Names of models to test
    models = ["LaBSE","all-MiniLM-L6-v2","all-MiniLM-L12-v2","all-distilroberta-v1","paraphrase-multilingual-MiniLM-L12-v2","multi-qa-mpnet-base-cos-v1"]

    # Mappings between model names (formatted to _ format) and model instances
    models = {x.replace("-", "_"): SentenceTransformerEmbeddings(model_name=x) for x in models}

    #print(ontologies)
    # Mappings between mapping methodology names and functions
    methodologies = {"Label": embed_using_label, "Description": embed_using_desc, "DomainPLUSRange": embed_using_domain_plus_range, "Domain": embed_using_domain, "Range": embed_using_range}

    # Languages to consider
    languages = ["en", "fr", "None"]#["en", "fr", "None"]

    # All combinations between models, methodologies and languages
    all_combos = []
    for model in models:

        for method in methodologies:
            for lang in languages:
                all_combos.append({"case_name": f"{model}___{method}___{lang}","model": model, "method": method, "lang":lang})

    #print(set([c["case_name"] for c in all_combos]))

    formatted_for_upload = []
    if create_new:
        # Formatting of the ontology data to upload to the collection
        print("Generating embeddings and formatting data")
        for i, d in enumerate(all_data):
            tp = len(all_data) / 10
            
            if i % int(tp) == 0:
                print(i, "/", len(all_data))
                
            formatted = {}
            term, ontology, datatype, label, description, domain, rang, language = d
            uuid = generate_uuid5(term)
            formatted["Term"] = term
            formatted["RDF_type"] = datatype
            formatted["Ontology"] = ontology
            formatted["Label"] = label
            formatted["Description"] = description
            formatted["Domain"] = domain
            formatted["Range"] = rang
            formatted["Language"] = language

            
            embeddings = {}
            for case in all_combos:
                name = case["case_name"]
                model = case["model"]
                method = case["method"]
                case_lang = case["lang"]
                if case_lang == language:
                    
                    if not name in embeddings:
                        embeddings[name] = []
                    
                    embeddings[name] = methodologies[method](d, models[model])
                    
            formatted_for_upload.append([formatted, embeddings, uuid])

    # Configurations for custom vectorizers (one for every case)
    vectorizer_config = [wvc.config.Configure.NamedVectors.none(name=x["case_name"]) for x in all_combos]

    if create_new:
        # Delete the last iteration of the collection (for testing purposes)
        client.collections.delete("Individuals")

        # Create a new collection with the vectorizer configs
        print("Creating collection")
        collection = client.collections.create(
                name="Individuals",
                description="Text2kg benchmark individuals",
                
                vectorizer_config=vectorizer_config,

                # I believe that defining those "properties" is only needed in case we use Weviate-hosted models, to determine what they should embed for each named vector
                # But I defined them anyway
                properties=[
                    wvc.config.Property(name="Term", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="RDF_type", data_type=wvc.config.DataType.TEXT), # 
                    wvc.config.Property(name="Label", data_type=wvc.config.DataType.TEXT), # Some labels are inferred based on the IRI, TODO later
                    wvc.config.Property(name="Description", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="Domain", data_type=wvc.config.DataType.TEXT_ARRAY),
                    wvc.config.Property(name="Range", data_type=wvc.config.DataType.TEXT_ARRAY),
                    wvc.config.Property(name="Language", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="Ontology", data_type=wvc.config.DataType.TEXT),

            ],
            )
    else:
        collection = client.collections.get(name="Individuals")
    
    # # Upload the formatted data
    print("Uploading data")
    objects_to_upload = []
    for d in formatted_for_upload:
        objects_to_upload.append(wvc.data.DataObject(
        properties=d[0],
        vector=d[1],
        uuid=d[2]
        ))
        
    for f in formatted_for_upload:
        print(f[0]["Domain"])
    if create_new:


        batches = split_list(objects_to_upload, 4)


        try:
            for i, batch in enumerate(batches):
                print(f"Uploading batch {i+1}")
                # Perform the insert operation
                #collection.data.insert_many(objects_to_upload)
                collection.data.insert_many(batch)

        except Exception as e:
            print(f"Error during insert_many: {e}")

def object_property_collection_creation():
    # Get ontology property data from endpoint
    print("Loading data")
    all_data = get_data(url_endpoint, type="object_properties")

    # Names of models to test
    models = ["LaBSE","all-MiniLM-L6-v2","all-MiniLM-L12-v2","all-distilroberta-v1","paraphrase-multilingual-MiniLM-L12-v2","multi-qa-mpnet-base-cos-v1"]

    # Mappings between model names (formatted to _ format) and model instances
    models = {x.replace("-", "_"): SentenceTransformerEmbeddings(model_name=x) for x in models}

    #print(ontologies)
    # Mappings between mapping methodology names and functions
    methodologies = {"Label": embed_using_label, "Description": embed_using_desc, "DomainPLUSRange": embed_using_domain_plus_range, "Domain": embed_using_domain, "Range": embed_using_range}

    # Languages to consider
    languages = ["en", "fr", "None"]#["en", "fr", "None"]

    # All combinations between models, methodologies and languages
    all_combos = []
    for model in models:

        for method in methodologies:
            for lang in languages:
                all_combos.append({"case_name": f"{model}___{method}___{lang}","model": model, "method": method, "lang":lang})

    #print(set([c["case_name"] for c in all_combos]))

    formatted_for_upload = []
    if create_new:
        # Formatting of the ontology data to upload to the collection
        print("Generating embeddings and formatting data")
        for i, d in enumerate(all_data):
            tp = len(all_data) / 10
            
            if i % int(tp) == 0:
                print(i, "/", len(all_data))
                
            formatted = {}
            term, ontology, datatype, label, description, domain, rang, language = d
            uuid = generate_uuid5(term)
            formatted["Term"] = term
            formatted["RDF_type"] = datatype
            formatted["Ontology"] = ontology
            formatted["Label"] = label
            formatted["Description"] = description
            formatted["Domain"] = domain
            formatted["Range"] = rang
            formatted["Language"] = language

            
            embeddings = {}
            for case in all_combos:
                name = case["case_name"]
                model = case["model"]
                method = case["method"]
                case_lang = case["lang"]
                if case_lang == language:
                    
                    if not name in embeddings:
                        embeddings[name] = []
                    
                    embeddings[name] = methodologies[method](d, models[model])
                    
            formatted_for_upload.append([formatted, embeddings, uuid])

    # Configurations for custom vectorizers (one for every case)
    vectorizer_config = [wvc.config.Configure.NamedVectors.none(name=x["case_name"]) for x in all_combos]

    if create_new:
        # Delete the last iteration of the collection (for testing purposes)
        client.collections.delete("ObjectProperties")

        # Create a new collection with the vectorizer configs
        print("Creating collection")
        collection = client.collections.create(
                name="ObjectProperties",
                description="Text2kg benchmark properties",
                
                vectorizer_config=vectorizer_config,

                properties=[
                    wvc.config.Property(name="Term", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="RDF_type", data_type=wvc.config.DataType.TEXT), # 
                    wvc.config.Property(name="Label", data_type=wvc.config.DataType.TEXT), # Some labels are inferred based on the IRI, TODO later
                    wvc.config.Property(name="Description", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="Domain", data_type=wvc.config.DataType.TEXT_ARRAY),
                    wvc.config.Property(name="Range", data_type=wvc.config.DataType.TEXT_ARRAY),
                    wvc.config.Property(name="Language", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="Ontology", data_type=wvc.config.DataType.TEXT),

            ],
            )
    else:
        collection = client.collections.get(name="ObjectProperties")
    
    # # Upload the formatted data
    print("Uploading data")
    objects_to_upload = []
    for d in formatted_for_upload:
        objects_to_upload.append(wvc.data.DataObject(
        properties=d[0],
        vector=d[1],
        uuid=d[2]
        ))
        
    for f in formatted_for_upload:
        print(f[0]["Domain"])
    if create_new:

        batches = split_list(objects_to_upload, 4)


        try:
            for i, batch in enumerate(batches):
                print(f"Uploading batch {i+1}")
                # Perform the insert operation
                #collection.data.insert_many(objects_to_upload)
                collection.data.insert_many(batch)

        except Exception as e:
            print(f"Error during insert_many: {e}")

def data_property_collection_creation():
    # Get ontology property data from endpoint
    print("Loading data")
    all_data = get_data(url_endpoint, type="data_properties")

    # Names of models to test
    models = ["LaBSE","all-MiniLM-L6-v2","all-MiniLM-L12-v2","all-distilroberta-v1","paraphrase-multilingual-MiniLM-L12-v2","multi-qa-mpnet-base-cos-v1"]

    # Mappings between model names (formatted to _ format) and model instances
    models = {x.replace("-", "_"): SentenceTransformerEmbeddings(model_name=x) for x in models}

    ##print(ontologies)
    # Mappings between mapping methodology names and functions
    methodologies = {"Label": embed_using_label, "Description": embed_using_desc, "DomainPLUSRange": embed_using_domain_plus_range, "Domain": embed_using_domain, "Range": embed_using_range}

    # Languages to consider
    languages = ["en", "fr", "None"]#["en", "fr", "None"]

    # All combinations between models, methodologies and languages
    all_combos = []
    for model in models:

        for method in methodologies:
            for lang in languages:
                all_combos.append({"case_name": f"{model}___{method}___{lang}","model": model, "method": method, "lang":lang})

    ##print(set([c["case_name"] for c in all_combos]))

    formatted_for_upload = []
    if create_new:
        # Formatting of the ontology data to upload to the collection
        print("Generating embeddings and formatting data")
        for i, d in enumerate(all_data):
            tp = len(all_data) / 10
            
            if i % int(tp) == 0:
                print(i, "/", len(all_data))
                
            formatted = {}
            term, ontology, datatype, label, description, domain, rang, language = d
            uuid = generate_uuid5(term)
            formatted["Term"] = term
            formatted["RDF_type"] = datatype
            formatted["Ontology"] = ontology
            formatted["Label"] = label
            formatted["Description"] = description
            formatted["Domain"] = domain
            formatted["Range"] = rang
            formatted["Language"] = language

            
            embeddings = {}
            for case in all_combos:
                name = case["case_name"]
                model = case["model"]
                method = case["method"]
                case_lang = case["lang"]
                if case_lang == language:
                    
                    if not name in embeddings:
                        embeddings[name] = []
                    
                    embeddings[name] = methodologies[method](d, models[model])
                    
            formatted_for_upload.append([formatted, embeddings, uuid])

    # Configurations for custom vectorizers (one for every case)
    vectorizer_config = [wvc.config.Configure.NamedVectors.none(name=x["case_name"]) for x in all_combos]
    # TODO: try huggingface instead of none
    
    if create_new:
        # Delete the last iteration of the collection (for testing purposes)
        client.collections.delete("DataProperties")

        # Create a new collection with the vectorizer configs
        print("Creating collection")
        collection = client.collections.create(
                name="DataProperties",
                description="Text2kg benchmark data properties",
                
                vectorizer_config=vectorizer_config,

                properties=[
                    wvc.config.Property(name="Term", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="RDF_type", data_type=wvc.config.DataType.TEXT), # 
                    wvc.config.Property(name="Label", data_type=wvc.config.DataType.TEXT), # Some labels are inferred based on the IRI, TODO later
                    wvc.config.Property(name="Description", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="Domain", data_type=wvc.config.DataType.TEXT_ARRAY),
                    wvc.config.Property(name="Range", data_type=wvc.config.DataType.TEXT_ARRAY),
                    wvc.config.Property(name="Language", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="Ontology", data_type=wvc.config.DataType.TEXT),

            ],
        )
    else:
        collection = client.collections.get(name="DataProperties")
    
    # # Upload the formatted data
    print("Uploading data")
    objects_to_upload = []
    for d in formatted_for_upload:
        objects_to_upload.append(wvc.data.DataObject(
        properties=d[0],
        vector=d[1], 
        uuid=d[2]
        ))
        
    for f in formatted_for_upload:
        print(f[0]["Domain"])
    if create_new:
        
        batches = split_list(objects_to_upload, 4)


        try:
            for i, batch in enumerate(batches):
                print(f"Uploading batch {i+1}")
                # Perform the insert operation
                #collection.data.insert_many(objects_to_upload)
                collection.data.insert_many(batch)

        except Exception as e:
            print(f"Error during insert_many: {e}")

def rdftype_collection_creation():
    # Get ontology property data from endpoint
    print("Loading data")
    all_data = get_data(url_endpoint, type="RDF_types")

    # Names of models to test
    models = ["LaBSE","all-MiniLM-L6-v2","all-MiniLM-L12-v2","all-distilroberta-v1","paraphrase-multilingual-MiniLM-L12-v2","multi-qa-mpnet-base-cos-v1"]

    # Mappings between model names (formatted to _ format) and model instances
    models = {x.replace("-", "_"): SentenceTransformerEmbeddings(model_name=x) for x in models}

    ##print(ontologies)
    # Mappings between mapping methodology names and functions
    methodologies = {"Label": embed_using_label, "Description": embed_using_desc, "Superclass": embed_using_superclass}

   # Languages to consider
    languages = ["en", "fr", "None"]

    # All combinations between models, methodologies and languages
    all_combos = []
    for model in models:
        for method in methodologies:
            for lang in languages:
                all_combos.append({"case_name": f"{model}___{method}___{lang}","model": model, "method": method, "lang":lang})
    print([x["case_name"] for x in all_combos])
    ##print(set([c["case_name"] for c in all_combos]))

    formatted_for_upload = []
    if create_new:
        # Formatting of the ontology data to upload to the collection
        print("Generating embeddings and formatting data")
        for i, d in enumerate(all_data):
            tp = len(all_data) / 10
            
            if i % int(tp) == 0:
                print(i, "/", len(all_data))
                
            formatted = {}
            term, ontology, datatype, label, description, domain, rang, language = d
            
            uuid = generate_uuid5(term)
            formatted["Term"] = term
            formatted["Type"] = datatype
            formatted["Ontology"] = ontology
            formatted["Label"] = label
            formatted["Description"] = description
            
            formatted["Superclass"] = rang
            formatted["Language"] = language
                
            print(formatted)
            embeddings = {}
            for case in all_combos:
                name = case["case_name"]
                model = case["model"]
                method = case["method"]
                case_lang = case["lang"]
                if case_lang == language:
                    
                    if not name in embeddings:
                        embeddings[name] = []
                    
                    embeddings[name] = methodologies[method](d, models[model])
                    
            formatted_for_upload.append([formatted, embeddings, uuid])

    # Configurations for custom vectorizers (one for every case)
    vectorizer_config = [wvc.config.Configure.NamedVectors.none(name=x["case_name"]) for x in all_combos]
    
    # TODO: modify above so some uri for "special models" 

    if create_new:
        # Delete the last iteration of the collection (for testing purposes)
        client.collections.delete("RDF_types")

        # Create a new collection with the vectorizer configs
        print("Creating collection")
        collection = client.collections.create(
                name="RDF_types",
                description="Text2kg benchmark RDF_types",
                
                vectorizer_config=vectorizer_config,

                properties = [
                    wvc.config.Property(name="Term", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="RDF_type", data_type=wvc.config.DataType.TEXT), # 
                    wvc.config.Property(name="Label", data_type=wvc.config.DataType.TEXT), # Some labels are inferred based on the IRI, TODO later
                    wvc.config.Property(name="Description", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="Superclass", data_type=wvc.config.DataType.TEXT_ARRAY),
                    wvc.config.Property(name="Language", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="Ontology", data_type=wvc.config.DataType.TEXT),
                ],
            )
    else:
        collection = client.collections.get(name="RDF_types")
    
    # # Upload the formatted data
    print("Uploading data")
    objects_to_upload = []
    for d in formatted_for_upload:
        objects_to_upload.append(wvc.data.DataObject(
        properties=d[0],
        vector=d[1],
        uuid=d[2]
        ))
        
    # for f in formatted_for_upload:
    #     print(f[0]["Label"])
    if create_new:

        batches = split_list(objects_to_upload, 4)

        try:
            for i, batch in enumerate(batches):
                print(f"Uploading batch {i+1}")
                # Perform the insert operation
                #collection.data.insert_many(objects_to_upload)
                collection.data.insert_many(batch)

        except Exception as e:
            print(f"Error during insert_many: {e}")


import traceback
if create_new:
    try:
        rdftype_collection_creation()
    except Exception as e:
        print("Error rdftype", e, traceback.format_exc())
    
    try:
        print("Creating class collection")
        class_collection_creation()
    except Exception as e:
        print("Error class", e, traceback.format_exc())
    try:
        print("Creating data_property collection")
        data_property_collection_creation()
    except Exception as e:
        print("Error data", e, traceback.format_exc())
    try:
        print("Creating individual collection")
        individual_collection_creation()
    except Exception as e:
        print("Error individual", e, traceback.format_exc())
    try:
        print("Creating object property collection")
        object_property_collection_creation()
    except Exception as e:
        print("Error object property", e, traceback.format_exc())
        


