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

wcd_url = os.getenv("WCD_URL")
wcd_api_key = os.getenv("WCD_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
hf_key = os.getenv("HF_KEY")
url_endpoint =  os.getenv("SPARQL_ENDPOINT")
local_weaviate_port = int(os.getenv("WEAVIATE_PORT"))
local_weaviate_port_grpc = int(os.getenv("WEAVIATE_PORT_GRPC"))

# def class_collection_creation():
#     # Get ontology property data from endpoint
#     print("Loading data")
#     endpoint_query_results = fetch_data_from_endpoint(url_endpoint, type="classes")

#     # Names of models to test
#     model_names = ["LaBSE","all-MiniLM-L6-v2","all-MiniLM-L12-v2","all-distilroberta-v1","paraphrase-multilingual-MiniLM-L12-v2","multi-qa-mpnet-base-cos-v1"]

#     # # Mappings between model names (formatted_object to _ format) and model instances
#     models = {x.replace("-", "_"): SentenceTransformerEmbeddings(model_name=x) for x in model_names}

#     #print(ontologies)
#     methodologies = {"Label": embed_using_label, "Description": embed_using_desc, "SubclassPLUSSuperclass": embed_using_subclass_plus_superclass, "Subclass": embed_using_subclass, "Superclass": embed_using_superclass}

#     # Languages to consider
#     languages = ["en", "fr", "None"]
    

# #    relationship_properties = {"SuperClass": copy_over_superclass_vectors}
    
#     # All combinations between models, methodologies and languages
#     all_combos = []
#     for model in models:
#         for method in methodologies:
#             for lang in languages:
#                 all_combos.append({"case_name": f"{model}___{method}___{lang}","model": model, "method": method, "lang":lang})

#     ##print(set([c["case_name"] for c in all_combos]))

#     formatted_objects_for_upload = []
#     if create_new:
        
#         # Formatting of the ontology data to upload to the collection
#         print("Generating embeddings and formatting data")
#         for i, result_doc in enumerate(endpoint_query_results):
#             tp = len(endpoint_query_results) / 10
            
#             if i % int(tp) == 0:
#                 print(i, "/", len(endpoint_query_results))
                
#             formatted_object = {}
           
#             uuid = generate_uuid5(result_doc.termIRI)
#             formatted_object["TermIRI"] = result_doc.termIRI
#             formatted_object["RDF_type"] = result_doc.rdfType
#             formatted_object["Ontology"] = result_doc.ontology
#             formatted_object["Label"] = result_doc.label
#             formatted_object["Description"] = result_doc.description
#             formatted_object["Subclass"] = result_doc.subclass
#             #print(result_doc.subclass)
#             formatted_object["Superclass"] = result_doc.superclass
#             #print(result_doc.superclass)
#             formatted_object["Language"] = result_doc.language

#             embeddings = {}
#             for case in all_combos:
#                 name = case["case_name"]
#                 model = case["model"]
#                 method = case["method"]
#                 case_lang = case["lang"]
#                 if case_lang == result_doc.language:
                    
#                     if not name in embeddings:
#                         embeddings[name] = []
                    
#                     embeddings[name] = methodologies[method](result_doc, models[model])

#             formatted_objects_for_upload.append([formatted_object, embeddings, uuid])

#     # Configurations for custom vectorizers (one for every case)
#     vectorizer_config = [wvc.config.Configure.NamedVectors.none(name=x["case_name"]) for x in all_combos]

#     # TODO: Define new vectors
#     # parentClassC
#     {properties : copy_relation}
    
#     if create_new:
#         # Delete the last iteration of the collection (for testing purposes)
#         client.collections.delete("Classes")

#         # Create a new collection with the vectorizer configs
#         print("Creating collection")
#         collection = client.collections.create(
#                 name="Classes",
#                 description="Text2kg benchmark classes",
                
#                 vectorizer_config=vectorizer_config,

#                 properties = [
#                     wvc.config.Property(name="TermIRI", data_type=wvc.config.DataType.TEXT),
#                     wvc.config.Property(name="RDF_type", data_type=wvc.config.DataType.TEXT), # 
#                     wvc.config.Property(name="Label", data_type=wvc.config.DataType.TEXT), # Some labels are inferred based on the IRI, TODO later
#                     wvc.config.Property(name="Description", data_type=wvc.config.DataType.TEXT),
#                     wvc.config.Property(name="Subclass", data_type=wvc.config.DataType.TEXT_ARRAY),
#                     wvc.config.Property(name="Superclass", data_type=wvc.config.DataType.TEXT_ARRAY),
#                     wvc.config.Property(name="Language", data_type=wvc.config.DataType.TEXT),
#                     wvc.config.Property(name="Ontology", data_type=wvc.config.DataType.TEXT),
#                 ],
#             )
#     else:
#         collection = client.collections.get(name="Classes")
    
#     # # Upload the formatted_object data
#     print("Uploading data")
#     objects_to_upload = []
#     for d in formatted_objects_for_upload:
#         objects_to_upload.append(wvc.data.DataObject(
#             properties=d[0],
#             vector=d[1],
#             uuid=d[2]
#         ))
        
#     # for f in formatted_objects_for_upload:
#     #     print(f[0]["Label"])
#     if create_new:

#         batches = split_list(objects_to_upload, 4)

#         try:
#             for i, batch in enumerate(batches):
#                 print(f"Uploading batch {i+1}")
#                 # Perform the insert operation
#                 #collection.data.insert_many(objects_to_upload)
#                 collection.data.insert_many(batch)

#         except Exception as e:
#             print(f"Error during insert_many: {e}")

def class_collection_creation_hf_integration():
    # Get ontology property data from endpoint
    print("Loading data")
    endpoint_query_results = fetch_data_from_endpoint(url_endpoint, type="classes")

    # Names of models to test
    model_names = ["LaBSE","all-MiniLM-L6-v2","all-MiniLM-L12-v2","all-distilroberta-v1","paraphrase-multilingual-MiniLM-L12-v2","multi-qa-mpnet-base-cos-v1"]

    # # Mappings between model names (formatted_object to _ format) and model instances
    models = {x.replace("-", "_"): SentenceTransformerEmbeddings(model_name=x) for x in model_names}

    #print(ontologies)
    # Mappings between mapping methodology names and functions
    methodologies = {"Label": embed_using_label, "Description": embed_using_desc, "SubclassPLUSSuperclass": embed_using_subclass_plus_superclass, "Subclass": embed_using_subclass, "Superclass": embed_using_superclass}

    # Languages to consider
    languages = ["en", "fr", "None"]

    @dataclass
    class VirtualNamedVectorEmbedding: #-> Rquivalent to named___vector___strings I use right now + the new copied vectors
        # On __init__ vectors (AKA named___vector___strings)
        field_name: str = field(default_factory=str) #-> description
        language: str = field(default_factory=str) #-> en
        vectorizer: str = field(default_factory=str) # mostly model name and version -> labse
        embed_strategy: str = field(default_factory=str) #-> flatten
        
        # CopiedVectors
        copy_relationship : str = field(default_factory=str) #e.g. "domain"
        copy_relationship_index : str = field(default_factory=str)
        
        name = f"{vectorizer}___default___{field_name}___{language}___CP_SEPARATOR___{copy_relationship}___{copy_relationship_index}"
    
    all_named_vectors = []
    for field_name in ["Label", "Description", "Subclass", "Superclass"]:
        for lang in languages:
            for model in models:
                embed_strategy = methodologies[field_name]
                for copy_relationship, collection in [("Domain", "Classes_hf")]:
                    all_named_vectors.append(VirtualNamedVectorEmbedding(field_name, lang, model, embed_strategy, copy_relationship, collection))

    formatted_objects_for_upload = []
    if create_new:
        
        # Formatting of the ontology data to upload to the collection
        print("Generating embeddings and formatting data")
        for i, result_doc in enumerate(endpoint_query_results):
            tp = len(endpoint_query_results) / 10
            
            if i % int(tp) == 0:
                print(i, "/", len(endpoint_query_results))
                
            formatted_object = {}

            uuid = generate_uuid5(result_doc.termIRI)
            formatted_object["TermIRI"] = result_doc.termIRI
            formatted_object["RDF_type"] = result_doc.rdfType
            formatted_object["Ontology"] = result_doc.ontology
            formatted_object["Label"] = result_doc.label
            print(result_doc.label)
            formatted_object["Description"] = result_doc.description
            formatted_object["Subclass"] = result_doc.subclass
            formatted_object["Superclass"] = result_doc.superclass
            formatted_object["Language"] = result_doc.language

                    
            formatted_objects_for_upload.append([formatted_object, uuid])

    # Configurations for custom vectorizers (one for every case)
    # vectorizer_config = []
    # for x in all_combos:
    #     print("Named vector:", x["case_name"])
    #     print("Model:", )
    vectorizer_config = [wvc.config.Configure.NamedVectors.text2vec_huggingface(name=x.name, model=x.vectorizer, source_properties=x.field_name) for x in all_named_vectors] 
    
    if create_new:
        # Delete the last iteration of the collection (for testing purposes)
        client.collections.delete("Classes_hf")

        # Create a new collection with the vectorizer configs
        print("Creating collection")
        collection = client.collections.create(
                name="Classes_hf",
                description="Text2kg benchmark classes",
                
                vectorizer_config=vectorizer_config,

                properties = [
                    wvc.config.Property(name="TermIRI", data_type=wvc.config.DataType.TEXT),
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
        collection = client.collections.get(name="Classes_hf")
    


    # # Upload the formatted_object data
    print("Uploading data")
    objects_to_upload = []
    for d in formatted_objects_for_upload:
        objects_to_upload.append(wvc.data.DataObject(
        properties=d[0],
        uuid=d[1]
        ))
    

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
    endpoint_query_results = fetch_data_from_endpoint(url_endpoint, type="individuals")

    # Names of models to test
    models = ["LaBSE","all-MiniLM-L6-v2","all-MiniLM-L12-v2","all-distilroberta-v1","paraphrase-multilingual-MiniLM-L12-v2","multi-qa-mpnet-base-cos-v1"]

    # Mappings between model names (formatted_object to _ format) and model instances
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

    formatted_objects_for_upload = []
    if create_new:
        # Formatting of the ontology data to upload to the collection
        print("Generating embeddings and formatting data")
        for i, result_doc in enumerate(endpoint_query_results):
            tp = len(endpoint_query_results) / 10
            
            if i % int(tp) == 0:
                print(i, "/", len(endpoint_query_results))
                
            formatted_object = {}
            
            uuid = generate_uuid5(result_doc.termIRI)
            formatted_object["TermIRI"] = result_doc.termIRI
            formatted_object["RDF_type"] = result_doc.rdfType
            formatted_object["Ontology"] = result_doc.ontology
            formatted_object["Label"] = result_doc.label
            formatted_object["Description"] = result_doc.description
            formatted_object["Domain"] = result_doc.domain
            formatted_object["Range"] = result_doc.range
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

    if create_new:
        # Delete the last iteration of the collection (for testing purposes)
        client.collections.delete("Individuals")

        # Create a new collection with the vectorizer configs
        print("Creating collection")
        collection = client.collections.create(
                name="Individuals",
                description="Text2kg benchmark individuals",
                
                vectorizer_config=vectorizer_config,

                # I believe that defining those "properties" is only needed in case we use Weviate-hosted models, to determIRIine what they should embed for each named vector
                # But I defined them anyway
                properties=[
                    wvc.config.Property(name="TermIRI", data_type=wvc.config.DataType.TEXT),
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
    
    # # Upload the formatted_object data
    print("Uploading data")
    objects_to_upload = []
    for d in formatted_objects_for_upload:
        objects_to_upload.append(wvc.data.DataObject(
        properties=d[0],
        vector=d[1],
        uuid=d[2]
        ))
        
    for f in formatted_objects_for_upload:
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
    endpoint_query_results = fetch_data_from_endpoint(url_endpoint, type="object_properties")
    print(endpoint_query_results[:50])
    # Names of models to test
    models = ["LaBSE","all-MiniLM-L6-v2","all-MiniLM-L12-v2","all-distilroberta-v1","paraphrase-multilingual-MiniLM-L12-v2","multi-qa-mpnet-base-cos-v1"]

    # Mappings between model names (formatted_object to _ format) and model instances
    models = {x.replace("-", "_"): SentenceTransformerEmbeddings(model_name=x) for x in models}

    #print(ontologies)
    # Mappings between mapping methodology names and functions
    methodologies = {"Label": embed_using_label, "Description": embed_using_desc, "DomainPLUSRange": embed_using_domain_plus_range, "Domain": embed_using_domain, "Range": embed_using_range} # TODO: Probably dont need domainplusrange

    # TODO: Make variable names such as "data" more specific and interpretable
    
    # Languages to consider
    languages = ["en", "fr", "None"]


    @dataclass
    class VirtualNamedVectorEmbedding:
        # On __init__ vectors (AKA named___vector___strings)
        field_name: str = field(default_factory=str)  # description
        language: str = field(default_factory=str)  # e.g. "en"
        vectorizer: str = field(default_factory=str)  # mostly model name and version -> labse
        embed_strategy: str = field(default_factory=str)  # e.g. "flatten"
        
        # CopiedVectors
        copy_relationship: str = field(default_factory=str)  # e.g. "domain"
        copy_relationship_index: str = field(default_factory=str)  # e.g. "Class"
        
        def __post_init__(self):
            # Name without copy relationship or index
            object.__setattr__(self, 'name', f"{self.vectorizer}___default___{self.field_name}___{self.language}")

        def __hash__(self):
            return hash(self.name)

        def __eq__(self, other):
            if isinstance(other, VirtualNamedVectorEmbedding):
                return self.name == other.name
            return False

    @dataclass
    class VirtualNamedVectorEmbedding_Copy(VirtualNamedVectorEmbedding):
        def __hash__(self):
            return hash(self.name)

        def __eq__(self, other):
            if isinstance(other, VirtualNamedVectorEmbedding):
                return self.name == other.name
            return False

    def create_NVs(item):
        
        field_name, language, vectorizer, embed_strategy, copy_relationship, copy_relationship_index = item

        # Create the original embedding instance
        original_embedding = VirtualNamedVectorEmbedding(
            field_name=field_name,
            language=language,
            vectorizer=vectorizer,
            embed_strategy=embed_strategy,
            copy_relationship=copy_relationship,
            copy_relationship_index=copy_relationship_index
        )

        # List to store the instances
        embeddings = [original_embedding]
        
        for n in range(1, 4):  
            copy_embedding = VirtualNamedVectorEmbedding_Copy(
                field_name=original_embedding.field_name,
                language=original_embedding.language,
                vectorizer=original_embedding.vectorizer,
                embed_strategy=original_embedding.embed_strategy,
                copy_relationship=original_embedding.copy_relationship,
                copy_relationship_index=original_embedding.copy_relationship_index
            )
            # Set the name format according to the requirement
            copy_embedding.name = f"{original_embedding.name}___CP_SEPARATOR___{copy_relationship}___{copy_relationship_index}___{n}"
            embeddings.append(copy_embedding)

        return embeddings
    
    all_combos = []
    # Example usage
    for field_name in ["Label", "Description"]:
        for lang in languages:
            for model in models:
                
                for copy_relationship, collection in [("Domain", "Classes"), ("Range", "Classes")]:
                    if model == "LaBSE" and lang == "en":
                        all_combos.append([field_name, lang, model, "default", copy_relationship, collection])
                    
    all_named_vectors = []
    for item in all_combos:
        named_vectors = create_NVs(item)
        
        for n in named_vectors:
            all_named_vectors.append(n)

    for e in sorted(set(all_named_vectors), key=lambda x: x.name):
        print(e.name)
    all_named_vectors = set(all_named_vectors)
    
    formatted_objects_for_upload = []
    if create_new:
        # Formatting of the ontology data to upload to the collection
        print("Generating embeddings and formatting data")
        for i, result_doc in enumerate(endpoint_query_results):
            tp = len(endpoint_query_results) / 10
            
            if i % int(tp) == 0:
                print(i, "/", len(endpoint_query_results))
                
            formatted_object = {}

            uuid = generate_uuid5(result_doc.termIRI)
            formatted_object["TermIRI"] = result_doc.termIRI
            formatted_object["RDF_type"] = result_doc.rdfType
            formatted_object["Ontology"] = result_doc.ontology
            formatted_object["Label"] = result_doc.label
            formatted_object["Description"] = result_doc.description
            formatted_object["Domain"] = result_doc.domain
            formatted_object["Range"] = result_doc.range
            formatted_object["Language"] = result_doc.language

            
            # FILL NON-COPY NAMED VECTORS
            embeddings = {}
            for vector in all_named_vectors:
                name = vector.name
                if not "___CP_SEPARATOR___" in name:
                    model = vector.vectorizer
                    field_name = vector.field_name
                    case_lang = vector.language
                    if case_lang == result_doc.language:
                        
                        if not name in embeddings:
                            embeddings[name] = []
                        
                        embeddings[name] = methodologies[field_name](result_doc, models[model])
            
            for name in embeddings:
                print("Filled", name)
            formatted_objects_for_upload.append([formatted_object, embeddings, uuid])

    # Configurations for custom vectorizers (one for every case)
    vectorizer_config = [wvc.config.Configure.NamedVectors.none(name=x.name) for x in all_named_vectors]
    for v in vectorizer_config:
        print("Added", v.name, "to config")
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
                    wvc.config.Property(name="TermIRI", data_type=wvc.config.DataType.TEXT),
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
    
    # # Upload the formatted_object data
    print("Uploading data")
    objects_to_upload = []
    for d in formatted_objects_for_upload:
        objects_to_upload.append(wvc.data.DataObject(
        properties=d[0],
        vector=d[1],
        uuid=d[2]
        ))
        
    for f in formatted_objects_for_upload:
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
    endpoint_query_results = fetch_data_from_endpoint(url_endpoint, type="data_properties")

    # Names of models to test
    models = ["LaBSE","all-MiniLM-L6-v2","all-MiniLM-L12-v2","all-distilroberta-v1","paraphrase-multilingual-MiniLM-L12-v2","multi-qa-mpnet-base-cos-v1"]

    # Mappings between model names (formatted_object to _ format) and model instances
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

    formatted_objects_for_upload = []
    if create_new:
        # Formatting of the ontology data to upload to the collection
        print("Generating embeddings and formatting data")
        for i, result_doc in enumerate(endpoint_query_results):
            tp = len(endpoint_query_results) / 10
            
            if i % int(tp) == 0:
                print(i, "/", len(endpoint_query_results))
                
            formatted_object = {}

            uuid = generate_uuid5(result_doc.termIRI)
            
            formatted_object["TermIRI"] = result_doc.termIRI
            formatted_object["RDF_type"] = result_doc.rdfType
            formatted_object["Ontology"] = result_doc.ontology
            formatted_object["Label"] = result_doc.label
            formatted_object["Description"] = result_doc.description
            formatted_object["Domain"] = result_doc.domain
            formatted_object["Range"] = result_doc.range
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
                    wvc.config.Property(name="TermIRI", data_type=wvc.config.DataType.TEXT),
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
    
    # # Upload the formatted_object data
    print("Uploading data")
    objects_to_upload = []
    for d in formatted_objects_for_upload:
        objects_to_upload.append(wvc.data.DataObject(
        properties=d[0],
        vector=d[1], 
        uuid=d[2]
        ))
        
    for f in formatted_objects_for_upload:
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
    endpoint_query_results = fetch_data_from_endpoint(url_endpoint, type="RDF_types")

    # Names of models to test
    models = ["LaBSE","all-MiniLM-L6-v2","all-MiniLM-L12-v2","all-distilroberta-v1","paraphrase-multilingual-MiniLM-L12-v2","multi-qa-mpnet-base-cos-v1"]

    # Mappings between model names (formatted_object to _ format) and model instances
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

    formatted_objects_for_upload = []
    if create_new:
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
                    wvc.config.Property(name="TermIRI", data_type=wvc.config.DataType.TEXT),
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

def ontology_collection_creation():
    # Get ontology property data from endpoint
    print("Loading data")
    endpoint_query_results = fetch_data_from_endpoint(url_endpoint, type="Ontologies")

    # Names of models to test
    models = ["LaBSE", "all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "all-distilroberta-v1", "paraphrase-multilingual-MiniLM-L12-v2", "multi-qa-mpnet-base-cos-v1"]

    # Mappings between model names (formatted_object to _ format) and model instances
    models = {x.replace("-", "_"): SentenceTransformerEmbeddings(model_name=x) for x in models}

    # Mappings between mapping methodology names and functions
    methodologies = {"OntologySpecial": embed_ontology_classes, "OntologyDataproperties": embed_ontology_dataproperties}  # Add your embedding methodologies here

    # Languages to consider
    languages = ["en", "fr", "None"]

    # All combinations between models, methodologies and languages
    all_combos = []
    for model in models:
        for method in methodologies:
            for lang in languages:
                all_combos.append({"case_name": f"{model}___{method}___{lang}", "model": model, "method": method, "lang": lang})

    print([x["case_name"] for x in all_combos])

    formatted_objects_for_upload = []
    if create_new:
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

    if create_new:
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
    else:
        collection = client.collections.get(name="Ontologies")

    # Upload the formatted_object data
    print("Uploading data")
    objects_to_upload = []
    for d in formatted_objects_for_upload:
        objects_to_upload.append(wvc.data.DataObject(
            properties=d[0],
            vector=d[1],
            uuid=d[2]
        ))

    if create_new:
        batches = split_list(objects_to_upload, 4)

        try:
            for i, batch in enumerate(batches):
                print(f"Uploading batch {i+1}")
                collection.data.insert_many(batch)

        except Exception as e:
            print(f"Error during insert_many: {e}",  traceback.format_exc())

            
# Create a client instance
headers = {
    "X-HuggingFace-Api-Key": hf_key,
}
client = weaviate.connect_to_local(
    
    port=local_weaviate_port,
    grpc_port=local_weaviate_port_grpc,
    headers=headers
    
)

create_new = True

if create_new:
    try:
        object_property_collection_creation()
    except Exception as e:
        print("Error rdftype", e, traceback.format_exc())
    
    # try:
    #     print("Creating class collection")
    #     class_collection_creation()
    # except Exception as e:
    #     print("Error class", e, traceback.format_exc())
    # try:
    #     print("Creating data_property collection")
    #     data_property_collection_creation()
    # except Exception as e:
    #     print("Error data", e, traceback.format_exc())
    # try:
    #     print("Creating individual collection")
    #     individual_collection_creation()
    # except Exception as e:
    #     print("Error individual", e, traceback.format_exc())
    # try:
    #     print("Creating object property collection")
    #     object_property_collection_creation()
    # except Exception as e:
    #     print("Error object property", e, traceback.format_exc())
        

#class_collection_creation_hf_integration()
#ontology_collection_creation()
#object_property_collection_creation()
