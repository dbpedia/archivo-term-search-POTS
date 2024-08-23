import weaviate
import weaviate.classes as wvc

from langchain_community.embeddings import SentenceTransformerEmbeddings
from weaviate_test_aux_functions import *
from dotenv import load_dotenv
import os
load_dotenv()
wcd_url = os.getenv("WCD_URL")
wcd_api_key = os.getenv("WCD_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
url_endpoint = "http://95.217.207.179:8995/sparql/"
print(wcd_url)

# Get ontology property data from endpoint
print("Loading data")
all_data = get_data(url_endpoint, type="properties")

# Create a client instance
print("Connecting to cloud")
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,                                    
    auth_credentials=wvc.init.Auth.api_key(wcd_api_key),    
)

# Names of models to test
models = ["LaBSE","all-MiniLM-L6-v2","all-MiniLM-L12-v2","all-distilroberta-v1","paraphrase-multilingual-MiniLM-L12-v2","multi-qa-mpnet-base-cos-v1"]

# Mappings between model names (formatted to _ format) and model instances
models = {x.replace("-", "_"): SentenceTransformerEmbeddings(model_name=x) for x in models}

# Mappings between mapping methodology names and functions
methodologies = {"Label": embed_using_label, "Description": embed_using_desc, "DomainPLUSRange": embed_using_domain_plus_range, "Domain": embed_using_domain, "Range": embed_using_range}

# Languages to consider
languages = ["en", "fr", "None"]

# All combinations between models, methodologies and languages
all_combos = []
for model in models:
    for method in methodologies:
        for lang in languages:
            all_combos.append({"case_name": f"{model}SEP{method}SEP{lang}","model": model, "method": method, "lang":lang})


# Formatting of the ontology data to upload to the collection
print("Generating embeddings and formatting data")
formatted_for_upload = []
for d in all_data:
    formatted = {}
    term, label, description, domain, rang, language = d
    formatted["Term"] = term
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
            
    formatted_for_upload.append([formatted, embeddings])

# Configurations for custom vectorizers (one for every case)
vectorizer_config = [wvc.config.Configure.NamedVectors.none(name=x["case_name"]) for x in all_combos]

# Delete the last iteration of the collection (for testing purposes)
client.collections.delete("Properties")

# Create a new collection with the vectorizer configs
print("Creating collection")
collection = client.collections.create(
        name="Properties",
        description="Text2kg benchmark properties",
        
        vectorizer_config=vectorizer_config,

        # I believe that defining those "properties" is only needed in case we use Weviate-hosted models, to determine what they should embed for each named vector
        # But I defined them anyway
        properties=[
        wvc.config.Property(name="Term", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="Label", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="Description", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="Domain", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="Range", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="Language", data_type=wvc.config.DataType.TEXT),

    ],
    )

# Upload the formatted data
print("Uploading data")
objects_to_upload = []
for d in formatted_for_upload:
    objects_to_upload.append(wvc.data.DataObject(
    properties=d[0],
    vector=d[1]
    ))
collection.data.insert_many(objects_to_upload)

# Test different simple queries
print("Querying")
queries = ["Agent", "Language", "Instruction", "Rights"]
for q in queries:
    for i in range(5):
        print()
        print("Looking for", q,"in", all_combos[i]["case_name"])
        
        results = collection.query.near_vector(near_vector=models[all_combos[i]["case_name"].split("SEP")[0].replace("_", "-")].embed_query(q), target_vector=all_combos[i]["case_name"]).objects
        print("Found:")
        for x in results:
            print(x.properties["label"]) 
            
# Close the connection to the client
client.close()

