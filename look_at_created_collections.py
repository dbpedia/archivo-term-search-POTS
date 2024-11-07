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
from weaviate_client import get_weaviate_client
load_dotenv()

# Global variables
wcd_url = os.getenv("WCD_URL")
wcd_api_key = os.getenv("WCD_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
hf_key = os.getenv("HF_KEY")
url_endpoint =  os.getenv("SPARQL_ENDPOINT")
weaviate_port = int(os.getenv("WEAVIATE_PORT"))
weaviate_port_grpc = int(os.getenv("WEAVIATE_PORT_GRPC"))
weaviate_address = os.getenv("WEAVIATE_ADDRESS")
create_new = os.getenv("DELETE_OLD_INDEX")

# Available models
# Create a client instance
headers = {
    "X-HuggingFace-Api-Key": hf_key,
}
 
client = get_weaviate_client()

for collection_name in client.collections.list_all(simple=True):
    collection = client.collections.get(name=collection_name)
    print(f"Objects in collection '{collection.name}':")
    for obj in collection.iterator():
        print(obj.properties)
    print("---")

client.close()