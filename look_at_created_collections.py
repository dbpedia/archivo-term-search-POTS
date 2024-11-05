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
# Create a client instance
headers = {
    "X-HuggingFace-Api-Key": hf_key,
}
 
        
client = weaviate.connect_to_local(

    port=8080,
    grpc_port=50051,
    headers=headers

)
for x in client.collections.list_all(simple=True):
    print(client.collections.get(name=x).name, "exists")
    
client.close()