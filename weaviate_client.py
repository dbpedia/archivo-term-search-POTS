import os
import weaviate
from weaviate.auth import Auth
from weaviate.classes.init import AdditionalConfig, Timeout
from dotenv import load_dotenv
import time
import logging
import contextlib

load_dotenv()

@contextlib.contextmanager
def get_weaviate_client():
    weaviate_api_uri = os.getenv("WEAVIATE_API_URI")
    weaviate_grpc_uri = os.getenv("WEAVIATE_GRPC_URI")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    hf_key = os.getenv("HF_KEY")

    http_host, http_port, http_secure = parse_uri(weaviate_api_uri)
    grpc_host, grpc_port, grpc_secure = parse_uri(weaviate_grpc_uri)

    auth_credentials = Auth.api_key(weaviate_api_key) if weaviate_api_key else None
    # todo add other headers and think whether that makes sense here to pass the hf_key
    # because that would be job of the api-modules in weaviate
    headers = {"X-HuggingFace-Api-Key": hf_key} if hf_key else None

    # todo extend this with some backoff interval algorithm (try/catch) to reconnect with increased timeout for n iterations. also log each attempt to connect with the parameters
    logging.basicConfig(level=logging.INFO) # todo needs to be configured from the outside
    logging.info(f"Attempting to connect to Weaviate with the following parameters: http_host={http_host}, http_port={http_port}, http_secure={http_secure}, grpc_host={grpc_host}, grpc_port={grpc_port}, grpc_secure={grpc_secure}, auth_credentials={auth_credentials}, headers={headers}")

    client = weaviate.connect_to_custom(
        http_host=http_host,
        http_port=http_port,
        http_secure=http_secure,
        grpc_host=grpc_host,
        grpc_port=grpc_port,
        grpc_secure=grpc_secure,
        auth_credentials=auth_credentials,
        headers=headers,
        additional_config=AdditionalConfig(
        timeout=Timeout(init=30, query=60, insert=120)  # Values in seconds
    )
    )
    try:
        yield client
    finally:
        client.close()

def parse_uri(uri):
    if uri.startswith("https://"):
        secure = True
        uri = uri[8:]
    elif uri.startswith("http://"):
        secure = False
        uri = uri[7:]
    else:
        secure = False

    if ':' in uri:
        host, port = uri.split(':')
        port = int(port)
    else:
        host = uri
        port = 443 if secure else 80
    return host, port, secure
