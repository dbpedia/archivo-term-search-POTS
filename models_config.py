# Default model schemas:

# WEAVIATE INTEGRATION 
# {
#   "integration-type": "weaviate",
#   "vectorized-function-name": "Configure.NamedVectors.text2vec_ollama",    # THe name of the function that will be used to vectorize the text (any available from Configure.NamedVectors)
#   "api-model-name": "snow-aea 2020 40 supce col&%/&()l",                   # The name of the weaviate-compatible model to use
#   "internal-model-name": "snow-aea",                                       # The name to use internally
#   "vectorizer-parameters": {
#     "api_endpoint": "http://mygpu-server:11434",                           # The endpoint of the inference engine server
#     "model": "snowflake-arctic-embed"                                      # The name of the model in the server
#   }


# EMBEDDED INTEGRATION
# { 
#   "integration-type": "embedded",
#   "internal-model-name": "snow-aea",                                       # The name to use internally
#   "api-model-name": "snow-aea 2020 40 supce col&%/&()l",                   # The name of the weaviate-compatible model to use
#   "vectorizer-parameters": {
#     "model_name": "sentence-transformers/all-mpnet-base-v2"                # The name of the model to use
#   }

model_1 = {
    "integration-type": "weaviate",
    "vectorized-function-name": "Configure.NamedVectors.text2vec_ollama",
    "api-model-name": "snow-aea 2020 40 supce col&%/&()l",  # needs to be weaviate compatible
    "internal-model-name": "snow-aea",
    "vectorizer-parameters": {
        "api_endpoint": "http://mygpu-server:11434",
        "model": "snowflake-arctic-embed"
    }
}

model_2 = {
    "integration-type": "embedded",
    "internal-model-name": "snow-aea",
    "api-model-name": "snow-aea 2020 40 supce col&%/&()l",  # needs to be weaviate compatible
    "vectorizer-parameters": {
        "model_name": "sentence-transformers/all-mpnet-base-v2"
    }
}


MODEL_CONFIGS = [model_1, model_2]