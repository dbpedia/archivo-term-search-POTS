# Ontology Terms MetaV+ Search

Ontology Terms MetaV+ Search is an advanced ontology and terminology search tool that allows filtering for terms and ontologies based on semantic search (via vector embeddings) but also classic filters. The filter can be applied for individual metadata properties of the terms (e.g. label, description, domain/range). This enables more precise and context-aware search results, enhancing the discovery and utilization of ontological data especially for LLM-driven use-cases.

## Setup / Deployment

### requirements

1. Docker Compose (version 1.28 or higher required due to usage of compose profiles)
2. ontology data you want to search over: either
   * a) SPARQL endpoint with ontology data loaded (recommended is that each owl:Ontology is stored in its dedicated named graph for clean results)
   * b) a Databus Collection that contains your desired ontologies and can be automatically loaded into a containerized SPARQL endpoint with the QuickStartDropin. A list of useful example collection that select subsets from DBpedia Archivo Ontology Archivo (containing more than 1800 ontologies) is given in the compose-config.env file
   * c) a set of files containing ontology data in RDF format (besides JSON-LD) (for clean results one ontology per file with a separate <filen-name>.graph file containing a unique named-graph identifier for that ontology)


### configure .env file

decide which .yml dropins are to be used and add them to `COMPOSE_FILE` variable

### run setup stages with compose

`cd docker-compose-setup`

1. Make your infrastructure ready for indexing - these commands may not be required for any setup (but they can be run anyhow since they will have no effect)
   1. To download the model data of containerized model dropins or to download the ontology data for SPARQL-quickstart run `EXIT_AFTER_DOWNLOAD=1 docker compose --profile downloading up`
   2. Prepare required services `docker compose --profile prepare-indexing up --abort-on-container-exit` (currently only required to setup Virtuoso SPARQL (SPARQL-quickstart) by ingesting the (downloaded) ontologies from `ontology-data`folder into the SPARQL endpoint
2. Start the indexing `docker compose --profile indexing up --exit-code-from indexer`
3. Serve the API `docker compose --profile serve-api up`

### data persistence and re-running setup for new data

* **containerized models**: data is saved in anonymous volumes, these will be re-used on model container startup, to download the model data again delete the volume(s) of the model containers and rerun the downloading phase
* **SPARQL-quickstart**:
  * downloaded ontologies are saved in`ontology-data` folder, re-running download phase will **add** all new ontologies / files to`ontology-data`and **overwrite** existing ones, no files will be deleted
  * Virtuoso SPARQL endpoint uses the `virtuoso` folder, re-running *prepare-indexing* phase will **add** all ontologies from `ontology-data` folder to the endpoint and also add all new triples to it, no quads/triples will be deleted
* **weaviate-index:** currently there is no support to update or ingest new data into weaviate, delete the `weavia-data` folder or run with flag `DELETE_OLD_INDEX=FALSE` to autmatically delete an existing index on startup
