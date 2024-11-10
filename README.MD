# Ontology Terms MetaV+ Search

Ontology Terms MetaV+ Search is an advanced ontology and terminology search tool that allows filtering for terms and ontologies based on semantic search (via vector embeddings) but also classic string matching filters. The filter can be applied for individual metadata properties of the terms (e.g. label, description, domain/range of an). This enables more precise and context-aware search results, enhancing the discovery and utilization of ontological data especially for LLM-driven use-cases.

## Search API

The heart of this project is an HTTP API which allows to query and filter for term types and ontologies.

### HTTP API endpoint and paths

By default the search API is exposed on `127.0.0.1:9090` but can be configured using `SEARCH_API_EXPOSED_PORT`
The API is expecting a JSON query object in the body / payload of the request.

* `/search` allows to search and filter for all term types and the ontologies

### General Query JSON Object

The base query consists of (3 different types of) JSON elements describing the filters to be applied in the query and an optional `result_limit` field. 
There are 2 types of filters that can be defined `exact_filters` and `fuzzy_filters`. `exact_filters` describe filters that are supposed to be applied based on the exact string sequence given.
`fuzzy_filters` are intended to define some approximate/fuzzy search filter (e.g. vector-based nearest neighbor search or BM-25 on trigrams) based on the given string.
Altough the query filters are optional at least one of the filter elements has to be defined (non-empty) in order to  form a meaningful query and not let web-bots allow to trigger "blank" search requests. 

Depending on the combinations of filters defined, different kinds of query types will be executed in the (Weaviate) backend.

* only `exact_filters` set => classic string-matching ("database style") search filters (at the moment classical exact match only)
* `fuzzy_filters` is set => near_vector or hybrid search (with optional string-matching filtering applied)
* `fuzzy_filters` and `exact_filters` empty => invalid search

```json
{
  "fuzzy_filters": {  // OPTIONAL fuzzy filters (vector-embedding-based search)
      "label": "Termlabel", // OPTIONAL filter by label
      "description": "word or sentence describing the term or its intended usage", // OPTIONAL
      "ontology": "word or sentence describing the ontology the term is originated from", // OPTIONAL
      // .. more metadata fields depending on the term type 
  },
  "fuzzy_filter_configs": { // only REQUIRED if fuzzy_filters is used
      "model_name": "name_of_embedding_model_to_use", // REQUIRED IF FUZZY
      "lang": "en", // OPTIONAL (default is en)
      "hybrid_search_field": "label" // OPTIONAL (default is null meaning no hybrid_search)
  },
  "exact_filters": { // OPTIONAL
      "term_type":  "", 
      "rdf_type": "http://www.w3.org/2002/07/owl#Class", //OPTIONAL rdf:type 
      "ontology": "http://myOntologyNamespace.com/ThatIsUsedAsPrefix" // OPTIONAL
      // .. more metadata fields depending on the term type 
  },
  "result_limit": 10 // OPTIONAL 
}
```

#### fuzzy_filters

## Setup / Deployment

### requirements

1. Docker Compose (version 1.28 or higher required due to usage of compose profiles)
2. Ontology (meta)data you want to search over: either
   * a) SPARQL endpoint with ontology data loaded (recommended is that each owl:Ontology is stored in its dedicated named graph for clean results)
   * b) a Databus Collection that contains your desired ontologies and can be automatically loaded into a containerized SPARQL endpoint with the SPARQL-quickstart Dropin. A list of useful example collections that select subsets from the DBpedia Archivo Ontology Archivo (containing more than 1800 ontologies) is given in the .env file
   * c) a set of files containing ontology data in RDF format (besides JSON-LD) (for clean results one ontology per file with a separate <filen-name>.graph file containing a unique named-graph identifier for that ontology)

### configure .env file

decide which .yml dropins are to be used and define them in the `COMPOSE_FILE` variable

### run setup stages with compose

`cd docker-compose-setup`

1. Make your infrastructure ready for indexing - these commands may not be required for any setup (but they can be run anyhow since they will have no effect)
   1. To download the model data of containerized model dropins or to download the ontology data for SPARQL-quickstart run `EXIT_AFTER_DOWNLOAD=1 docker compose --profile downloading up`
   2. Prepare required services `docker compose --profile prepare-indexing up --abort-on-container-exit` (currently only required to setup Virtuoso SPARQL (SPARQL-quickstart) by ingesting the (downloaded) ontologies from `ontology-data` folder into the SPARQL endpoint
2. Start the indexing `docker compose --profile indexing up --exit-code-from indexer`
3. Serve the API `docker compose --profile serve-api up`

### data persistence and re-running setup for new data

* **containerized models**: data is saved in anonymous volumes, these will be re-used on model container startup, to download the model data again delete the volume(s) of the model containers and rerun the downloading phase
* **SPARQL-quickstart**:
  * downloaded ontologies are saved in`ontology-data` folder, re-running download phase will **add** all new ontologies / files to`ontology-data`and **overwrite** existing ones, no files will be deleted
  * Virtuoso SPARQL endpoint uses the `virtuoso` folder, re-running *prepare-indexing* phase will **add** all new ontologies from `ontology-data` folder to the endpoint and also add the content of all it, no quads/triples will be deleted, and files that were loaded in a previous run wont be re-loaded even if they changed.
* **weaviate-index:** currently there is no support to update or ingest new data into weaviate, delete the `weavia-data` folder or run with flag `DELETE_OLD_INDEX=FALSE` to automatically delete an existing index on startup

## Tweaking of the search and indexing behaviour

* `EMPTY_ARRAY_PROPERTY_SLOT_FILLING_STRATEGY`

  * `empty` : empty slots are filled with the embedding of an empty string
  * `average`: empty slots are filled with the average of the existing embeddings of the defined slots for that property.
  * Explanation: Weaviate can not have individual vectors for its properties that are of array type. This option controls our workaround to "flatten" an array into a fixed set of slots per array. E.g. rdfs:subClassOf can have multiple values. We support 3 slots - so 3 values. Every slot will have its own named vector. When we want to perform a vector search on the rdfs:subClassOf metadata we need to perform 3 queries - one query for every slot. If a term has less then 3 values we need to fill those unoccupied slots with a vector.
* `UNKNOWN_LABEL_REPLACEMENT_EMBEDDING_STRATEGY`- if there is no label information for a given language

  * `empty` : the empty string will be used instead
  * `IRI-suffix`: the suffix of the IRI will be used as label for that language
