# Ontology Terms MetaV+ Search

Ontology Terms MetaV+ Search is an advanced ontology and terminology search tool that allows filtering for terms and ontologies based on semantic search (via vector embeddings) but also classic filters. The filter can be applied for individual metadata properties of the terms (e.g. label, description, domain/range). This enables more precise and context-aware search results, enhancing the discovery and utilization of ontological data especially for LLM-driven use-cases.

## Setup / Deployment
we recommend to setup via Docker Compose (version 1.28 or higher required due to usage of profiles).
`cd docker-compose-setup`

### configure .env file
decide which dropins are to be used and add them to `COMPOSE_FILE` variable

### run setup stages
docker compose --profile downloading up # download data required for subsequent processes (e.g. ontologies or locally hosted models)
docker compose --profile prepare-indexing up --abort-on-container-exit # currently only required for automatic sparql-endpoint setup
docker compose --profile indexing up --exit-code-from indexer
docker compose --profile serve-api up 

