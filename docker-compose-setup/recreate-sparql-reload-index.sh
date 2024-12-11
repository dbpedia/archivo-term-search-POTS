#! /usr/bin/env sh
if [ "${PWD##*/}" != "docker-compose-setup" ]; then
    cd docker-compose-setup
fi
docker compose --profile prepare-indexing stop
rm -rf virtuoso && docker compose --profile prepare-indexing up -d && docker compose logs -f sparql-store-loader && docker compose --profile indexing --profile serve-api up -d && docker compose logs -f indexer 
# docker compose --profile prepare-indexing up -d && docker compose wait sparql-store-loader && docker compose --profile indexing --profile serve-api up -d && docker compose wait indexer && echo "Go !!!!!!!!!!"

