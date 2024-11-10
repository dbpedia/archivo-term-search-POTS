#! /usr/bin/env sh
if [ "${PWD##*/}" != "docker-compose-setup" ]; then
    cd docker-compose-setup
fi
# ask user whether to create a name for the backup folder
echo "Remove the current weaviate_data folder by createing a backup folder for it:"
echo "Do you want to give this backup folder a named suffix (date will be used otherwise)? (y/n)"
read create_backup_folder
if [ "$create_backup_folder" = "y" ]; then
    echo "Enter the name of the backup folder"
    read backup_folder_name
else
    backup_folder_name=$(date +"%Y-%m-%d-%H:%M:%S")
fi

# create a backup folder with the name suffix by the user or the date if no name is provided
mkdir -p old_weaviate_data/
docker compose --profile prepare-indexing --profile indexing stop
mv weaviate_data old_weaviate_data/weaviate_data-$backup_folder_name
# start compose pipeline
rm -rf virtuoso && docker compose --profile prepare-indexing up -d && docker compose logs -f sparql-store-loader && docker compose --profile indexing --profile serve-api up -d && docker compose logs -f indexer 
