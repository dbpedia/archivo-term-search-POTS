#! /usr/bin/env bash

# Get the project name (defaults to the current directory name if not set)

#!/bin/bash

# Get the project name (defaults to the current directory name if not set)
project_name=$(docker compose config | awk '/^name:/{print $2}')
if [ -z "$project_name" ]; then
  project_name=$(basename "$(pwd)")
fi

# Get all container IDs associated with the project, including stopped ones
docker ps -a --filter "label=com.docker.compose.project=$project_name" -q | while read -r container_id; do
  container_name=$(docker inspect -f '{{ .Name }}' "$container_id" | cut -c2-)
  echo "Container: $container_name"

  # Extract mount information for both named and anonymous volumes
  docker inspect -f '{{ range .Mounts }}{{ .Name | printf "%s" }}|{{ .Source | printf "%s" }}|{{ .Destination | printf "%s" }}{{ "\n" }}{{ end }}' "$container_id" | while IFS='|' read -r volume_name source_path destination_path; do
    # **Skip empty lines**
    if [ -z "$volume_name" ] && [ -z "$source_path" ] && [ -z "$destination_path" ]; then
      continue
    fi

    # Label mounted volumes
    if [ -z "$volume_name" ] || [ "$volume_name" == "<no value>" ]; then
      volume_name="NONE - host mounted directory !"
    fi
    # Calculate size if the source path exists
    if [ -d "$source_path" ]; then
      size=$(du -sm "$source_path" 2>/dev/null | cut -f1)
    else
      size=0
    fi
    echo "    Volume ID: $volume_name"
    echo "    Source Path: $source_path"
    echo "    Destination: $destination_path"
    echo "    Size: ${size} MiB"
    echo "    ----------------------------"
  done
done


# project_name=$(docker compose config | awk '/^name:/{print $2}')
# if [ -z "$project_name" ]; then
#   project_name=$(basename "$(pwd)")
# fi

# # Get all container IDs associated with the project, including stopped ones
# docker ps -a --filter "label=com.docker.compose.project=$project_name" -q | while read -r container_id; do
#   container_name=$(docker inspect -f '{{ .Name }}' "$container_id" | cut -c2-)
#   echo "Container: $container_name"

#   # Extract mount information for both named and anonymous volumes
#   docker inspect -f '{{ range .Mounts }}{{ .Name | printf "%s" }}|{{ .Source | printf "%s" }}|{{ .Destination | printf "%s" }}{{ "\n" }}{{ end }}' "$container_id" | while IFS='|' read -r volume_name source_path destination_path; do
#     # # Label anonymous volumes
#     # if [ -z "$volume_name" ] || [ "$volume_name" == "<no value>" ]; then
#     #   volume_name="anonymous"
#     # fi
#     # Calculate size if the source path exists
#     if [ -d "$source_path" ]; then
#       size=$(du -sm "$source_path" 2>/dev/null | cut -f1)
#     else
#       size=0
#     fi
#     echo "    __________________________"
#     echo "    Volume ID: $volume_name"
#     echo "    Source Path: $source_path"
#     echo "    Destination: $destination_path"
#     echo "    Size: ${size} MiB"
#   done
# done

# only for running containers
# docker compose ps -q | while read container_id; do
#   container_name=$(docker inspect -f '{{ .Name }}' "$container_id" | cut -c2-)
#   echo "Container: $container_name"
#   docker inspect -f '{{ range .Mounts }}{{ if .Name }}{{ .Name }} {{ .Source }}{{ "\n" }}{{ end }}{{ end }}' "$container_id" | while read volume_name source_path; do
#     size=$(du -sm "$source_path" 2>/dev/null | cut -f1)
#     echo "  Volume: $volume_name"
#     echo "  Size: ${size} MiB"
#   done
# done.