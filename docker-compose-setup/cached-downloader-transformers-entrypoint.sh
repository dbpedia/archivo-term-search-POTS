#!/bin/sh
# script to download the model (but only if the model is not already downloaded) and start transformers inference api 
# to be used as entrypoint for the weaviate custom transformers image semitechnologies/transformers-inference:custom-1.9.6
# see https://github.com/weaviate/t2v-transformers-models/tree/d7d8312d5b58d3f7fd5220485961d8ca746c2d57 for download.py and Dockerfile

echo "Starting transformer container with model to use: $MODEL_NAME" &&
if [ ! -d /app/models ] || [ -z "$(ls -A /app/models)" ]; then
  echo "Starting download script ..." &&
  ./download.py
else
  echo "Using model already downloaded and persisted in volume" &&
  echo "Last modification date $(stat -c %y /app/models) and total size of the folder  $(du -sh /app/models | cut -f1)"
  [ -f /app/models/model/config.json ] && cat /app/models/model/config.json || echo "no config to display"
fi &&
uvicorn app:app --host 0.0.0.0 --port 8080