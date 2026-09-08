#!/usr/bin/env bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="procedure-algorithm"
MISSING_MODEL=0

for model_path in \
  "${SCRIPT_DIR}/resources/w64/model.safetensors" \
  "${SCRIPT_DIR}/resources/nw2/model.safetensors"
do
  if [ ! -f "$model_path" ]; then
    echo "Missing ${model_path}"
    MISSING_MODEL=1
  fi
done

if [ "$MISSING_MODEL" -ne 0 ]; then
  echo "Run: python scripts/download_weights.py"
  exit 1
fi

docker build \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG"  \
  "$SCRIPT_DIR" 2>&1
