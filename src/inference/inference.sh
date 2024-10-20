#!/bin/bash


export GOOGLE_APPLICATION_CREDENTIALS="/home/wel019/secrets.json"
pipenv run dvc pull --remote fashion_ai_models

# build the docker
docker build -t fashionai_qa -f Dockerfile .

docker run --rm -ti fashionai_qa



