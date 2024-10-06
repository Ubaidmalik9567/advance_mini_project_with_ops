#!/bin/bash

# Login to  AWS ECR

# aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 730335254649.dkr.ecr.eu-north-1.amazonaws.com
# docker pull 730335254649.dkr.ecr.eu-north-1.amazonaws.com/docker_image_ecr2ec2:latest
# docker stop my-app || true
# docker rm my-app || true
# docker run -d -p 8000:8000 -e DAGSHUB_PAT=c6ea9f2beb523e479f6f8150da41c0535c1f50d1 --name my-app 730335254649.dkr.ecr.eu-north-1.amazonaws.com/docker_image_ecr2ec2:latest


# or can you this if above not working

aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 730335254649.dkr.ecr.eu-north-1.amazonaws.com
docker pull 730335254649.dkr.ecr.eu-north-1.amazonaws.com/docker_image_ecr2ec2:latest

# Check if the container 'my-app' is running
if [ "$(docker ps -q -f name=my-app)" ]; then
    # Stop the running container
    docker stop my-app
fi

# Check if the container 'my-app' exists (stopped or running)
if [ "$(docker ps -aq -f name=my-app)" ]; then
    # Remove the container if it exists
    docker rm my-app
fi
docker run -d -p 8000:8000 -e DAGSHUB_PAT=c6ea9f2beb523e479f6f8150da41c0535c1f50d1 --name my-app 730335254649.dkr.ecr.eu-north-1.amazonaws.com/docker_image_ecr2ec2:latest
