#!/bin/bash
# Login to  AWS ECR

aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 730335254649.dkr.ecr.eu-north-1.amazonaws.com
docker pull 730335254649.dkr.ecr.eu-north-1.amazonaws.com/docker_image_ecr2ec2:latest
docker stop my-app || true
docker rm my-app || true
docker run -d -p 8000:8000 -e DAGSHUB_PAT=c6ea9f2beb523e479f6f8150da41c0535c1f50d1 --name my-app 730335254649.dkr.ecr.eu-north-1.amazonaws.com/docker_image_ecr2ec2:latest
