#!/bin/bash

# === 설정 ===
USERNAME=your-dockerhub-username     # ⚠️ 본인 Docker Hub ID로 수정
IMAGE_NAME=embedding-server
TAG=latest

# === 빌드 ===
echo "🔨 Building Docker image..."
docker build -t $IMAGE_NAME .

# === 태그 ===
echo "🏷️ Tagging image as $USERNAME/$IMAGE_NAME:$TAG"
docker tag $IMAGE_NAME $USERNAME/$IMAGE_NAME:$TAG

# === 푸시 ===
echo "📤 Pushing image to Docker Hub..."
docker push $USERNAME/$IMAGE_NAME:$TAG

echo "✅ Done!"
