#!/bin/bash

# Sentiview Deployment Script

set -e

echo "🚀 Starting Sentiview deployment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build and start services
echo "📦 Building Docker images..."
docker-compose build

echo "🔄 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Health check
echo "🏥 Performing health checks..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API service is healthy"
else
    echo "❌ API service health check failed"
    docker-compose logs api
    exit 1
fi

if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Web service is healthy"
else
    echo "❌ Web service health check failed"
    docker-compose logs web
    exit 1
fi

echo "🎉 Deployment successful!"
echo "📱 Frontend: http://localhost:3000"
echo "🔧 API: http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/docs"
