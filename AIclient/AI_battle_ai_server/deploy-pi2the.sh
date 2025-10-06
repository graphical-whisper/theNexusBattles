#!/bin/bash
set -e

echo "🔨 Construyendo imagen Docker para PI2THE..."
docker build -t pi2the-ia-server:latest .

echo "🛑 Deteniendo contenedor anterior..."
docker stop pi2the-ia-server || true
docker rm pi2the-ia-server || true

echo "🚀 Iniciando nuevo contenedor..."
docker run -d \
  --name pi2the-ia-server \
  -p 8000:8000 \
  -e INVENTORY_BASE_URL=http://localhost:9000 \
  -e INVENTORY_TIMEOUT_SECS=5.0 \
  -e HOST=0.0.0.0 \
  -e PORT=8000 \
  --restart unless-stopped \
  pi2the-ia-server:latest

echo "✅ Despliegue PI2THE completado!"
echo "📊 Ver logs: docker logs -f pi2the-ia-server"
echo "🌐 Health check: curl http://localhost:8000/health"