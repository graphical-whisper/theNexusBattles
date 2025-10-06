#!/bin/bash

# Build de la imagen
echo "Construyendo imagen Docker..."
docker build -t nexus-ia-server:latest .

# Parar contenedor existente si existe
echo "Deteniendo contenedor existente..."
docker stop ia-server || true
docker rm ia-server || true

# Ejecutar nuevo contenedor
echo "Iniciando nuevo contenedor..."
docker run -d \
  --name ia-server \
  -p 8000:8000 \
  -e INVENTORY_BASE_URL=http://tu-vm-ip:9000 \
  -e INVENTORY_TIMEOUT_SECS=5.0 \
  --restart unless-stopped \
  nexus-ia-server:latest

echo "Despliegue completado. Verifica con: docker logs ia-server"