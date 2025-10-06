# Build con tu Dockerfile actual
docker build -t nexus-ia-server .

# Ejecutar
docker run -d -p 8000:8000 \
  -e INVENTORY_BASE_URL=http://localhost:9000 \
  --name ia-server nexus-ia-server

# Verificar que funciona
curl http://localhost:8000/health