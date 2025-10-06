#!/bin/bash
echo "Starting NexusBattle IA Server..."
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Waiting for dependencies to be ready..."
sleep 2
exec python -m app.server