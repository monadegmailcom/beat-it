#!/bin/bash
set -e

echo "Building local CPU Docker image for testing..."
docker build -t beat-it-runpod:cpu .

echo ""
echo "=========================================================="
echo " Starting Docker Container Simulator... "
echo " Run mode: ${RUN_MODE:-train}"
echo " Tensorboard will be accessible at http://localhost:6006"
echo " Optuna Dashboard will be accessible at http://localhost:8080"
echo " Press Ctrl+C to simulate a random RunPod interrupt."
echo " Restarting the script will resume from the last checkpoint."
echo "=========================================================="

# Run container mapping ports and volumes
docker run -it --rm \
    -e RUN_MODE=${RUN_MODE:-train} \
    -e OPTUNA_MODE=${OPTUNA_MODE:-train} \
    -e ENV_TYPE=test \
    -e TENSORBOARD_PORT=6006 \
    -p 6006:6006 \
    -p 8080:8080 \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/runs:/app/runs" \
    beat-it-runpod:cpu
