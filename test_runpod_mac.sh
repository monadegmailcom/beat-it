#!/bin/bash
set -e

echo "Building local CPU Docker image for testing..."
docker build -t beat-it-runpod:cpu .

echo ""
echo "=========================================================="
echo " Starting Docker Container Simulator... "
echo " This maps your local 'runs' and 'models' dir to the container."
echo " Tensorboard will be accessible at http://localhost:6006"
echo " Press Ctrl+C to simulate a random RunPod interrupt."
echo " Restarting the script will resume from the last checkpoint."
echo "=========================================================="

# Run container mapping ports and volumes
docker run -it --rm \
    -e TENSORBOARD_PORT=6006 \
    -p 6006:6006 \
    beat-it-runpod:cpu
