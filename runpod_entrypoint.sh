#!/bin/bash
set -e

# Default to port 6006 if not set
TENSORBOARD_PORT=${TENSORBOARD_PORT:-6006}

echo "Starting Tensorboard in the background on port $TENSORBOARD_PORT..."
tensorboard --logdir=/app/runs --host=0.0.0.0 --port=$TENSORBOARD_PORT &

# We check if there's a recent checkpoint in subdirectories, 
# or if not, use the base starting checkpoint.
LATEST_CHECKPOINT=$(ls -t /app/models/*/checkpoint.pt 2>/dev/null | head -n 1)
START_CHECKPOINT="/app/models/checkpoint.pt"

RESUME_ARGS=""
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "Found recent checkpoint at $LATEST_CHECKPOINT, resuming from it..."
    RESUME_ARGS="--resume_from $LATEST_CHECKPOINT"
elif [ -f "$START_CHECKPOINT" ]; then
    echo "No recent run checkpoint found, but found starting checkpoint at $START_CHECKPOINT."
    RESUME_ARGS="--resume_from $START_CHECKPOINT"
else
    echo "No checkpoints found. Starting fresh."
fi

# Make sure we use the right Python environment if needed. 
# Dockerfile already sets PATH="/app/.venv/bin:$PATH", so we just call python.
echo "Starting training..."
python -m train.main --game uttt $RESUME_ARGS
