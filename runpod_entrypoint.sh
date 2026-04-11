#!/bin/bash
set -e

# Default to port 6006 if not set
TENSORBOARD_PORT=${TENSORBOARD_PORT:-6006}

# --- Determine storage paths based on environment ---
# In TEST mode (local Mac via Lima/VirtioFS), all writes go to ephemeral /tmp
# to bypass VirtioFS file-locking which breaks SQLite, TensorBoard, and torch.save.
# /app/models (host-mounted read-only) is still used to SEED the starting checkpoint.
# In PROD mode (RunPod), all writes go to the persistent mounted volumes.
if [ "$ENV_TYPE" = "test" ]; then
    echo "Running in TEST mode: writes go to ephemeral /tmp to bypass VirtioFS locking."
    export BASE_RUNS_DIR="/tmp/runs"
    export BASE_MODELS_DIR="/tmp/models"
    # Source of truth for initial checkpoints is still the read-only host mount
    CHECKPOINT_SOURCE_DIR="/app/models"
else
    echo "Running in PROD mode: using persistent /app storage."
    export BASE_RUNS_DIR="${BASE_RUNS_DIR:-/app/runs}"
    export BASE_MODELS_DIR="${BASE_MODELS_DIR:-/app/models}"
    CHECKPOINT_SOURCE_DIR="${BASE_MODELS_DIR}"
fi

mkdir -p "$BASE_RUNS_DIR"
mkdir -p "$BASE_MODELS_DIR"

# --- Handle Persistent Configuration ---
# Find the baked-in config file and link it to the persistent volume
CONFIG_PATH=$(find /app -maxdepth 2 -name "uttt_config.json" | head -n 1)
if [ -n "$CONFIG_PATH" ]; then
    if [ ! -f "$BASE_RUNS_DIR/uttt_config.json" ]; then
        echo "Copying default uttt_config.json to persistent storage ($BASE_RUNS_DIR)..."
        cp "$CONFIG_PATH" "$BASE_RUNS_DIR/uttt_config.json"
    fi
    echo "Symlinking persistent uttt_config.json to $CONFIG_PATH..."
    rm -f "$CONFIG_PATH"
    ln -s "$BASE_RUNS_DIR/uttt_config.json" "$CONFIG_PATH"
fi

# --- Start SSH Daemon ---
if command -v sshd > /dev/null; then
    echo "Starting SSH daemon..."
    mkdir -p /var/run/sshd
    
    # RunPod automatically injects your SSH key via the PUBLIC_KEY environment variable
    if [ -n "$PUBLIC_KEY" ]; then
        echo "Installing SSH key from RunPod..."
        mkdir -p /root/.ssh
        chmod 700 /root/.ssh
        echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
    fi

    service ssh start
fi

echo "Starting Tensorboard in the background on port $TENSORBOARD_PORT..."
tensorboard --logdir=$BASE_RUNS_DIR --host=0.0.0.0 --port=$TENSORBOARD_PORT &

# --- Find starting checkpoint ---
# In test mode, look in the ephemeral dir first (in case a previous test wrote one),
# then fall back to the read-only host source for bootstrapping.
LATEST_CHECKPOINT=$(ls -t $BASE_MODELS_DIR/*/checkpoint.pt 2>/dev/null | head -n 1)

if [ -z "$LATEST_CHECKPOINT" ] && [ "$ENV_TYPE" = "test" ]; then
    # Try seeding from the read-only host mount
    LATEST_CHECKPOINT=$(ls -t $CHECKPOINT_SOURCE_DIR/*/checkpoint.pt 2>/dev/null | head -n 1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo "TEST MODE: Seeding from host checkpoint at $LATEST_CHECKPOINT (read-only, writes go to /tmp)."
    fi
fi

START_CHECKPOINT="$BASE_MODELS_DIR/checkpoint.pt"
if [ -z "$START_CHECKPOINT_FILE_EXISTS" ] && [ "$ENV_TYPE" = "test" ] && [ ! -f "$START_CHECKPOINT" ]; then
    # Also fall back to host-mounted base checkpoint for seeding
    if [ -f "$CHECKPOINT_SOURCE_DIR/checkpoint.pt" ]; then
        START_CHECKPOINT="$CHECKPOINT_SOURCE_DIR/checkpoint.pt"
    fi
fi

RESUME_ARGS=""
MODEL_PATH=""
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "Found checkpoint at $LATEST_CHECKPOINT"
    RESUME_ARGS="--resume_from $LATEST_CHECKPOINT"
    MODEL_PATH="$LATEST_CHECKPOINT"
elif [ -f "$START_CHECKPOINT" ]; then
    echo "No run checkpoint found, using starting checkpoint at $START_CHECKPOINT."
    RESUME_ARGS="--resume_from $START_CHECKPOINT"
    MODEL_PATH="$START_CHECKPOINT"
else
    echo "No checkpoints found. Starting fresh."
fi

# --- Launch the selected mode ---
if [ "$RUN_MODE" = "optuna" ]; then
    if [ -z "$MODEL_PATH" ]; then
        echo "Error: Optuna mode requires a valid model checkpoint to optimize!"
        exit 1
    fi

    OPTUNA_DB="$BASE_RUNS_DIR/optuna.db"
    # Launch the dashboard only after opt_selfplay.py has created and initialized
    # the DB (with its schema tables). Polling avoids the 'no such table: version_info'
    # crash that occurs when the dashboard opens a brand-new empty file.
    (
        echo "Waiting for Optuna DB to be initialized before starting dashboard..."
        while [ ! -f "$OPTUNA_DB" ]; do sleep 1; done
        # Give Optuna one extra second to finish writing the schema
        sleep 2
        echo "Starting Optuna Dashboard in the background on port 8080..."
        optuna-dashboard sqlite:///$OPTUNA_DB --host 0.0.0.0 --port 8080
    ) &

    OPTUNA_MODE=${OPTUNA_MODE:-train}
    echo "Starting Optuna Hyperparameter Optimization in mode: $OPTUNA_MODE..."
    python -u -m train.opt_selfplay --model_path "$MODEL_PATH" --game uttt --mode $OPTUNA_MODE --n_trials 50 2>&1 | tee -a $BASE_RUNS_DIR/console_output.log
else
    echo "Starting training..."
    python -u -m train.main --game uttt $RESUME_ARGS 2>&1 | tee -a $BASE_RUNS_DIR/console_output.log
fi
