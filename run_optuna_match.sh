#!/bin/bash
# ==============================================================================
# MODE: Optuna Hyperparameter Optimization - Evaluation Throughput (MATCH)
# ==============================================================================
# This mode searches for the best `max_batch_size` (GPU batching) during 
# match evaluations. Since matches are played linearly, `parallel_games` is 
# locked to 1, and `parallel_simulations` is locked to hardware concurrency.
#
# Environment Variables:
#   RUN_MODE=optuna
#   OPTUNA_MODE=match
#
# Output:
#   Optuna Dashboard: http://localhost:8080
# ==============================================================================

export RUN_MODE="optuna"
export OPTUNA_MODE="match"
./test_runpod_mac.sh