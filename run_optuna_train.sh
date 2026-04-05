#!/bin/bash
# ==============================================================================
# MODE: Optuna Hyperparameter Optimization - Self-Play Throughput (TRAIN)
# ==============================================================================
# This mode searches for the best `max_batch_size` (GPU batching) and 
# `parallel_games` (MCTS parallel game states) parameters to maximize 
# AlphaZero self-play node generation throughput (positions/second).
#
# Environment Variables:
#   RUN_MODE=optuna
#   OPTUNA_MODE=train
#
# Output:
#   Optuna Dashboard: http://localhost:8080
# ==============================================================================

export RUN_MODE="optuna"
export OPTUNA_MODE="train"
./test_runpod_mac.sh