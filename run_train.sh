#!/bin/bash
# ==============================================================================
# MODE: Default Training (Self-Play + Neural Network Optimization)
# ==============================================================================
# This is the standard execution mode. It launches the RunPod simulator and 
# starts generating self-play games, training the neural network, and evaluating 
# the checkpoints.
#
# Environment Variables:
#   RUN_MODE=train (default)
# ==============================================================================

export RUN_MODE="train"
./test_runpod_mac.sh