# Optimizing AlphaZero Self-Play: Throughput vs. Search Quality

## Background & Motivation
In AlphaZero-style self-play (`train` mode), we use MCTS guided by a neural network. To saturate GPU throughput, the engine batches node evaluation requests. When a thread reaches a leaf node, it pushes a request to the inference queue, applies a "virtual loss" to that node to discourage other threads from exploring the exact same path, and immediately starts a new simulation.

Optuna currently optimizes purely for **throughput** (positions per second). It naturally discovers that minimizing OS thread count (`parallel_games`) while maximizing GPU batch size (`max_batch_size`) yields the highest raw speed by minimizing context switching.

## The Problem (Blind Search / Virtual Loss Degradation)
If Optuna chooses `parallel_games = 10` and `max_batch_size = 256`, the engine aggressively polls the queue. While the GPU is processing a batch, those 10 threads will race to fill the next queue of 256. 
This forces each game tree to generate ~25 consecutive simulations **completely blind**, guided *only* by the stacking virtual loss penalty instead of actual neural network evaluations. The MCTS quality degrades significantly, polluting the self-play training data.

## Proposed Solutions

I have researched how state-of-the-art engines like KataGo and Leela Chess Zero handle this. They all face this "Batching Penalty": larger batches increase raw speed but decrease Elo per node. 

Here are three ways we can adapt `opt_selfplay.py` to handle this trade-off:

### 1. Strict Constraint (Forced Quality)
We mathematically constrain the Optuna search space so it cannot choose a configuration that heavily degrades quality.
*   **Implementation:** `parallel_games = trial.suggest_int('parallel_games', max_batch_size, max_batch_size * 4)`
*   **Pros:** Guarantees that on average, no game is forced to contribute more than 1 state per batch. Maximum training data quality.
*   **Cons:** Limits absolute max throughput, especially on CPUs with fewer cores that struggle to context-switch hundreds of threads.

### 2. Manual Pareto Inspection (The Dashboard Approach)
We don't change the Python code at all. We let Optuna search the full, unconstrained space.
*   **Implementation:** No code changes. You open the Optuna Dashboard (`http://localhost:8080`).
*   **How it works:** You inspect the results. You look for trials that achieved *good* throughput but maintained a healthy ratio (where `parallel_games` is close to or larger than `max_batch_size`). This is finding the "Pareto Front" manually—the curve where you can't improve throughput without sacrificing the ratio, and vice versa.
*   **Pros:** Total flexibility. You can see the actual performance numbers before deciding on a compromise.
*   **Cons:** Requires manual analysis of the Optuna database to pick the winning configuration.

### 3. Soft Penalty Function (Multi-Objective Optimization)
We change Optuna to a multi-objective study, or we artificially penalize the throughput score based on the ratio.
*   **Implementation:** We calculate a penalty factor: e.g., `ratio = max_batch_size / parallel_games`. If `ratio > 1`, we multiply the measured throughput by `(1 / ratio)`. Optuna tries to maximize this adjusted "quality-throughput" score.
*   **Pros:** Fully automated. It teaches Optuna to mathematically care about the trade-off.
*   **Cons:** The penalty function is an arbitrary heuristic. It might artificially skew results away from a genuinely fast configuration that only had a slight quality degradation.

## Recommendation
I recommend **Option 1 (Strict Constraint)** for `train` mode. In self-play training, the quality of the data is paramount; generating millions of garbage positions faster does not yield a stronger model. By forcing `parallel_games >= max_batch_size`, we ensure Optuna only explores configurations that preserve MCTS integrity.