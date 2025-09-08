import ctypes
import os
import argparse
import time
import json
import torch
import optuna
from typing import Callable

# --- Ctypes Definitions ---
# These must match the structs in lib_interface.cpp


number_of_selfplay_workers = "number_of_selfplay_workers"
number_of_threads_per_selfplay_worker = "number_of_threads_per_selfplay_worker"
max_number_of_threads_per_selfplay_worker = \
    "max_number_of_threads_per_selfplay_worker"
min_batch_size = "min_batch_size"


class OptimizerParams(ctypes.Structure):
    _fields_ = [
        (number_of_selfplay_workers, ctypes.c_uint32),
        (number_of_threads_per_selfplay_worker, ctypes.c_uint32),
        (max_number_of_threads_per_selfplay_worker, ctypes.c_uint32),
        (min_batch_size, ctypes.c_uint32),
    ]


class FixParams(ctypes.Structure):
    _fields_ = [
        ("simulations_per_move", ctypes.c_uint32),
        ("number_of_games", ctypes.c_uint32),
    ]


# Define the function signature for the C measurement function. This creates a
# type alias for clarity. Using typing.Callable is the standard, idiomatic
# way to hint function-like objects for mypy and other type checkers.
MeasureFuncType = Callable[
    [ctypes.c_void_p, ctypes.POINTER(FixParams),  # type: ignore
     ctypes.POINTER(OptimizerParams)],  # type: ignore
    int  # The C uint32_t will be returned as a Python int
]


def objective(
        trial: optuna.Trial, session_handle: ctypes.c_void_p,  # type: ignore
        initial_params: dict, fix_params: FixParams,
        c_measure_func: MeasureFuncType) -> float:
    """
    The objective function for Optuna to optimize.

    It suggests hyperparameter values, calls the C++ measurement function,
    and returns the throughput (number of positions generated per second).
    """
    # --- Suggest Hyperparameters ---
    # N: Number of parallel self-play workers (threads)
    # P: Number of internal threads per worker (for MCTS)
    # T: Minimum batch size for the neural network inference
    n_workers = trial.suggest_int(
        number_of_selfplay_workers,
        1,
        5)
    p_threads_per_worker = trial.suggest_int(
        number_of_threads_per_selfplay_worker,
        10,
        20)
    t_min_batch_size = trial.suggest_int(
        min_batch_size,
        1,
        10)

    # --- Prepare Parameters for C++ Call ---
    # These are the parameters we are optimizing
    opt_params = OptimizerParams(
        n_workers, p_threads_per_worker, t_min_batch_size)

    # --- Run Measurement ---
    print(
        f"  Trial {trial.number}: Testing N={n_workers}, "
        f"P={p_threads_per_worker}, T={t_min_batch_size}...")

    start_time = time.time()
    total_positions = c_measure_func(
        session_handle, ctypes.byref(fix_params), ctypes.byref(opt_params))
    duration = time.time() - start_time

    if duration > 0:
        positions_per_second = total_positions / duration
    else:
        positions_per_second = 0.0

    print(f"  ... Generated {total_positions} positions in {duration:.2f}s. "
          f"Throughput: {positions_per_second:.2f} pos/s")
    return positions_per_second


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for self-play throughput.")
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to a model checkpoint (.pt file) to use for inference.')
    parser.add_argument(
        '--n_trials', type=int, default=100,
        help='Number of optimization trials to run.')
    parser.add_argument(
        '--simulations_per_move', type=int, default=100,
        help='Number of MCTS simulations per move for each game in a trial.')
    parser.add_argument(
        '--number_of_games', type=int, default=20,
        help='Number of games to play per trial to measure throughput.')
    parser.add_argument(
        '--study_name', type=str, default="selfplay_throughput_opt",
        help='Name for the Optuna study, used for organizing runs.')
    args = parser.parse_args()

    session_handle: ctypes.c_void_p
    try:
        # --- C++ Library and Session Setup ---
        lib_path = os.path.join('obj', 'libalphazero.so')
        alphazero_lib = ctypes.CDLL(lib_path)

        alphazero_lib.create_session.restype = ctypes.c_void_p
        alphazero_lib.destroy_session.argtypes = [ctypes.c_void_p]
        session_handle = alphazero_lib.create_session()
        if not session_handle:
            raise RuntimeError("Failed to create C++ session.")

        # --- Load Model and Modify Metadata ---
        print(f"Loading model from: {args.model_path}")
        with open(args.model_path, 'rb') as f:
            model_bytes = f.read()

        # Extract metadata without starting background workers.
        # We load the metadata, set 'threads' to 0, and then pass this
        # modified metadata to set_model. This prevents the normal self-play
        # workers from starting and interfering with our measurements.
        extra_files = {'metadata.json': b''}
        torch.jit.load(
            args.model_path, map_location='cpu', _extra_files=extra_files)
        metadata = json.loads(extra_files['metadata.json'])

        # do not start workers
        metadata['self_play_config']['threads'] = 0
        modified_metadata_json = json.dumps(metadata).encode('utf-8')

        # --- Initialize C++ Session with the Model ---
        c_set_model_func = alphazero_lib.set_uttt_model
        c_set_model_func.restype = ctypes.c_int
        c_set_model_func.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
            ctypes.c_char_p, ctypes.c_int32
        ]

        result = c_set_model_func(
            session_handle, model_bytes, len(model_bytes),
            modified_metadata_json, len(modified_metadata_json))
        if result < 0:
            raise RuntimeError("Failed to set model in C++ session.")
        print("C++ session initialized successfully.")

        # --- Setup Measurement Function ---
        c_measure_func = alphazero_lib.measure_uttt_selfplay_throughput
        c_measure_func.restype = ctypes.c_uint32
        c_measure_func.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(FixParams),
            ctypes.POINTER(OptimizerParams)
        ]

        # --- Run Optuna Study ---
        study = optuna.create_study(
            storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
            study_name=args.study_name,
            direction="maximize")

        # Enqueue a trial with a known good starting point. Optuna will run
        # this trial first before starting its own search.
        cpu_count = os.cpu_count() or 1
        initial_params = {
            number_of_selfplay_workers: max(1, int(cpu_count / 2)),
            number_of_threads_per_selfplay_worker: int(cpu_count),
            max_number_of_threads_per_selfplay_worker: int(cpu_count * 8),
            min_batch_size: max(1, int(cpu_count / 2))
        }
        print(f"Enqueuing initial trial with parameters: {initial_params}")
        study.enqueue_trial(initial_params)

        # Create the fixed parameters object from the command-line arguments
        fix_params = FixParams(
            simulations_per_move=args.simulations_per_move,
            number_of_games=args.number_of_games)

        study.optimize(
            lambda trial: objective(
                trial, session_handle, initial_params, fix_params,
                c_measure_func),
            n_trials=args.n_trials)

        # --- Print Results ---
        print("\nOptimization finished.")
        print(f"  Number of trials: {len(study.trials)}")
        best_trial = study.best_trial
        print("  Best trial:")
        print(
            f"    Value (Throughput): {best_trial.value:.2f} positions/sec")
        print("    Params: ")
        for key, value in best_trial.params.items():
            print(f"      {key}: {value}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        if session_handle:
            print("\nCleaning up C++ session...")
            alphazero_lib.destroy_session(session_handle)
            print("C++ session cleanup complete.")
