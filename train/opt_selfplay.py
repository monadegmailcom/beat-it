import ctypes
import os
import argparse
import time
import json
import torch
import optuna
from typing import Callable, cast
import importlib

from .utils import (
    GameType, create_inference_model_bundle, Hyperparameters,
    TrainingHyperparameters
)

def measure_throughput(
        alphazero_lib: ctypes.CDLL,
        game_type: GameType,
        model_bytes: bytes,
        hp: Hyperparameters,
        number_of_positions: int) -> float:
    """
    Measures throughput by calling the C++ measure_selfplay_throughput function.
    Returns positions per second.
    """
    
    # Define C func signature
    c_measure_func = alphazero_lib.measure_selfplay_throughput
    c_measure_func.restype = ctypes.c_uint32
    c_measure_func.argtypes = [
        ctypes.c_int32,          # GameType
        ctypes.c_char_p,         # model_data
        ctypes.c_uint32,         # model_data_len
        ctypes.POINTER(Hyperparameters), # hp
        ctypes.c_uint32          # number_of_positions
    ]

    start_time = time.time()
    total_positions = c_measure_func(
        game_type.value,
        model_bytes,
        len(model_bytes),
        ctypes.byref(hp),
        number_of_positions
    )
    duration = time.time() - start_time

    if duration > 0:
        return total_positions / duration
    else:
        return 0.0


def objective(
        trial: optuna.Trial,
        alphazero_lib: ctypes.CDLL,
        game_type: GameType,
        model_bytes: bytes,
        base_hp_config: dict,
        mode: str,
        number_of_positions: int) -> float:
    """
    Optuna objective function.
    """
    # Create a copy of config to modify
    config = base_hp_config.copy()

    # --- Suggest Hyperparameters based on Mode ---
    
    # Common parameter: batch size
    # We can probably search in powers of 2 or typical gpu sizes
    max_batch_size = trial.suggest_int('max_batch_size', 16, 4096, log=True)
    config['max_batch_size'] = max_batch_size

    if mode == "train":
        parallel_games = trial.suggest_int('parallel_games', 1, 256, log=True)
        config['parallel_games'] = parallel_games
        config['parallel_simulations'] = 1
    elif mode == "match":
        parallel_simulations = os.cpu_count() or 1
        config['parallel_simulations'] = parallel_simulations
        config['parallel_games'] = 1
    
    hp = Hyperparameters(config)

    print(f"  Trial {trial.number}: Mode={mode}, "
          f"PG={hp.parallel_games}, PS={hp.parallel_simulations}, "
          f"BS={hp.max_batch_size}...")

    throughput = measure_throughput(
        alphazero_lib, game_type, model_bytes, hp, number_of_positions
    )

    print(f"  ... Throughput: {throughput:.2f} pos/s")
    return throughput


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for self-play throughput.")
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to a model checkpoint (.pt file).')
    parser.add_argument(
        '--game', type=str, required=True,
        help='The game to train on (e.g., "ttt", "uttt").')
    parser.add_argument(
        '--mode', type=str, required=True, choices=['train', 'match'],
        help='Optimization mode: "train" or "match".')
    parser.add_argument(
        '--n_trials', type=int, default=50,
        help='Number of optimization trials to run.')
    parser.add_argument(
        '--number_of_positions', type=int, default=1000,
        help='Number of positions to generate per trial.')
    parser.add_argument(
        '--study_name', type=str, default=None,
        help='Name for the Optuna study.')
    
    args = parser.parse_args()

    # --- Load Library ---
    lib_dir = os.environ.get('KAGGLE_LIB_DIR', '/kaggle/input/alphazero-lib/torch_lib')
    if os.path.exists(lib_dir):
        print(f"Loading dependencies from {lib_dir}...")
        os.environ['LD_LIBRARY_PATH'] = f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        
        # Pre-load torch dependencies with RTLD_GLOBAL to avoid undefined symbol errors
        deps = ["libc10.so", "libc10_cuda.so", "libtorch_cpu.so", "libtorch_cuda.so", "libtorch.so"]
        for dep in deps:
            dep_path = os.path.join(lib_dir, dep)
            if os.path.exists(dep_path):
                try:
                    ctypes.CDLL(dep_path, mode=ctypes.RTLD_GLOBAL)
                except Exception as e:
                    print(f"Warning: Could not pre-load {dep}: {e}")

    possible_paths = [
        os.path.join('build', 'libalphazero.dylib'),
        os.path.join('build', 'libalphazero.so'),
        os.path.join('obj', 'libalphazero.so'),
        '/kaggle/input/alphazero-lib/libalphazero.so',
        'libalphazero.so'
    ]
    lib_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if lib_path is None:
        raise FileNotFoundError(f"Could not find libalphazero shared library. Checked: {possible_paths}")
    
    print(f"Loading library: {lib_path}")
    alphazero_lib = ctypes.CDLL(lib_path)

    # --- Game Type ---
    try:
        game_type = GameType[args.game.upper()]
    except KeyError:
        print(f"Error: Invalid game type '{args.game}'. Available: {[e.name for e in GameType]}")
        exit(1)

    # --- Load Model & Config ---
    print(f"Loading model from: {args.model_path}")
    
    # We need to load configurations to pass base settings (like c_base, etc.)
    # that we aren't optimizing but need to exist.
    try:
        config_path = os.path.join(
            os.path.dirname(__file__), f"{args.game}_config.json")
        with open(config_path, 'r') as f:
            full_config = json.load(f)
        self_play_config = full_config.get('self_play_config', {})
        game_config = full_config.get('game_config', {})
        training_hyperparams = full_config.get('training_hyperparams', {})
    except Exception as e:
        print(f"Warning: Could not load config file for {args.game}: {e}")
        self_play_config = {}
        game_config = {}
        training_hyperparams = {}

    # Load model to create the bundle (we need model_bytes)
    # We use cpu to load initial model for bundling
    device = torch.device("cpu")
    try:
        game_module = importlib.import_module(f".{args.game}", package=__package__)
        with open(args.model_path, 'rb') as f:
            model_bytes = f.read()

    except Exception as e:
        print(f"Error preparing model: {e}")
        exit(1)

    # --- Run Optuna ---
    if args.study_name is None:
        args.study_name = f"selfplay_{args.mode}_{int(time.time())}"

    os.makedirs(os.environ.get('BASE_RUNS_DIR', 'runs'), exist_ok=True)
    db_path = os.path.abspath(
        os.path.join(os.environ.get('BASE_RUNS_DIR', 'runs'), 'optuna.db')
    )
    
    study = optuna.create_study(
        storage=f"sqlite:///{db_path}",
        study_name=args.study_name,
        direction="maximize",
        load_if_exists=True
    )

    print(f"Starting optimization for mode: {args.mode}")
    print(f"Study name: {args.study_name}")
    print(f"Positions per trial: {args.number_of_positions}")

    study.optimize(
        lambda trial: objective(
            trial, alphazero_lib, game_type, model_bytes, 
            self_play_config, args.mode, args.number_of_positions
        ),
        n_trials=args.n_trials
    )

    print("\nOptimization finished.")
    print("Best trial:")
    print(f"  Value: {study.best_value:.2f} pos/s")
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
