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
    TrainingHyperparameters, CppStats, check_and_merge_config,
    fetch_selfplay_data_from_cpp, ReplayBuffer
)

def get_usable_cpu_count() -> int:
    import math
    try:
        # Check cgroup v2
        with open('/sys/fs/cgroup/cpu.max', 'r') as f:
            max_val, period = f.read().strip().split()
            if max_val != 'max':
                return math.ceil(int(max_val) / int(period))
    except Exception:
        pass
    try:
        # Check cgroup v1
        with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', 'r') as f:
            quota = int(f.read().strip())
        with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us', 'r') as f:
            period = int(f.read().strip())
        if quota > 0:
            return math.ceil(quota / period)
    except Exception:
        pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4

def measure_throughput(
        alphazero_lib: ctypes.CDLL,
        game_type: GameType,
        model_bytes: bytes,
        hp: Hyperparameters,
        number_of_positions: int) -> tuple[float, float]:
    """
    Measures throughput by calling the C++ measure_selfplay_throughput function.
    Returns (positions_per_second, average_inference_batch_size).
    """
    c_measure_func = alphazero_lib.measure_selfplay_throughput
    c_measure_func.restype = ctypes.c_uint32
    c_measure_func.argtypes = [
        ctypes.c_int32,          # GameType
        ctypes.c_char_p,         # model_data
        ctypes.c_uint32,         # model_data_len
        ctypes.POINTER(Hyperparameters), # hp
        ctypes.c_uint32,         # number_of_positions
        ctypes.POINTER(CppStats) # batch_size_stats
    ]

    batch_size_stats = CppStats()
    start_time = time.time()
    total_positions = c_measure_func(
        game_type.value,
        model_bytes,
        len(model_bytes),
        ctypes.byref(hp),
        number_of_positions,
        ctypes.byref(batch_size_stats)
    )
    duration = time.time() - start_time
    avg_batch_size = batch_size_stats.mean

    if duration > 0:
        return (total_positions / duration), avg_batch_size
    else:
        return 0.0, 0.0

def measure_throughput_with_training(
        alphazero_lib: ctypes.CDLL,
        game_type: GameType,
        model_bytes: bytes,
        hp: Hyperparameters,
        number_of_positions: int,
        training_hyperparams: dict,
        game_config: dict) -> tuple[float, float]:
    """
    Measures throughput by fetching data and doing mock PyTorch backward passes
    to capture actual training overhead.
    """
    import torch.nn.functional as F
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Mock python model
    game_module = importlib.import_module(f".{game_type.name.lower()}", package=__package__)
    model = game_module.create_model(game_config).to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    G_SIZE = game_config['input_channels'] * game_config['board_size'] * game_config['board_size']
    P_SIZE = game_config['num_actions']
    replay_buffer = ReplayBuffer(40000, G_SIZE, P_SIZE, device)

    # Initialize C++ session
    c_create_session = alphazero_lib.create_session
    c_create_session.restype = ctypes.c_void_p
    c_create_session.argtypes = [
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.POINTER(Hyperparameters)
    ]
    
    session_handle = c_create_session(
        game_type.value,
        model_bytes,
        len(model_bytes),
        ctypes.byref(hp)
    )

    c_fetch_data_func = alphazero_lib.fetch_selfplay_data
    
    c_destroy_session = alphazero_lib.destroy_session
    c_destroy_session.argtypes = [ctypes.c_void_p]

    total_fetched = 0
    start_time = time.time()
    
    avg_batch_sizes = []
    
    while total_fetched < number_of_positions:
        new_data, stats = fetch_selfplay_data_from_cpp(
            session_handle, c_fetch_data_func, game_type,
            256, G_SIZE, P_SIZE)

        if new_data:
            total_fetched += len(new_data)
            replay_buffer.add(new_data)
        
        if stats['inference_batch_size'].mean > 0:
            avg_batch_sizes.append(stats['inference_batch_size'].mean)

        batch_size = training_hyperparams.get('batch_size', 1024)
        if len(replay_buffer) >= batch_size:
            batch_states, batch_target_policies, batch_target_values = replay_buffer.sample(batch_size)
            batch_states = batch_states.to(device)
            batch_target_policies = batch_target_policies.to(device)
            batch_target_values = batch_target_values.to(device)

            optimizer.zero_grad()
            pred_values, pred_policy_logits = model(batch_states)
            
            loss_policy = F.cross_entropy(pred_policy_logits, batch_target_policies)
            loss_value = F.mse_loss(pred_values.squeeze(-1), batch_target_values)
            loss = loss_policy + loss_value
            loss.backward()
            optimizer.step()

    duration = time.time() - start_time
    c_destroy_session(session_handle)

    final_avg_batch_size = sum(avg_batch_sizes) / len(avg_batch_sizes) if avg_batch_sizes else 0.0

    if duration > 0:
        return (total_fetched / duration), final_avg_batch_size
    else:
        return 0.0, 0.0


def objective(
        trial: optuna.Trial,
        alphazero_lib: ctypes.CDLL,
        game_type: GameType,
        model_bytes: bytes,
        base_hp_config: dict,
        training_hyperparams: dict,
        game_config: dict,
        mode: str,
        number_of_positions: int) -> float:
    """
    Optuna objective function.
    """
    # Create a copy of config to modify
    config = base_hp_config.copy()

    # Dynamically scale search space based on the current host's actual usable CPU count.
    # RunPod/Docker exposes all host CPUs to os.cpu_count(), so we check cgroup quotas first.
    cpu_count = get_usable_cpu_count()
    min_threads = max(1, int(cpu_count * 0.5))
    max_threads = int(cpu_count * 1.5)

    if mode == "train":
        # 2. max_batch_size determines the GPU queue capacity. As you correctly noted, 
        # MCTS threads can push multiple evaluations (via virtual loss) before blocking.
        # Therefore, we must tune the queue capacity independently!
        max_batch_size = trial.suggest_int('max_batch_size', max_threads, 4096, log=True)
        config['max_batch_size'] = max_batch_size

        # 1. Tune parallel_games around the CPU count (pristine single-thread search per game)
        parallel_games = trial.suggest_int('parallel_games', min_threads, max_threads)
        config['parallel_games'] = parallel_games
        config['parallel_simulations'] = 1
        
    elif mode == "match":
        # 2. In match mode, use evaluation_max_batch_size to keep the dashboard and hyperparameter tracking perfectly decoupled.
        evaluation_max_batch_size = trial.suggest_int('evaluation_max_batch_size', max_threads, 4096, log=True)
        config['max_batch_size'] = evaluation_max_batch_size

        # 1. Tune both parallel_games and parallel_simulations for Match mode!
        # To prevent thread explosion (total threads = games * simulations) from 
        # crashing the CPU, we constrain their product to roughly 2x the CPU count.
        evaluation_parallel_games = trial.suggest_int('evaluation_parallel_games', 1, max_threads)
        max_sims_allowed = max(1, (max_threads * 2) // evaluation_parallel_games)
        evaluation_parallel_simulations = trial.suggest_int('evaluation_parallel_simulations', 1, max_sims_allowed)
        
        config['parallel_simulations'] = evaluation_parallel_simulations
        config['parallel_games'] = evaluation_parallel_games
    
    hp = Hyperparameters(config)

    print(f"  Trial {trial.number}: Mode={mode}, "
          f"PG={hp.parallel_games}, PS={hp.parallel_simulations}, "
          f"BS={hp.max_batch_size}...")

    if mode == "train":
        throughput, avg_batch_size = measure_throughput_with_training(
            alphazero_lib, game_type, model_bytes, hp, number_of_positions,
            training_hyperparams, game_config
        )
    else:
        throughput, avg_batch_size = measure_throughput(
            alphazero_lib, game_type, model_bytes, hp, number_of_positions
        )

    if avg_batch_size > 0.0:
        trial.set_user_attr('avg_inference_batch_size', avg_batch_size)
        print(f"  ... Throughput: {throughput:.2f} pos/s, Avg Batch Size: {avg_batch_size:.2f}")
    else:
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
        '--n_trials', type=int, default=200,
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
            
        # Smartly merge in new default config keys from pristine Git version if available
        git_config_path = config_path + ".git"
        if os.path.exists(git_config_path):
            try:
                with open(git_config_path, 'r') as gf:
                    git_config = json.load(gf)
                if check_and_merge_config(full_config, git_config, config_path):
                    print(f"Saving merged configuration back to persistent file: {config_path}")
                    with open(config_path, 'w') as f_write:
                        json.dump(full_config, f_write, indent=4)
            except Exception as ex:
                print(f"Warning: Could not merge with Git default configuration: {ex}")
                
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
            self_play_config, training_hyperparams, game_config, args.mode, args.number_of_positions
        ),
        n_trials=args.n_trials
    )

    print("\nOptimization finished.")
    print("Best trial:")
    print(f"  Value: {study.best_value:.2f} pos/s")
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
