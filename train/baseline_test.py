import os
import torch
import ctypes
from train.utils import (
    evaluate_models, evaluate_against_minimax_from_cpp, Hyperparameters, GameType,
    create_inference_model_bundle
)
from train.uttt import create_model

def main():
    checkpoint_path = "models/checkpoint_24_05.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return

    print(f"Loading {checkpoint_path}...")
    with open(checkpoint_path, 'rb') as f:
        model1_bytes = f.read()

    # Load configuration from uttt_config.json
    import json
    with open("train/uttt_config.json", 'r') as f:
        config = json.load(f)
    game_config = config['game_config']
    self_play_config = config['self_play_config']
    training_hyperparams = config['training_hyperparams']

    # Initialize a purely random model
    model2 = create_model(game_config)
    model2.eval()
    model2_bytes, _ = create_inference_model_bundle(
        model2, 0, None, game_config, self_play_config, training_hyperparams
    )

    # Setup C++ bindings
    possible_paths = [
        os.path.join('build', 'libalphazero.dylib'),
        os.path.join('build', 'libalphazero.so'),
    ]
    lib_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if lib_path is None:
        raise FileNotFoundError("Could not find libalphazero shared library.")
    alphazero_lib = ctypes.CDLL(lib_path)

    evaluate_func = alphazero_lib.evaluate_models
    evaluate_minimax_func = alphazero_lib.evaluate_against_minimax

    # Evaluation Hyperparameters
    hp_config = {
        'c_base': self_play_config.get('c_base', 19652.0),
        'c_init': self_play_config.get('c_init', 1.25),
        'dirichlet_alpha': self_play_config.get('dirichlet_alpha', 0.3),
        'dirichlet_epsilon': self_play_config.get('dirichlet_epsilon', 0.25),
        'simulations': training_hyperparams.get('evaluation_simulations', 100),
        'opening_moves': self_play_config.get('opening_moves', 5),
        'parallel_games': training_hyperparams.get('evaluation_parallel_games', 10),
        'parallel_simulations': training_hyperparams.get('evaluation_parallel_simulations', 2),
        'max_batch_size': training_hyperparams.get('evaluation_max_batch_size', 64)
    }
    hp = Hyperparameters(hp_config)

    game_type = GameType.UTTT
    rounds = 100

    print("\n========================================")
    print("Phase 1: Trained NN vs Random NN")
    print("========================================")
    res_random = evaluate_models(
        None, evaluate_func, game_type, model1_bytes, model2_bytes,
        hp, rounds, "test_eval_random.json", "random_test", 0
    )
    print(f"Result -> Wins: {res_random.wins_p1}, Losses: {res_random.wins_p2}, Draws: {res_random.draws}\n")

    print("========================================")
    print("Phase 2: Trained NN vs Minimax (Depth 3)")
    print("========================================")
    res_minimax = evaluate_against_minimax_from_cpp(
        None, evaluate_minimax_func, game_type, model1_bytes,
        hp, rounds, 3, "test_eval_minimax.json", "minimax_test", 0
    )
    print(f"Result -> Wins: {res_minimax.wins_p1}, Losses: {res_minimax.wins_p2}, Draws: {res_minimax.draws}")
    print("========================================\n")


if __name__ == "__main__":
    # Ensure PyTorch thread starvation doesn't occur
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    main()
