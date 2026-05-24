import os
import json
import torch
import ctypes
from train.utils import (
    evaluate_models, evaluate_against_minimax_from_cpp,
    evaluate_minimax_vs_minimax_from_cpp, Hyperparameters, GameType,
    create_inference_model_bundle
)

# Standard default parameters for MCTS search
DEFAULT_MCTS_PARAMS = {
    'c_base': 19652.0,
    'c_init': 1.25,
    'dirichlet_alpha': 0.3,
    'dirichlet_epsilon': 0.25,
    'simulations': 800,
    'opening_moves': 5,
    'parallel_simulations': 2,
    'max_batch_size': 64
}

def create_model_dynamic(game_type_str, game_config):
    if game_type_str == "ttt":
        from train.ttt_cnn import create_model
    else:
        from train.uttt import create_model
    return create_model(game_config)

def main():
    # 1. Load Evaluation Configuration
    eval_config_path = "train/eval_config.json"
    eval_config_default_path = "train/eval_config.json.default"
    
    if os.path.exists(eval_config_path):
        print(f"Loading configuration from {eval_config_path}...")
        with open(eval_config_path, 'r') as f:
            eval_config = json.load(f)
    else:
        print(f"Configuration file {eval_config_path} not found. Falling back to {eval_config_default_path}...")
        with open(eval_config_default_path, 'r') as f:
            eval_config = json.load(f)

    # Determine game type
    game_type_str = eval_config.get('game_type', 'uttt').lower()
    if game_type_str == "ttt":
        game_type = GameType.TTT
    else:
        game_type = GameType.UTTT

    print(f"Game type resolved to: {game_type_str.upper()}")

    # 2. Setup C++ shared library bindings
    possible_paths = [
        os.path.join('build', 'libalphazero.dylib'),
        os.path.join('build', 'libalphazero.so'),
    ]
    lib_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if lib_path is None:
        raise FileNotFoundError("Could not find libalphazero shared library. Did you compile it?")
    alphazero_lib = ctypes.CDLL(lib_path)

    # 3. Helper functions for player loading and MCTS hyperparameter creation
    def load_player_model_bytes(player_config):
        path = player_config.get('model_path', '')
        if path == "random":
            print("Creating a randomly initialized NN model...")
            # We ONLY load the training architecture config here to generate the untrained layer layout
            arch_config_path = "train/ttt_config.json" if game_type_str == "ttt" else "train/uttt_config.json"
            if not os.path.exists(arch_config_path):
                raise FileNotFoundError(f"Required architecture configuration file '{arch_config_path}' not found.")
            with open(arch_config_path, 'r') as f:
                arch_config = json.load(f)
            
            model = create_model_dynamic(game_type_str, arch_config['game_config'])
            model.eval()
            
            m_bytes, _ = create_inference_model_bundle(
                model, 0, None, arch_config['game_config'],
                arch_config.get('self_play_config', {}),
                arch_config.get('training_hyperparams', {})
            )
            return m_bytes
        else:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: '{path}'")
            print(f"Reading model file: '{path}'")
            with open(path, 'rb') as f:
                return f.read()

    def make_hyperparameters(player_config):
        hp_config = {
            'c_base': player_config.get('c_base', DEFAULT_MCTS_PARAMS['c_base']),
            'c_init': player_config.get('c_init', DEFAULT_MCTS_PARAMS['c_init']),
            'dirichlet_alpha': player_config.get('dirichlet_alpha', DEFAULT_MCTS_PARAMS['dirichlet_alpha']),
            'dirichlet_epsilon': player_config.get('dirichlet_epsilon', DEFAULT_MCTS_PARAMS['dirichlet_epsilon']),
            'simulations': player_config.get('simulations', DEFAULT_MCTS_PARAMS['simulations']),
            'opening_moves': player_config.get('opening_moves', DEFAULT_MCTS_PARAMS['opening_moves']),
            'parallel_games': eval_config.get('parallel_games', 10),
            'parallel_simulations': player_config.get('parallel_simulations', DEFAULT_MCTS_PARAMS['parallel_simulations']),
            'max_batch_size': player_config.get('max_batch_size', DEFAULT_MCTS_PARAMS['max_batch_size'])
        }
        return Hyperparameters(hp_config)

    # 4. Dispatch Matchup
    p1 = eval_config['player1']
    p2 = eval_config['player2']
    p1_type = p1['type'].lower()
    p2_type = p2['type'].lower()

    rounds = eval_config.get('rounds', 100)
    save_path = eval_config.get('save_path', 'test_eval_custom.json')
    run_name = eval_config.get('run_name', 'custom_matchup')
    step = eval_config.get('step', 0)
    parallel_games = eval_config.get('parallel_games', 10)

    # MCTS vs MCTS Matchup
    if p1_type == "mcts" and p2_type == "mcts":
        model1_bytes = load_player_model_bytes(p1)
        model2_bytes = load_player_model_bytes(p2)
        
        hp1 = make_hyperparameters(p1)
        hp2 = make_hyperparameters(p2)
        
        print(f"Starting MCTS vs MCTS Matchup ({rounds} rounds)...")
        res = evaluate_models(
            None, alphazero_lib.evaluate_models, game_type,
            model1_bytes, model2_bytes, hp1, hp2, rounds,
            save_path, run_name, step
        )
        print("\n========================================")
        print("Match Complete")
        print("========================================")
        print(f"Player 1 (MCTS: {p1.get('model_path')}) wins: {res.wins_p1}")
        print(f"Player 2 (MCTS: {p2.get('model_path')}) wins: {res.wins_p2}")
        print(f"Draws: {res.draws}")
        print("========================================\n")

    # Minimax vs Minimax Matchup
    elif p1_type == "minimax" and p2_type == "minimax":
        depth1 = p1.get('depth', 3)
        depth2 = p2.get('depth', 3)
        
        print(f"Starting Minimax vs Minimax Matchup (Depth {depth1} vs Depth {depth2}, {rounds} rounds)...")
        res = evaluate_minimax_vs_minimax_from_cpp(
            None, alphazero_lib.evaluate_minimax_vs_minimax, game_type,
            rounds, depth1, depth2, save_path, run_name, step, parallel_games
        )
        print("\n========================================")
        print("Match Complete")
        print("========================================")
        print(f"Player 1 (Minimax Depth {depth1}) wins: {res.wins_p1}")
        print(f"Player 2 (Minimax Depth {depth2}) wins: {res.wins_p2}")
        print(f"Draws: {res.draws}")
        print("========================================\n")

    # MCTS vs Minimax / Minimax vs MCTS Matchup
    elif (p1_type == "mcts" and p2_type == "minimax") or (p1_type == "minimax" and p2_type == "mcts"):
        is_p1_mcts = (p1_type == "mcts")
        mcts_player = p1 if is_p1_mcts else p2
        minimax_player = p2 if is_p1_mcts else p1
        
        model_bytes = load_player_model_bytes(mcts_player)
        hp = make_hyperparameters(mcts_player)
        minimax_depth = minimax_player.get('depth', 3)
        
        print(f"Starting MCTS ({mcts_player.get('model_path')}) vs Minimax (Depth {minimax_depth}) Matchup ({rounds} rounds)...")
        res = evaluate_against_minimax_from_cpp(
            None, alphazero_lib.evaluate_against_minimax, game_type,
            model_bytes, hp, rounds, minimax_depth, save_path, run_name, step
        )
        
        # Handle result swapping so player 1 gets their mapped outcomes
        if is_p1_mcts:
            wins_p1 = res.wins_p1
            wins_p2 = res.wins_p2
        else:
            wins_p1 = res.wins_p2
            wins_p2 = res.wins_p1
            
        print("\n========================================")
        print("Match Complete")
        print("========================================")
        p1_label = f"MCTS: {p1.get('model_path')}" if is_p1_mcts else f"Minimax Depth {p1.get('depth')}"
        p2_label = f"Minimax Depth {p2.get('depth')}" if is_p1_mcts else f"MCTS: {p2.get('model_path')}"
        print(f"Player 1 ({p1_label}) wins: {wins_p1}")
        print(f"Player 2 ({p2_label}) wins: {wins_p2}")
        print(f"Draws: {res.draws}")
        print("========================================\n")

    else:
        print(f"Unsupported matchup types: {p1_type} vs {p2_type}")

if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    main()
