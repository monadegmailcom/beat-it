import os
import json
import torch
import ctypes
from train.utils import (
    evaluate_matchup_from_cpp, MatchupPlayerConfig, Hyperparameters, GameType,
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

    rounds = eval_config.get('rounds', 100)
    save_path = eval_config.get('save_path', 'temp/test_eval_custom.json')
    run_name = eval_config.get('run_name', 'custom_matchup')
    step = eval_config.get('step', 0)

    def build_matchup_player_config(p_cfg_dict):
        p_type = p_cfg_dict.get('type', '').lower()
        if p_type == "mcts":
            t = 1
            sims_or_depth = p_cfg_dict.get('simulations', DEFAULT_MCTS_PARAMS['simulations'])
            model_bytes = load_player_model_bytes(p_cfg_dict)
            hp_struct = make_hyperparameters(p_cfg_dict)
        elif p_type in ("minimax", "standard_minimax"):
            t = 2
            sims_or_depth = p_cfg_dict.get('depth', 3)
            model_bytes = b""
            hp_struct = Hyperparameters(DEFAULT_MCTS_PARAMS)
        elif p_type in ("tree_minimax", "tree", "treeminimax", "tree minimax"):
            t = 3
            sims_or_depth = p_cfg_dict.get('depth', 3)
            model_bytes = b""
            hp_struct = Hyperparameters(DEFAULT_MCTS_PARAMS)
        else:
            raise ValueError(f"Unknown player type: {p_type}")

        return MatchupPlayerConfig(
            type=t,
            simulations_or_depth=sims_or_depth,
            model_data=model_bytes,
            model_data_len=len(model_bytes),
            hp=hp_struct
        )

    p1_cfg = build_matchup_player_config(p1)
    p2_cfg = build_matchup_player_config(p2)

    print(f"Starting Matchup: Player 1 (Type: {p1['type']}) vs Player 2 (Type: {p2['type']}) ({rounds} rounds)...")
    
    res = evaluate_matchup_from_cpp(
        None, alphazero_lib.evaluate_matchup, game_type,
        p1_cfg, p2_cfg, rounds, save_path, run_name, step
    )
    
    print("\n========================================")
    print("Match Complete")
    print("========================================")
    
    def get_player_label(p, p_cfg):
        ptype = p['type'].lower()
        if ptype == "mcts":
            return f"MCTS: {p.get('model_path')}"
        elif ptype in ("minimax", "standard_minimax"):
            return f"Minimax (Depth {p_cfg.simulations_or_depth})"
        elif ptype in ("tree_minimax", "tree", "treeminimax", "tree minimax"):
            return f"Tree Minimax (Depth {p_cfg.simulations_or_depth})"
        return p['type']

    p1_label = get_player_label(p1, p1_cfg)
    p2_label = get_player_label(p2, p2_cfg)
    
    print(f"Player 1 ({p1_label}) wins: {res.wins_p1}")
    print(f"Player 2 ({p2_label}) wins: {res.wins_p2}")
    print(f"Draws: {res.draws}")
    print("========================================\n")

if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    main()
