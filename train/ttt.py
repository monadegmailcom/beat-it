from ttt_cnn import TicTacToeCNN

# --- Game-specific constants ---
G_SIZE = 27  # 3 planes * 9 cells
P_SIZE = 9   # 9 possible moves

# --- Game and Network Configuration Dictionary ---
game_config = {
    'board_size': 3,
    'num_actions': 9,
    'input_channels': 3,  # X pieces, O pieces, player-to-move
}

# --- Model Instantiation ---
model = TicTacToeCNN(**game_config)

# --- Training Configuration ---
basename = "ttt_alphazero_experiment" # For TensorBoard runs
set_model_func_name = "set_ttt_model"
fetch_data_func_name = "fetch_ttt_selfplay_data"
