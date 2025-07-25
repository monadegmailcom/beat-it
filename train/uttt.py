from .alphazero_cnn import AlphaZeroCNN

# --- Game and Network Configuration Dictionary ---
game_config = {
    'board_size': 9,
    'num_actions': 81,
    'input_channels': 4,  # X pieces, O pieces, valid-moves, player-to-move
    'num_res_blocks': 2,
    'res_block_channels': 64,
    'fc_hidden_size': 256  # Hidden layer size for value/policy heads
}

# --- Model Instantiation ---
model = AlphaZeroCNN(**game_config)

# --- Training Configuration ---
basename = "uttt_alphazero_experiment"
set_model_func_name = "set_uttt_model"  # Assuming you'll create this in C++
fetch_data_func_name = "fetch_uttt_selfplay_data"
