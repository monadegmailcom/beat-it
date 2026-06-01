from .ttt_cnn import TicTacToeCNN


def create_model(game_config):
    """Factory function to create a model from a config dictionary."""
    return TicTacToeCNN(
        input_channels=game_config['input_channels'],
        board_size=game_config['board_size'],
        num_actions=game_config['num_actions'],
        conv_channels=game_config.get('res_block_channels', 64),
        fc_hidden_size=game_config.get('fc_hidden_size', 128)
    )

# --- Training Configuration ---
basename = "ttt_alphazero_experiment"  # For TensorBoard runs
set_model_func_name = "set_ttt_model"
fetch_data_func_name = "fetch_ttt_selfplay_data"
