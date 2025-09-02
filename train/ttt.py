from .ttt_cnn import TicTacToeCNN


def create_model(game_config):
    """Factory function to create a model from a config dictionary."""
    return TicTacToeCNN(**game_config)

# --- Training Configuration ---
basename = "ttt_alphazero_experiment"  # For TensorBoard runs
set_model_func_name = "set_ttt_model"
fetch_data_func_name = "fetch_ttt_selfplay_data"
