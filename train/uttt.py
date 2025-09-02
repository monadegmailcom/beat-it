from .alphazero_cnn import AlphaZeroCNN


def create_model(game_config):
    """Factory function to create a model from a config dictionary."""
    return AlphaZeroCNN(**game_config)

# --- Training Configuration ---
basename = "uttt_alphazero_experiment"
set_model_func_name = "set_uttt_model"
fetch_data_func_name = "fetch_uttt_selfplay_data"
