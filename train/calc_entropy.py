import torch
from train.test_overfit import *

def main():
    game_type = GameType.UTTT
    possible_paths = [os.path.join('build', 'libalphazero.dylib'), os.path.join('build', 'libalphazero.so')]
    lib_path = next((p for p in possible_paths if os.path.exists(p)), None)
    alphazero_lib = ctypes.CDLL(lib_path)
    c_fetch_data_func = alphazero_lib.fetch_selfplay_data
    c_fetch_data_func.restype = None
    c_fetch_data_func.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(DataPointers), ctypes.c_uint32, ctypes.POINTER(CppStats), ctypes.POINTER(CppStats), ctypes.POINTER(CppStats)]
    alphazero_lib.create_session.restype = ctypes.c_void_p
    alphazero_lib.create_session.argtypes = [ctypes.c_int32, ctypes.c_char_p, ctypes.c_uint32, ctypes.POINTER(Hyperparameters)]
    
    with open("train/uttt_config.json", 'r') as f:
        config = json.load(f)
    game_config = config['game_config']
    G_SIZE = game_config['input_channels'] * game_config['board_size'] * game_config['board_size']
    P_SIZE = game_config['num_actions']
    model = create_model(game_config)
    self_play_config = config['self_play_config']
    hp = Hyperparameters(self_play_config)
    model_bytes, _ = create_inference_model_bundle(model, 0, None, game_config, self_play_config, {})
    session_handle = alphazero_lib.create_session(GameType.UTTT.value, model_bytes, len(model_bytes), ctypes.byref(hp))
    
    new_data, _ = fetch_selfplay_data_from_cpp(session_handle, c_fetch_data_func, game_type, 128, G_SIZE, P_SIZE)
    batch_target_policies = torch.tensor(new_data['policy_targets'], dtype=torch.float32)
    # add small epsilon to avoid log(0)
    eps = 1e-9
    entropy = -torch.sum(batch_target_policies * torch.log(batch_target_policies + eps), dim=1).mean()
    print("Theoretical Minimum Policy Loss (Shannon Entropy):", entropy.item())

if __name__ == "__main__":
    main()
