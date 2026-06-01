import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import ctypes
import json
from train.utils import fetch_selfplay_data_from_cpp, GameType, Hyperparameters, DataPointers, CppStats
from train.ttt import create_model
from train.utils import create_inference_model_bundle

def main():
    game_type = GameType.TTT
    possible_paths = [
        os.path.join('build', 'libalphazero.dylib'),
        os.path.join('build', 'libalphazero.so'),
    ]
    lib_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if lib_path is None:
        raise FileNotFoundError("Could not find libalphazero shared library.")
    alphazero_lib = ctypes.CDLL(lib_path)
    
    c_fetch_data_func = alphazero_lib.fetch_selfplay_data
    c_fetch_data_func.restype = None
    c_fetch_data_func.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(DataPointers),
        ctypes.c_uint32,
        ctypes.POINTER(CppStats), ctypes.POINTER(CppStats),
        ctypes.POINTER(CppStats)
    ]
    
    alphazero_lib.create_session.restype = ctypes.c_void_p
    alphazero_lib.create_session.argtypes = [
        ctypes.c_int32, ctypes.c_char_p, ctypes.c_uint32, ctypes.POINTER(Hyperparameters)
    ]

    # Load config to get dims
    with open("train/ttt_config.json", 'r') as f:
        config = json.load(f)
    game_config = config['game_config']
    
    G_SIZE = game_config['input_channels'] * game_config['board_size'] * game_config['board_size']
    self_play_config = config['self_play_config']
    training_hyperparams = config['training_hyperparams']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Init model
    model = create_model(game_config).to(device)
    model.eval()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    value_loss_fn = nn.MSELoss()
    
    # 3. Create Model Bundle and start C++ Session
    print("Creating initial model bundle...")
    model_bytes, _ = create_inference_model_bundle(model, 0, None, game_config, self_play_config, {})

    hp = Hyperparameters(self_play_config)
    
    print("Starting C++ session (Self-Play threads)...")
    session_handle = alphazero_lib.create_session(
        GameType.TTT.value, model_bytes, len(model_bytes), ctypes.byref(hp)
    )

    G_SIZE = game_config['input_channels'] * game_config['board_size'] * game_config['board_size']
    P_SIZE = game_config['num_actions']

    print("Fetching 128 positions from C++...")
    states, policies, values = [], [], []
    for i in range(128):
        new_data, _ = fetch_selfplay_data_from_cpp(
            session_handle, c_fetch_data_func, GameType.TTT,
            1, G_SIZE, P_SIZE
        )
        if new_data and len(new_data['game_states']) > 0:
            states.append(new_data['game_states'][0])
            policies.append(new_data['policy_targets'][0])
            values.append(new_data['value_targets'][0])
        if (i+1) % 10 == 0:
            print(f"Fetched {i+1}/128 positions...")
            
    if len(states) == 0:
        print("Failed to fetch data. Is the C++ engine generating games?")
        return
        
    batch_states = torch.tensor(states, dtype=torch.float32)
    batch_target_policies = torch.tensor(policies, dtype=torch.float32)
    batch_target_values = torch.tensor(values, dtype=torch.float32)

    print(f"Loaded batch size: {batch_states.shape[0]}")
    print("Training on this single batch for 200 epochs to check for overfitting...")

    # Train loop on single batch
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        
        pred_values, pred_policy_logits = model(batch_states)
        
        # Loss calculation exactly as in main.py
        loss_policy = -torch.sum(batch_target_policies * F.log_softmax(pred_policy_logits, dim=1), dim=1).mean()
        loss_value = value_loss_fn(pred_values.squeeze(-1), batch_target_values)
        loss = loss_policy + loss_value
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Total Loss: {loss.item():.4f} | "
                  f"Policy Loss: {loss_policy.item():.4f} | Value Loss: {loss_value.item():.4f}")
            
    print("\nIf the Total Loss does not approach ~0.0, the neural network architecture or loss function is failing to learn.")

if __name__ == "__main__":
    main()
