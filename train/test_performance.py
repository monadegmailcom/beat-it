import argparse
import json
import logging
import os
import sys
import time
import ctypes

import torch
import torch.nn.functional as F
import torch.optim as optim

from .alphazero_cnn import AlphaZeroCNN
from .utils import (
    pause_session, resume_session, fetch_selfplay_data_from_cpp, GameType, ReplayBuffer
)

def main():
    parser = argparse.ArgumentParser(description="Performance test for AlphaZero")
    parser.add_argument("--game", type=str, default="uttt", help="Game to train (e.g. ttt, uttt)")
    parser.add_argument("--steps", type=int, default=15, help="Number of training steps to run")
    parser.add_argument("--mode", type=str, choices=["baseline", "paused", "dedicated_stream"], default="baseline",
                        help="Which mode to test")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    args = parser.parse_args()

    game_type = GameType[args.game.upper()]
    config_file = "train/uttt_config.json" if args.game == "uttt" else "train/ttt_config.json"
    with open(config_file, "r") as f:
        config = json.load(f)

    # Force configurations for the test to ensure we hit the 2048 bottleneck scenario
    config["self_play_config"]["max_batch_size"] = 2048
    if args.mode == "dedicated_stream":
        config["self_play_config"]["use_dedicated_cuda_stream"] = True
    else:
        config["self_play_config"]["use_dedicated_cuda_stream"] = False

    possible_paths = [
        os.path.join('build', 'libalphazero.dylib'),
        os.path.join('build', 'libalphazero.so'),
    ]
    lib_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if lib_path is None:
        raise FileNotFoundError(f"Could not find libalphazero shared library. Checked: {possible_paths}")
    alphazero_lib = ctypes.CDLL(lib_path)

    from .utils import DataPointers, CppStats
    alphazero_lib.create_session.argtypes = [ctypes.c_int32, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_void_p]
    alphazero_lib.create_session.restype = ctypes.c_void_p
    alphazero_lib.fetch_selfplay_data.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(DataPointers),
        ctypes.c_uint32,
        ctypes.POINTER(CppStats), ctypes.POINTER(CppStats),
        ctypes.POINTER(CppStats), ctypes.POINTER(CppStats),
        ctypes.POINTER(CppStats)
    ]
    alphazero_lib.fetch_selfplay_data.restype = None
    alphazero_lib.pause_session.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    alphazero_lib.resume_session.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    alphazero_lib.destroy_session.argtypes = [ctypes.c_int32, ctypes.c_void_p]
    c_fetch_data_func = alphazero_lib.fetch_selfplay_data

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = AlphaZeroCNN(**config["game_config"]).to(device)
    scripted_model = torch.jit.script(model)
    metadata = {
        "self_play_config": config["self_play_config"]
    }
    model_buffer = scripted_model.save_to_buffer({
        "metadata.json": json.dumps(metadata).encode('utf-8')
    })

    # Setup dimension sizes depending on game type
    if args.game == "ttt":
        G_SIZE = 18
        P_SIZE = 9
    else:
        G_SIZE = 324
        P_SIZE = 81

    from .utils import Hyperparameters
    hp = Hyperparameters(config["self_play_config"])
    session_handle = alphazero_lib.create_session(game_type.value, model_buffer, len(model_buffer), ctypes.byref(hp))
    replay_buffer = ReplayBuffer(10000, G_SIZE, P_SIZE, device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    print(f"\n--- Running Performance Test: Mode {args.mode.upper()} ---")
    
    total_step_time = 0
    total_selfplay_time = 0
    total_positions = 0
    
    # We fetch a fixed number per step to keep timing consistent
    num_positions_to_fetch = 256
    
    for step in range(args.steps):
        # 1. Fetch data
        fetch_start = time.time()
        new_data, _ = fetch_selfplay_data_from_cpp(
            session_handle, c_fetch_data_func, game_type, 
            num_positions_to_fetch, G_SIZE, P_SIZE
        )
        fetch_duration = time.time() - fetch_start
        
        if new_data:
            replay_buffer.add(new_data)
        
        if replay_buffer.size < args.batch_size:
            continue

        # 2. Train
        train_start = time.time()
        model.train()
        
        if args.mode == "paused":
            pause_session(session_handle, alphazero_lib, game_type)

        batch_states, batch_target_policies, batch_target_values = replay_buffer.sample(args.batch_size)
        batch_states = batch_states.to(device)
        batch_target_policies = batch_target_policies.to(device)
        batch_target_values = batch_target_values.to(device)

        optimizer.zero_grad()
        pred_values, pred_policy_logits = model(batch_states)
        loss = F.cross_entropy(pred_policy_logits, batch_target_policies) + F.mse_loss(pred_values.squeeze(-1), batch_target_values)
        loss.backward()
        optimizer.step()

        if args.mode == "paused":
            resume_session(session_handle, alphazero_lib, game_type)
            
        step_duration = time.time() - train_start
        
        # We skip the first step as warmup
        if step > 0:
            total_step_time += step_duration
            total_selfplay_time += fetch_duration
            total_positions += len(new_data) if new_data else 0
            
        print(f"Step {step+1}/{args.steps} | Step Time: {step_duration*1000:7.2f}ms | Fetch Time: {fetch_duration*1000:7.2f}ms | Positions: {len(new_data) if new_data else 0}")

    if args.steps > 1:
        avg_step_time = (total_step_time / (args.steps - 1)) * 1000
        avg_selfplay_time = (total_selfplay_time / (args.steps - 1)) * 1000
        total_time = total_step_time + total_selfplay_time
        pos_per_sec = total_positions / total_time if total_time > 0 else 0
        
        print("\n--- Summary ---")
        print(f"Mode:               {args.mode.upper()}")
        print(f"Avg Step Time:      {avg_step_time:.2f} ms")
        print(f"Avg Fetch Time:     {avg_selfplay_time:.2f} ms")
        print(f"Overall Throughput: {pos_per_sec:.2f} pos/sec")

    # Cleanup
    alphazero_lib.destroy_session(game_type.value, session_handle)

if __name__ == "__main__":
    main()
