import ctypes
import os
import json
import numpy as np
import time
from train.utils import fetch_selfplay_data_from_cpp, GameType, Hyperparameters, DataPointers, CppStats
from train.uttt import create_model
from train.utils import create_inference_model_bundle

print("Starting diagnose script...")

possible_paths = [os.path.join('build', 'libalphazero.dylib'), os.path.join('build', 'libalphazero.so')]
lib_path = next((p for p in possible_paths if os.path.exists(p)), None)
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

with open('train/uttt_config.json', 'r') as f:
    config = json.load(f)
game_config = config['game_config']
self_play_config = config['self_play_config']

model = create_model(game_config)
model_bytes, _ = create_inference_model_bundle(model, 0, None, game_config, self_play_config, {})

hp = Hyperparameters(self_play_config)
session_handle = alphazero_lib.create_session(
    GameType.UTTT.value, model_bytes, len(model_bytes), ctypes.byref(hp)
)
print("Started self play session. Fetching data...")

G_SIZE = game_config['input_channels'] * game_config['board_size'] * game_config['board_size']
P_SIZE = game_config['num_actions']

try:
    new_data, _ = fetch_selfplay_data_from_cpp(session_handle, c_fetch_data_func, GameType.UTTT, 1, G_SIZE, P_SIZE)
    print("Fetch completed.", len(new_data['value_targets']))
    if new_data:
        val = new_data['value_targets'][0]
        print('Value target:', val)
        pol = new_data['policy_targets'][0]
        print('Policy target sum:', sum(pol))
        state = new_data['game_states'][0].reshape(4, 9, 9)
        print("Plane 1 sum:", state[0].sum())
        print("Plane 2 sum:", state[1].sum())
        print("Plane 3 sum:", state[2].sum())
        print("Plane 4 (0,0):", state[3,0,0])
except Exception as e:
    print("Exception occurred:", e)
