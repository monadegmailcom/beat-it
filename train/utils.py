import torch
import numpy as np
import ctypes
import io
import os
import time
import json
import collections
import subprocess
from typing import TypedDict
import threading
from enum import IntEnum

train_buffer_metadata_file = 'train_buffer_metadata.json'
validation_buffer_metadata_file = 'validation_buffer_metadata.json'


class GameType(IntEnum):
    TTT = 1
    UTTT = 2


class DataPointers(ctypes.Structure):
    _fields_ = [
        ("game_states", ctypes.POINTER(ctypes.c_float)),
        ("policy_targets", ctypes.POINTER(ctypes.c_float)),
        ("value_targets", ctypes.POINTER(ctypes.c_float)),
        ("player_indices", ctypes.POINTER(ctypes.c_int32)),
    ]


class CppStats(ctypes.Structure):
    """Mirrors the Statistics class in C++ for data transfer."""
    _fields_ = [
        ("min", ctypes.c_float),
        ("max", ctypes.c_float),
        ("mean", ctypes.c_float),
        ("stddev", ctypes.c_float),
    ]


class Hyperparameters(ctypes.Structure):
    """Mirrors the Hyperparameters struct in C++ for data transfer."""
    _fields_ = [
        ("c_base", ctypes.c_float),
        ("c_init", ctypes.c_float),
        ("dirichlet_alpha", ctypes.c_float),
        ("dirichlet_epsilon", ctypes.c_float),
        ("simulations", ctypes.c_size_t),
        ("opening_moves", ctypes.c_size_t),
        ("parallel_games", ctypes.c_size_t),
        ("parallel_simulations", ctypes.c_size_t),
        ("max_batch_size", ctypes.c_size_t),
        ("nodes_per_block", ctypes.c_size_t),
    ]

    def __init__(self, config):
        self.c_base = config.get('c_base', 19652.0)
        self.c_init = config.get('c_init', 1.25)
        self.dirichlet_alpha = config.get('dirichlet_alpha', 0.3)
        self.dirichlet_epsilon = config.get('dirichlet_epsilon', 0.25)
        self.simulations = config.get('simulations', 800)
        self.opening_moves = config.get('opening_moves', 30)
        self.parallel_games = config.get('parallel_games', 1)
        self.parallel_simulations = config.get('parallel_simulations', 4)
        self.max_batch_size = config.get('max_batch_size', 1024)
        
        # Default nodes_per_block logic from C++
        default_nodes_per_block = 50 * self.simulations
        self.nodes_per_block = config.get(
            'nodes_per_block', default_nodes_per_block)


class TrainingHyperparameters(TypedDict):
    """A TypedDict to enforce the structure of training hyperparameters."""
    learning_rate: float
    weight_decay: float
    batch_size: int
    log_freq_steps: int
    total_training_steps: int
    model_update_freq_steps: int
    checkpoint_freq_steps: int
    replay_buffer_size: int
    min_replay_buffer_size: int
    target_replay_ratio: float
    validation_split_percentage: float
    validation_freq_steps: int
    lr_schedule_milestones: list[int]
    lr_schedule_gamma: float


def set_model(session_handle, set_model_func, game_type: GameType,
              model_data: bytes):
    """Generic function to set a model in the C++ library."""
    inference_batch_size = CppStats()
    inference_time = CppStats()
    allocator_size = CppStats()
    set_model_func(
        session_handle, game_type, model_data, len(model_data),
        ctypes.byref(inference_batch_size),
        ctypes.byref(inference_time),
        ctypes.byref(allocator_size))
    return inference_batch_size, inference_time, allocator_size


def fetch_selfplay_data_from_cpp(
        session_handle, fetch_data_func, game_type: GameType,
        number_of_positions: int, g_size: int, p_size: int):
    """
    Generic function to fetch self-play data from the C++ library. Handles
    memory allocation and data transfer.
    """
    if number_of_positions <= 0:
        return None, 0

    game_states = np.zeros((number_of_positions, g_size), dtype=np.float32)
    policy_targets = np.zeros((number_of_positions, p_size), dtype=np.float32)
    value_targets = np.zeros(number_of_positions, dtype=np.float32)
    player_indices = np.zeros(number_of_positions, dtype=np.int32)

    data_pointers = DataPointers()
    data_pointers.game_states = game_states.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float))
    data_pointers.policy_targets = policy_targets.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float))
    data_pointers.value_targets = value_targets.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float))
    data_pointers.player_indices = player_indices.ctypes.data_as(
        ctypes.POINTER(ctypes.c_int32))

    fetch_data_func(
        session_handle, game_type, ctypes.byref(data_pointers), 
        number_of_positions)

    result = {
        "game_states": game_states,
        "policy_targets": policy_targets,
        "value_targets": value_targets,
        "player_indices": player_indices
    }
    return result


def get_git_revision_hash() -> str:
    """Retrieves the current git commit hash."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "N/A"


def create_inference_model_bundle(
        model, step, current_loss, game_config, self_play_config,
        training_hyperparams):
    """Gathers metadata and saves the scripted model to an in-memory buffer."""
    metadata = {
        "model_architecture": model.__class__.__name__,
        "training_steps": step,
        "loss": current_loss.item() if current_loss is not None else None,
        "hyperparameters": training_hyperparams,
        "self_play_config": self_play_config,
        "game_config": game_config,
        "git_revision": get_git_revision_hash(),
        "save_timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metadata_json = json.dumps(metadata, indent=4)
    model.eval()
    scripted_model = torch.jit.script(model)
    buffer = io.BytesIO()
    torch.jit.save(scripted_model, buffer)
    return buffer.getvalue(), metadata_json


def save_checkpoint(
        model, optimizer, scheduler, step, current_loss, game_config,
        self_play_config, training_hyperparams: TrainingHyperparameters, path,
        train_buffer, validation_buffer):
    """Saves a full checkpoint including model, metadata, train buffer,
       validation buffer, and optimizer state to a file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    model_bytes, metadata_json = create_inference_model_bundle(
        model, step, current_loss, game_config, self_play_config,
        training_hyperparams
    )
    optimizer_state_buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), optimizer_state_buffer)
    scheduler_state_buffer = io.BytesIO()
    torch.save(scheduler.state_dict(), scheduler_state_buffer)

    loaded_model = torch.jit.load(io.BytesIO(model_bytes))

    extra_files_dict = {
        'metadata.json': metadata_json,
        'optimizer_state.pt': optimizer_state_buffer.getvalue(),
        'scheduler_state.pt': scheduler_state_buffer.getvalue()
    }

    if train_buffer:
        train_buffer_bytes = train_buffer.save_to_bytes()
        extra_files_dict['train_buffer_data.npz'] = train_buffer_bytes['data']
        extra_files_dict['train_buffer_metadata.json'] = \
            train_buffer_bytes['metadata']

    if validation_buffer:
        val_buffer_bytes = validation_buffer.save_to_bytes()
        extra_files_dict['validation_buffer_data.npz'] = \
            val_buffer_bytes['data']
        extra_files_dict['validation_buffer_metadata.json'] = \
            val_buffer_bytes['metadata']

    torch.jit.save(loaded_model, path, _extra_files=extra_files_dict)


def split_and_add_data(
        data, train_buffer, validation_buffer, validation_split_percentage):
    """Splits the fetched data into training and validation sets and adds it
       to the respective buffers.
    """
    if not data:
        return

    num_positions = len(data['game_states'])
    split_idx = int(num_positions * (1 - validation_split_percentage))

    train_buffer.add({k: v[:split_idx] for k, v in data.items()})
    validation_buffer.add({k: v[split_idx:] for k, v in data.items()})


class MetricLogger:
    """A helper class to accumulate and log training metrics."""
    def __init__(self, writer):
        self.writer = writer
        self.metrics = collections.defaultdict(float)
        self.counts = collections.defaultdict(int)
        self.inference_batch_size_stats = CppStats()
        self.inference_time_stats = CppStats()
        self.allocator_size_stats = CppStats()

    def update(self, **kwargs):
        """Update metrics for the current step."""
        for key, value in kwargs.items():
            if isinstance(value, CppStats):
                if key == 'inference_batch_size':
                    self.inference_batch_size_stats = value
                elif key == 'inference_time':
                    self.inference_time_stats = value
                elif key == 'allocator_size':
                    self.allocator_size_stats = value
            else:
                self.metrics[key] += value
                self.counts[key] += 1

    def log_and_reset(
            self, step: int, total_steps: int, replay_buffer_len: int,
            lr: float):
        """Averages metrics, logs to console and TensorBoard, then resets."""
        if self.counts['step_time_ms'] == 0:
            return  # Avoid division by zero if no steps were logged

        avg_loss = self.metrics['loss_total'] / self.counts['loss_total']
        avg_policy_loss =\
            self.metrics['loss_policy'] / self.counts['loss_policy']
        avg_value_loss = self.metrics['loss_value'] / self.counts['loss_value']
        avg_step_time_ms =\
            self.metrics['step_time_ms'] / self.counts['step_time_ms']
        avg_selfplay_time_ms =\
            self.metrics['selfplay_time_ms'] / self.counts['selfplay_time_ms']

        # Log to TensorBoard
        self.writer.add_scalar('Loss/Total', avg_loss, step)
        self.writer.add_scalar('Loss/Policy', avg_policy_loss, step)
        self.writer.add_scalar('Loss/Value', avg_value_loss, step)
        self.writer.add_scalar(
            'Performance/Training_Step_Time_ms', avg_step_time_ms, step)
        self.writer.add_scalar(
            'Performance/SelfPlay_Fetch_Time_ms', avg_selfplay_time_ms, step)
        self.writer.add_scalar(
            'Buffer/ReplayBuffer_Size', replay_buffer_len, step)

        self.writer.add_scalar('Hyperparameters/Learning_Rate', lr, step)

        # Log C++ stats
        for stats, name in [
            (self.inference_batch_size_stats, 'Inference_Batch_Size'),
            (self.inference_time_stats, 'Inference_Time_us'),
            (self.allocator_size_stats, 'Allocator_SizeBytes')
        ]:
            self.writer.add_scalars(
                f'Performance/{name}',
                {
                    'min': stats.min,
                    'max': stats.max,
                    'mean': stats.mean,
                    'minus_one_stddev': stats.mean - stats.stddev,
                    'plus_one_stddev': stats.mean + stats.stddev
                },
                step)

        # Log to console
        print(
            f"Step {step+1}/{total_steps} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Avg Step Time: {avg_step_time_ms:.2f}ms | "
            f"Avg Selfplay Time: {avg_selfplay_time_ms:.2f}ms"
        )

        # Reset for the next logging window
        self.metrics.clear()
        self.counts.clear()


class ReplayBuffer:
    """
    A fixed-size circular buffer to store self-play experience for training.
    It uses pre-allocated NumPy arrays for efficiency.
    """
    def __init__(self, capacity, g_size, p_size, device, seed=None):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.lock = threading.Lock()
        self.states = np.zeros((capacity, g_size), dtype=np.float32)
        self.policies = np.zeros((capacity, p_size), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.player_indices = np.zeros((capacity,), dtype=np.int32)
        self.g_size = g_size
        self.p_size = p_size
        self.rng = np.random.default_rng(seed)

    def add(self, data_dict):
        """Adds a batch of new experience to the buffer."""
        states, policies, values, player_indices = \
            data_dict['game_states'], data_dict['policy_targets'], \
            data_dict['value_targets'], data_dict['player_indices']
        """Adds a batch of new experience to the buffer."""
        num_positions = len(states)
        with self.lock:
            if self.ptr + num_positions <= self.capacity:
                self.states[self.ptr:self.ptr + num_positions] = states
                self.policies[self.ptr:self.ptr + num_positions] = policies
                self.values[self.ptr:self.ptr + num_positions] = \
                    values
                self.player_indices[self.ptr:self.ptr + num_positions] = \
                    player_indices
            else:
                space_left = self.capacity - self.ptr
                self.states[self.ptr:] = states[:space_left]
                self.policies[self.ptr:] = policies[:space_left]
                self.values[self.ptr:] = values[:space_left]
                self.player_indices[self.ptr:] = player_indices[:space_left]
                remaining = num_positions - space_left
                self.states[:remaining] = states[space_left:]
                self.policies[:remaining] = policies[space_left:]
                self.values[:remaining] = values[space_left:]
                self.player_indices[:remaining] = player_indices[space_left:]
            self.ptr = (self.ptr + num_positions) % self.capacity
            self.size = min(self.size + num_positions, self.capacity)

    def sample(self, batch_size):
        """Samples a random batch of experience and moves it to the target
           device.
        """
        with self.lock:
            indices = self.rng.integers(0, self.size, size=batch_size)

            return (
                torch.from_numpy(self.states[indices]).to(self.device),
                torch.from_numpy(self.policies[indices]).to(self.device),
                torch.from_numpy(self.values[indices]).to(self.device)
            )

    def __len__(self):
        with self.lock:
            return self.size

    def save_to_bytes(self):
        """Serializes the buffer's state and data into byte dictionaries."""
        with self.lock:
            # Use np.savez_compressed for efficient storage
            data_buffer = io.BytesIO()
            np.savez_compressed(
                data_buffer,
                states=self.states,
                policies=self.policies,
                values=self.values,
                player_indices=self.player_indices
            )
            data_buffer.seek(0)

            metadata = {
                'ptr': self.ptr,
                'size': self.size,
                'capacity': self.capacity,
                'g_size': self.g_size,
                'p_size': self.p_size
            }
            metadata_bytes = json.dumps(metadata).encode('utf-8')

            return {
                'data': data_buffer.getvalue(),
                'metadata': metadata_bytes
            }

    def load_from_bytes(self, data_bytes, metadata_bytes):
        """Loads the buffer's state from byte strings."""
        metadata = json.loads(metadata_bytes.decode('utf-8'))

        if metadata['capacity'] != self.capacity or \
           metadata['g_size'] != self.g_size or \
           metadata['p_size'] != self.p_size:
            print("Warning: Replay buffer configuration mismatch. "
                  "Not loading buffer data.")
            return False

        with self.lock:
            data_buffer = io.BytesIO(data_bytes)
            data = np.load(data_buffer, allow_pickle=False)

            self.states[:] = data['states']
            self.policies[:] = data['policies']
            self.values[:] = data['values']
            self.player_indices[:] = data['player_indices']
            self.ptr = metadata['ptr']
            self.size = metadata['size']
        print(f"Successfully loaded replay buffer with {self.size} elements.")
        return True
