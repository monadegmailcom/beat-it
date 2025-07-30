import torch
import numpy as np
import ctypes
import io
import os
import time
import json
import subprocess
import threading
import matplotlib.pyplot as plt
from PIL import Image


class DataPointers(ctypes.Structure):
    _fields_ = [
        ("game_states", ctypes.POINTER(ctypes.c_float)),
        ("policy_targets", ctypes.POINTER(ctypes.c_float)),
        ("value_targets", ctypes.POINTER(ctypes.c_float)),
        ("player_indices", ctypes.POINTER(ctypes.c_int32)),
    ]


def set_model(
        set_model_func, model_data: bytes,
        model_data_len: int, metadata_json_bytes: bytes):
    """Generic function to set a model in the C++ library."""
    result: int = set_model_func(
        model_data, model_data_len, metadata_json_bytes,
        len(metadata_json_bytes))
    if result < 0:
        raise RuntimeError(
            f"C++ set_model function returned an error code: {result}")


def fetch_selfplay_data_from_cpp(
        fetch_data_func, number_of_positions: int, g_size: int, p_size: int):
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

    queue_size = fetch_data_func(data_pointers, number_of_positions)

    if queue_size < 0:
        raise RuntimeError(f"C++ fetch function returned an error code: \
            {queue_size}")

    result = {
        "game_states": game_states,
        "policy_targets": policy_targets,
        "value_targets": value_targets,
        "player_indices": player_indices
    }
    return result, queue_size


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
        model, optimizer, step, current_loss, game_config,
        self_play_config, training_hyperparams, path):
    """Saves a full checkpoint including model, metadata, and optimizer state
       to a file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    model_bytes, metadata_json = create_inference_model_bundle(
        model, step, current_loss, game_config, self_play_config,
        training_hyperparams
    )
    optimizer_state_buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), optimizer_state_buffer)
    loaded_model = torch.jit.load(io.BytesIO(model_bytes))

    torch.jit.save(loaded_model, path, _extra_files={
        'metadata.json': metadata_json,
        'optimizer_state.pt': optimizer_state_buffer.getvalue()
    })


def log_histogram_as_image(writer, tag, data, step):
    """Creates a bar chart from histogram data and logs it as an image."""
    try:
        fig, ax = plt.subplots()
        indices = np.where(data > 0)[0]
        counts = data[indices]
        if len(indices) > 0:
            ax.bar(indices, counts, tick_label=indices)
            ax.set_xlabel("Inference Batch Size")
            ax.set_ylabel("Frequency (Count)")
            ax.set_title("Inference Batch Size Distribution")
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf)
            writer.add_image(tag, np.array(image), step, dataformats='HWC')
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Failed to generate histogram image: {e}")


class ReplayBuffer:
    """
    A fixed-size circular buffer to store self-play experience for training.
    It uses pre-allocated NumPy arrays for efficiency.
    """
    def __init__(self, capacity, g_size, p_size, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.lock = threading.Lock()
        self.states = np.zeros((capacity, g_size), dtype=np.float32)
        self.policies = np.zeros((capacity, p_size), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.player_indices = np.zeros((capacity,), dtype=np.int32)

    def add(self, states, policies, values, player_indices):
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
            indices = np.random.randint(0, self.size, size=batch_size)
            return (
                torch.from_numpy(self.states[indices]).to(self.device),
                torch.from_numpy(self.policies[indices]).to(self.device),
                torch.from_numpy(self.values[indices]).to(self.device)
            )

    def __len__(self):
        with self.lock:
            return self.size
