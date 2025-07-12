import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import ctypes
import io
import time
import threading
import queue
import json
import subprocess
from torch.utils.tensorboard import SummaryWriter

G_SIZE = 27  # 3 planes * 9 cells
P_SIZE = 9   # 9 possible moves

class DataPointers(ctypes.Structure):
    _fields_ = [
        ("game_states", ctypes.POINTER(ctypes.c_float)),
        ("policy_targets", ctypes.POINTER(ctypes.c_float)),
        ("value_targets", ctypes.POINTER(ctypes.c_float)),
        ("player_indices", ctypes.POINTER(ctypes.c_int32)), 
        ]

def set_model(lib, model_data, model_data_len):
    # Define the function signatures for the C++ functions.
    c_set_model = lib.set_ttt_model
    c_set_model.restype = ctypes.c_int
    c_set_model.argtypes = [
        ctypes.c_char_p, # model_data
        ctypes.c_int32, # model_data_len
    ]

    result = c_set_model(model_data, model_data_len)
    if result < 0:
        raise RuntimeError(f"C++ function returned an error code: {result}")
    
def get_selfplay_data_from_cpp(lib, config: dict):
    """
    Calls the C++ shared library to run a self-play game and retrieve the data.
    """
    # Define the function signatures for the C++ functions.
    c_run_selfplay = lib.run_ttt_selfplay
    c_run_selfplay.restype = ctypes.c_int
    c_run_selfplay.argtypes = [
        ctypes.c_int32, # threads
        ctypes.c_int32, # runs
        ctypes.c_float, # c_base
        ctypes.c_float, # c_init
        ctypes.c_float, # dirichlet_alpha
        ctypes.c_float, # dirichlet_epsilon
        ctypes.c_int32, # simulations
        ctypes.c_int32, # opening moves
        ctypes.POINTER(DataPointers) # data_pointers_out
    ]

    # Create an instance of our struct and call the C++ function.
    data_pointers = DataPointers()
    
    num_positions = c_run_selfplay(
        config['threads'],
        config['runs'],
        config['c_base'],
        config['c_init'],
        config['dirichlet_alpha'],
        config['dirichlet_epsilon'],
        config['simulations'],
        config['opening_moves'],
        ctypes.byref(data_pointers)
    )

    if num_positions < 0:
        raise RuntimeError(f"C++ function returned an error code: {num_positions}")

    if num_positions == 0:
        return None

    # Create NumPy arrays that view the memory pointed to by the struct fields.
    game_states = np.ctypeslib.as_array(data_pointers.game_states, shape=(num_positions, G_SIZE))
    policy_targets = np.ctypeslib.as_array(data_pointers.policy_targets, shape=(num_positions, P_SIZE))
    value_targets = np.ctypeslib.as_array(data_pointers.value_targets, shape=(num_positions,))
    player_indices = np.ctypeslib.as_array(data_pointers.player_indices, shape=(num_positions,))
     
    # Return a dictionary of VIEWS into the C++ library's memory buffer.
    # The caller is responsible for copying this data before the next call to this
    # function, as the underlying C++ buffer will be overwritten.
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
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "N/A"

def print_board(state_vector):
    """
    Prints a 3x3 Tic-Tac-Toe board from a serialized state vector.
    The state vector has 3 planes: X pieces, O pieces, and player-to-move.
    """
    board = [' '] * 9

    # Plane 1: 'X' pieces (Player 1)
    for i in range(9):
        if state_vector[i] == 1.0:
            board[i] = 'X'
    
    # Plane 2: 'O' pieces (Player 2)
    for i in range(9):
        if state_vector[i + 9] == 1.0:
            board[i] = 'O'

    # Plane 3: Player-to-move
    # The third plane is all 1.0s if X is to move, all 0.0s if O is to move.
    turn_to_move_symbol = 'X' if state_vector[18] == 1.0 else 'O'

    print(f"Turn to move: {turn_to_move_symbol}")
    print("-------------")
    print(f"| {board[0]} | {board[1]} | {board[2]} |")
    print("-------------")
    print(f"| {board[3]} | {board[4]} | {board[5]} |")
    print("-------------")
    print(f"| {board[6]} | {board[7]} | {board[8]} |")
    print("-------------")

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

        # Pre-allocate memory for the buffer
        self.states = np.zeros((capacity, g_size), dtype=np.float32)
        self.policies = np.zeros((capacity, p_size), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.player_indices = np.zeros((capacity,), dtype=np.int32)

    def add(self, states, policies, values, player_indices):
        """Adds a batch of new experience to the buffer."""
        num_positions = len(states)
        with self.lock:
            if self.ptr + num_positions <= self.capacity:
                # No wrap-around needed
                self.states[self.ptr:self.ptr + num_positions] = states
                self.policies[self.ptr:self.ptr + num_positions] = policies
                self.values[self.ptr:self.ptr + num_positions] = values
                self.player_indices[self.ptr:self.ptr + num_positions] = player_indices
            else:
                # Data wraps around the end of the buffer
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
        """Samples a random batch of experience and moves it to the target device."""
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

class SelfPlayWorker(threading.Thread):
    """
    A dedicated thread to continuously generate self-play data in the background.
    """
    def __init__(self, lib, config, replay_buffer, model_queue):
        super().__init__(daemon=True) # Daemon thread will exit when main program exits
        self.lib = lib
        self.config = config
        self.replay_buffer = replay_buffer
        self.model_queue = model_queue
        self.stop_event = threading.Event()
        # A thread-safe counter to track the number of positions generated by this worker.
        self.generated_positions_count = 0 # Number of positions generated since last check
        self.total_duration = 0.0          # Sum of durations for self-play cycles
        self.num_cycles = 0                # Number of self-play cycles run
        self.counter_lock = threading.Lock()

    def stop(self):
        self.stop_event.set()

    def get_and_reset_stats(self):
        """Atomically reads and resets the generated stats counters."""
        with self.counter_lock:
            positions = self.generated_positions_count
            duration = self.total_duration
            cycles = self.num_cycles
            self.generated_positions_count = 0
            self.total_duration = 0.0
            self.num_cycles = 0
        return positions, duration, cycles

    def run(self):
        print("[Worker] Self-play worker thread started. Waiting for initial model...")
        try:
            # Block here ONLY ONCE to get the initial model from the trainer.
            # This ensures we don't start generating data with an uninitialized model.
            model_bytes = self.model_queue.get(block=True, timeout=60)
            print("[Worker] Received initial model.")
            set_model(self.lib, model_bytes, len(model_bytes))
        except queue.Empty:
            print("[Worker] Timed out waiting for initial model. Stopping.")
            return
        except Exception as e:
            print(f"[Worker] Error receiving initial model: {e}")
            return

        while not self.stop_event.is_set():
            try:
                # 1. Check for an UPDATED model from the trainer (non-blocking).
                try:
                    new_model_bytes = self.model_queue.get_nowait()
                    print("[Worker] Received updated model from trainer.")
                    set_model(self.lib, new_model_bytes, len(new_model_bytes))
                except queue.Empty:
                    # No new model is available, so we continue generating data
                    # with the current model. This is the expected behavior.
                    pass
                
                # 2. Generate one batch of self-play data.
                start_time = time.time()
                new_data = get_selfplay_data_from_cpp(self.lib, self.config)
                duration = time.time() - start_time
                
                if new_data:
                    num_new_positions = len(new_data['game_states'])
                    # Atomically update the counter for the trainer to read.
                    with self.counter_lock:
                        self.generated_positions_count += num_new_positions
                        self.total_duration += duration
                        self.num_cycles += 1

                    # The .copy() is crucial as the underlying buffer in C++ is temporary.
                    self.replay_buffer.add(new_data['game_states'].copy(), new_data['policy_targets'].copy(),
                                           new_data['value_targets'].copy(), new_data['player_indices'].copy())
                    print(f"[Worker] Generated {num_new_positions} positions in {duration:.2f}s. Buffer size: {len(self.replay_buffer)}")
            except Exception as e:
                print(f"[Worker] Error in self-play worker: {e}")
                break
        print("[Worker] Self-play worker thread stopped.")

class ResidualBlock(nn.Module):
    """
    The core building block of a ResNet. It contains a 'skip connection'
    that adds the input of the block to its output. This helps combat
    vanishing gradients and allows for much deeper networks.
    """
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        # The 'skip connection'
        residual = x
        # The main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Add the input to the output of the convolutions
        out += residual
        out = F.relu(out)
        return out

class TicTacToeCNN(nn.Module):
    def __init__(self, input_channels=3, board_size=3, num_actions=9):
        super(TicTacToeCNN, self).__init__()
        self.board_size = board_size
        self.num_actions = num_actions
        self.input_channels = input_channels

        # Shared Body
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 64 filters
        self.bn2 = nn.BatchNorm2d(64)

        # Calculate the flattened size after convolutions
        # For a 3x3 board and the conv layers above, the output size remains 3x3
        # If you add more conv layers or change padding/stride, this needs adjustment.
        self.flattened_size = 64 * board_size * board_size

        # Policy Head
        self.fc_policy1 = nn.Linear(self.flattened_size, 128)
        self.fc_policy2 = nn.Linear(128, num_actions)

        # Value Head
        self.fc_value1 = nn.Linear(self.flattened_size, 128)
        self.fc_value2 = nn.Linear(128, 1) # Single scalar value

    def forward(self, x):
        # The input x from C++ is flat (batch, 18). Reshape it for the Conv2D layers.
        x = x.view(-1, self.input_channels, self.board_size, self.board_size)

        # Shared body
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x))) # if using conv3
        x = x.view(-1, self.flattened_size)  # Flatten for the fully connected layers

        # Policy head
        policy = F.relu(self.fc_policy1(x))
        policy = self.fc_policy2(policy) # Raw logits, softmax will be applied in loss or outside

        # Value head
        value = F.relu(self.fc_value1(x))
        value = torch.tanh(self.fc_value2(value)) # Output between -1 and 1

        return value, policy

class AlphaZeroCNN(nn.Module):
    def __init__(self, board_size, num_actions, input_channels, num_res_blocks, res_block_channels):
        """
        A configurable ResNet-based architecture inspired by AlphaZero.

        Args:
            board_size (int): The width and height of the board.
            num_actions (int): The size of the policy output.
            input_channels (int): Number of input planes. For UTTT, this could be:
            num_res_blocks (int): The number of residual blocks in the network body.
            res_block_channels (int): The number of channels used in the residual blocks.
        """
        super(AlphaZeroCNN, self).__init__()
        self.board_size = board_size
        self.num_actions = num_actions
        self.input_channels = input_channels

        # --- Network Body ---
        # 1. An initial convolutional layer to transform the input planes to the desired channel depth.
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, res_block_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(res_block_channels),
            nn.ReLU(inplace=True)
        )

        # 2. A stack of residual blocks. This is the core of the network.
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(res_block_channels) for _ in range(num_res_blocks)]
        )

        # --- Value Head (as in AlphaZero paper) ---
        self.value_head = nn.Sequential(
            nn.Conv2d(res_block_channels, 1, kernel_size=1, bias=False), # Reduce to 1 channel
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1 * board_size * board_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh() # Output value between -1 and 1
        )

        # --- Policy Head (as in AlphaZero paper) ---
        self.policy_head = nn.Sequential(
            nn.Conv2d(res_block_channels, 2, kernel_size=1, bias=False), # Reduce to 2 channels
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, num_actions)
        )

    def forward(self, x):
        # Input x from C++ is flat. Reshape it for the Conv2D layers.
        x = x.view(-1, self.input_channels, self.board_size, self.board_size)
        
        # Pass through the initial convolution and the residual blocks
        x = self.initial_conv(x)
        x = self.res_blocks(x)

        # The two heads operate on the output of the residual body
        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return value, policy_logits

# 2. Training Setup
if __name__ == '__main__':
    lib_path = os.path.join('obj', 'libalphazero.so')
    alphazero_lib = ctypes.CDLL(lib_path)
    print(f"Successfully loaded shared library from: {lib_path}")
    try:
        # --- Game and Network Configuration ---
        game_config = {
            'board_size': 3,
            'num_actions': 9,
            'input_channels': 3, # X pieces, O pieces, player-to-move
            'num_res_blocks': 1, # A smaller ResNet for a simple game like TTT
            'res_block_channels': 64
        }
        # Training loop hyperparameters
        learning_rate = 0.0005
        batch_size = 64
        log_freq_steps = 100 # How often to log training progress
        total_training_steps = 10000 # Total number of optimization steps
        model_update_freq_steps = 500 # How often to send the new model to the worker

        # Replay Buffer hyperparameters
        replay_buffer_size = 20000 # Max number of positions to store
        min_replay_buffer_size = 1000 # Start training only after this many positions are stored
        target_replay_ratio = 4.0 # S_pos_avg: avg times a position is sampled for training

        # Device
        # Device selection: Prefer MPS on Apple Silicon, then CUDA, then CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("MPS is available! Using Apple Silicon GPU for training.")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("CUDA is available! Using NVIDIA GPU for training.")
        else:
            device = torch.device("cpu")
            print("No GPU backend found. Using CPU for training.")

        # Model
        # model = AlphaZeroCNN(**game_config).to(device)
        model = TicTacToeCNN( game_config['input_channels'], game_config['board_size'], game_config['num_actions']).to(device)

        # A thread-safe queue to pass the latest model from the trainer to the worker
        model_queue = queue.Queue(maxsize=1)

        # Replay Buffer
        replay_buffer = ReplayBuffer(
            replay_buffer_size, G_SIZE, P_SIZE, device)

        # --- TensorBoard Setup ---
        # Find a unique directory for this training run to avoid overwriting logs.
        # The first run will be '..._experiment', the next '..._experiment_1', etc.
        base_log_dir = 'runs/ttt_alphazero_experiment'
        if not os.path.exists(base_log_dir):
            log_dir = base_log_dir
        else:
            counter = 1
            while os.path.exists(f"{base_log_dir}_{counter}"):
                counter += 1
            log_dir = f"{base_log_dir}_{counter}"
        
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs will be saved to: {writer.log_dir}")
        # Log the model graph
        writer.add_graph(model, torch.randn(1, G_SIZE).to(device))

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Configuration for the self-play run
        self_play_config = {
            'threads': 8,
            'runs': 100, # Number of games to generate per worker cycle
            'c_base': 19652.0,
            'c_init': 1.25,
            'dirichlet_alpha': 0.3,
            'dirichlet_epsilon': 0.25,
            'simulations': 100,
            'opening_moves': 0
        }

        # --- Start the Self-Play Worker Thread ---
        worker = SelfPlayWorker(alphazero_lib, self_play_config, replay_buffer, model_queue)
        worker.start()

        # --- Main Training Loop ---
        print("\n[Trainer] Starting main training loop.")
        
        # Put the initial random model on the queue for the worker to start generating data
        model.eval()
        scripted_model = torch.jit.script(model)
        buffer = io.BytesIO()
        torch.jit.save(scripted_model, buffer)
        model_queue.put(buffer.getvalue())

        # Wait for the replay buffer to have a minimum number of samples
        while len(replay_buffer) < min_replay_buffer_size:
            print(f"[Trainer] Waiting for replay buffer to fill... {len(replay_buffer)}/{min_replay_buffer_size}")
            time.sleep(2)

        print("\n[Trainer] Buffer filled. Starting training steps.")
        value_loss_fn = nn.MSELoss()
        
        step = 0
        while step < total_training_steps:
            # Wait for the worker to generate some new data
            num_new_positions, total_duration, num_cycles = worker.get_and_reset_stats()
            if num_new_positions == 0:
                time.sleep(0.1) # Avoid busy-waiting while worker is busy
                continue

            # Calculate how many training steps to perform for this new data
            # to maintain the target replay ratio.
            steps_to_run = max(1, int((target_replay_ratio * num_new_positions) / batch_size))
            print(f"[Trainer] Worker generated {num_new_positions} positions. Running {steps_to_run} training steps.")

            for _ in range(steps_to_run):
                if step >= total_training_steps:
                    break

                model.train()
                start_time = time.time()

                batch_states, batch_target_policies, batch_target_values = replay_buffer.sample(batch_size)

                optimizer.zero_grad()
                pred_values, pred_policy_logits = model(batch_states)

                loss_policy = -torch.sum(batch_target_policies * F.log_softmax(pred_policy_logits, dim=1), dim=1).mean()
                loss_value = value_loss_fn(pred_values.squeeze(-1), batch_target_values)
                loss = loss_policy + loss_value

                loss.backward()
                optimizer.step()
                duration = time.time() - start_time

                # Log metrics to TensorBoard periodically
                if (step + 1) % log_freq_steps == 0:
                    avg_selfplay_time = total_duration / num_cycles if num_cycles > 0 else 0.0

                    writer.add_scalar('Loss/Total', loss.item(), step)
                    writer.add_scalar('Loss/Policy', loss_policy.item(), step)
                    writer.add_scalar('Loss/Value', loss_value.item(), step)
                    writer.add_scalar('Performance/Training_Step_Time_ms', duration * 1000, step)
                    writer.add_scalar('Performance/Avg_SelfPlay_Time_s', avg_selfplay_time, step)
                    print(f"[Trainer] Step {step+1}/{total_training_steps} | Loss: {loss.item():.4f} | Step Time: {duration*1000:.2f}ms")

                # Periodically update the model for the self-play worker
                if (step + 1) % model_update_freq_steps == 0:
                    print(f"\n[Trainer] Step {step+1}: Updating model for self-play worker.")
                    model.eval()
                    scripted_model = torch.jit.script(model)
                    buffer = io.BytesIO()
                    torch.jit.save(scripted_model, buffer)
                    with model_queue.mutex:
                        model_queue.queue.clear()
                    model_queue.put_nowait(buffer.getvalue())

                step += 1

        print("\nTraining finished")

        # --- Save the final trained model with embedded metadata ---
        # Use the TensorBoard log directory name to create a unique model filename,
        # linking the model to its training run logs.
        final_model_path = os.path.join("models", f"{os.path.basename(writer.log_dir)}.pt")
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        print(f"\nSaving final trained model to {final_model_path}...")

        # 1. Gather all relevant metadata into a dictionary.
        metadata = {
            "model_architecture": model.__class__.__name__,
            "training_steps": step,
            "final_loss": loss.item() if 'loss' in locals() else None,
            "hyperparameters": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "total_training_steps": total_training_steps,
                "model_update_freq_steps": model_update_freq_steps,
                "replay_buffer_size": replay_buffer_size,
                "min_replay_buffer_size": min_replay_buffer_size,
                "target_replay_ratio": target_replay_ratio,
            },
            "self_play_config": self_play_config,
            "game_config": game_config,
            "git_revision": get_git_revision_hash(),
            "save_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # 2. Convert the metadata to a JSON string.
        metadata_json = json.dumps(metadata, indent=4)
        extra_files = {'metadata.json': metadata_json}
        print("Bundling metadata with the model:")
        print(metadata_json)

        # 3. Save the scripted model with the metadata embedded in the archive.
        # The C++ libtorch backend can still load this file directly, ignoring the extra data.
        model.eval()
        final_scripted_model = torch.jit.script(model)
        final_scripted_model.save(final_model_path, _extra_files=extra_files)
        print("Model saved successfully.")

        # Close the TensorBoard writer
        writer.close()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # --- Explicit C++ Resource Cleanup ---
        # This is crucial to prevent crashes on exit. We must explicitly tell the
        # C++ library to shut down its background threads and release resources
        # before the Python interpreter unloads the library.
        if 'worker' in locals() and worker.is_alive():
            print("\n[Trainer] Stopping self-play worker...")
            worker.stop()
            worker.join()

        print("\nCleaning up C++ resources...")
        cleanup_func = alphazero_lib.cleanup_resources
        cleanup_func.restype = None
        cleanup_func.argtypes = []
        cleanup_func()
        print("C++ cleanup complete.")
