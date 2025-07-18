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
import matplotlib.pyplot as plt
from PIL import Image

G_SIZE = 27  # 3 planes * 9 cells
P_SIZE = 9   # 9 possible moves

class DataPointers(ctypes.Structure):
    _fields_ = [
        ("game_states", ctypes.POINTER(ctypes.c_float)),
        ("policy_targets", ctypes.POINTER(ctypes.c_float)),
        ("value_targets", ctypes.POINTER(ctypes.c_float)),
        ("player_indices", ctypes.POINTER(ctypes.c_int32)), 
        ]

def set_model(lib, model_data, model_data_len, metadata_json_bytes):
    # Define the function signatures for the C++ functions.
    c_set_model = lib.set_ttt_model
    c_set_model.restype = ctypes.c_int
    c_set_model.argtypes = [
        ctypes.c_char_p, # model_data
        ctypes.c_int32, # model_data_len
        ctypes.c_char_p, # metadata_json
        ctypes.c_int32,  # metadata_len
    ]

    result = c_set_model(model_data, model_data_len, metadata_json_bytes, len(metadata_json_bytes))
    if result < 0:
        raise RuntimeError(f"C++ function returned an error code: {result}")
    
def fetch_selfplay_data_from_cpp(lib, number_of_positions: int):
    """
    Blocks until the C++ library's self-play workers have produced the
    requested number of positions, then fetches them.
    """
    # 1. Define C function signature
    c_fetch_data = lib.fetch_ttt_selfplay_data
    c_fetch_data.restype = ctypes.c_int
    c_fetch_data.argtypes = [
        DataPointers, # Pass struct by value
        ctypes.c_int32
    ]

    if number_of_positions <= 0:
        return None, 0

    # 2. Allocate NumPy arrays in Python to hold the incoming data
    game_states = np.zeros((number_of_positions, G_SIZE), dtype=np.float32)
    policy_targets = np.zeros((number_of_positions, P_SIZE), dtype=np.float32)
    value_targets = np.zeros(number_of_positions, dtype=np.float32)
    player_indices = np.zeros(number_of_positions, dtype=np.int32)

    # 3. Create DataPointers struct and populate with pointers to NumPy data buffers
    data_pointers = DataPointers()
    data_pointers.game_states = game_states.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    data_pointers.policy_targets = policy_targets.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    data_pointers.value_targets = value_targets.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    data_pointers.player_indices = player_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    # 4. Call the C++ function, which will block until data is ready and fill our arrays.
    queue_size = c_fetch_data(data_pointers, number_of_positions)

    if queue_size < 0:
        raise RuntimeError(f"C++ fetch function returned an error code: {queue_size}")

    # Return a dictionary of the now-filled NumPy arrays.
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
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "N/A"

def save_model_with_metadata(model, step, current_loss, game_config, self_play_config, training_hyperparams):
    """Gathers metadata and saves the scripted model to an in-memory buffer."""
    # 1. Gather all relevant metadata into a dictionary.
    metadata = {
        "model_architecture": model.__class__.__name__,
        "training_steps": step,
        "final_loss": current_loss.item() if current_loss is not None else None,
        "hyperparameters": training_hyperparams,
        "self_play_config": self_play_config,
        "game_config": game_config,
        "git_revision": get_git_revision_hash(),
        "save_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # 2. Convert the metadata to a JSON string.
    metadata_json = json.dumps(metadata, indent=4)
    extra_files = {'metadata.json': metadata_json}

    # 3. Save the scripted model to an in-memory buffer
    model.eval()
    scripted_model = torch.jit.script(model)
    buffer = io.BytesIO()
    torch.jit.save(scripted_model, buffer, _extra_files=extra_files)
    # scripted_model.save(buffer, _extra_files=extra_files)
    return buffer.getvalue(), metadata_json

def log_histogram_as_image(writer, tag, data, step):
    """Creates a bar chart from histogram data and logs it as an image."""
    try:
        fig, ax = plt.subplots()
        # Find indices with non-zero counts to make the plot cleaner
        indices = np.where(data > 0)[0]
        counts = data[indices]
        
        if len(indices) > 0:
            ax.bar(indices, counts, tick_label=indices)
            ax.set_xlabel("Inference Batch Size")
            ax.set_ylabel("Frequency (Count)")
            ax.set_title("Inference Batch Size Distribution")
            
            # Convert plot to an image tensor
            # Save the plot to an in-memory buffer and read it back as an image.
            # This is more robust across different matplotlib backends.
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf)
            writer.add_image(tag, np.array(image), step, dataformats='HWC')
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Failed to generate histogram image: {e}")

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
    writer = None  # Define writer in the outer scope
    try:
        # --- Game and Network Configuration ---
        game_config = {
            'board_size': 3,
            'num_actions': 9,
            'input_channels': 3, # X pieces, O pieces, player-to-move
            'num_res_blocks': 1, # A smaller ResNet for a simple game like TTT
            'res_block_channels': 64
        }
        # Group all training hyperparameters into a single dictionary for easy logging.
        training_hyperparams = {
            'learning_rate': 0.0005,
            'batch_size': 64,
            'log_freq_steps': 100,
            'total_training_steps': 2000,
            'model_update_freq_steps': 500,
            'replay_buffer_size': 20000,
            'min_replay_buffer_size': 1000,
            'target_replay_ratio': 4.0,
        }

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
        model = TicTacToeCNN( 
            game_config['input_channels'], 
            game_config['board_size'], 
            game_config['num_actions']).to(device)

        # Replay Buffer
        replay_buffer = ReplayBuffer(
            training_hyperparams['replay_buffer_size'], G_SIZE, P_SIZE, device)

        base_log_dir = 'runs/ttt_alphazero_experiment'
        if not os.path.exists(base_log_dir):
            log_dir = base_log_dir
        else:
            counter = 1
            while os.path.exists(f"{base_log_dir}_{counter}"):
                counter += 1
            log_dir = f"{base_log_dir}_{counter}"
        
        writer = SummaryWriter(log_dir)
        # Log the model graph
        writer.add_graph(model, torch.randn(1, G_SIZE).to(device))

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=training_hyperparams['learning_rate'])

        # Configuration for the self-play run
        self_play_config = { # This is now only for metadata logging
            'threads': 12, # Example: 1.5x a typical 8-core CPU
            'c_base': 19652.0,
            'c_init': 1.25,
            'dirichlet_alpha': 0.3,
            'dirichlet_epsilon': 0.25,
            'simulations': 100,
            'opening_moves': 0
        }

        # --- Initial Model Setup ---
        print("Setting initial model to start C++ self-play workers...")
        
        model_bytes, metadata_json = save_model_with_metadata(
            model,
            step=0,
            current_loss=None,
            game_config=game_config,
            self_play_config=self_play_config,
            training_hyperparams=training_hyperparams
        )
        set_model(alphazero_lib, model_bytes, len(model_bytes), metadata_json.encode('utf-8'))

        loss = None # Initialize loss to a default value
        step = 0

        print( f"Starting training loop for {training_hyperparams['total_training_steps']} steps...")
        while step < training_hyperparams['total_training_steps']:
            # 1. Fetch a small batch of new data to keep the buffer fresh.
            # This call blocks until the C++ workers have produced enough games.
            num_positions_to_fetch = max(1, int(training_hyperparams['batch_size'] / training_hyperparams['target_replay_ratio']))

            fetch_start_time = time.time()
            new_data, queue_size = fetch_selfplay_data_from_cpp(alphazero_lib, num_positions_to_fetch)
            fetch_duration = time.time() - fetch_start_time
            if new_data:
                replay_buffer.add(new_data['game_states'], new_data['policy_targets'], new_data['value_targets'], new_data['player_indices'])
            
            # Check if the buffer is large enough to start training.
            # This elegantly combines the pre-filling and training phases.
            if len(replay_buffer) < training_hyperparams['min_replay_buffer_size']:
                if (step + 1) % 10 == 0: # Log progress occasionally during pre-fill
                    print(f"Pre-filling replay buffer... {len(replay_buffer)}/{training_hyperparams['min_replay_buffer_size']}")
                # Skip the training part of the loop until the buffer is ready.                
                # We still increment step to avoid an infinite loop if something goes wrong.
                step += 1
                continue
            
            # 2. Perform one training step once the buffer is ready.
            model.train()
            start_time = time.time()

            batch_states, batch_target_policies, batch_target_values = replay_buffer.sample(training_hyperparams['batch_size'])

            optimizer.zero_grad()
            pred_values, pred_policy_logits = model(batch_states)
            value_loss_fn = nn.MSELoss()

            loss_policy = -torch.sum(batch_target_policies * F.log_softmax(pred_policy_logits, dim=1), dim=1).mean()
            loss_value = value_loss_fn(pred_values.squeeze(-1), batch_target_values)
            loss = loss_policy + loss_value

            loss.backward()
            optimizer.step()
            duration = time.time() - start_time

            # 3. Log metrics and update the C++ model periodically.
            if (step + 1) % training_hyperparams['log_freq_steps'] == 0:
                writer.add_scalar('Loss/Total', loss.item(), step)
                writer.add_scalar('Loss/Policy', loss_policy.item(), step)
                writer.add_scalar('Loss/Value', loss_value.item(), step)
                writer.add_scalar('Performance/Training_Step_Time_ms', duration * 1000, step)
                writer.add_scalar('Buffer/ReplayBuffer_Size', len(replay_buffer), step)
                writer.add_scalar('Performance/SelfPlay_Fetch_Time_ms', fetch_duration * 1000, step)
                writer.add_scalar('Buffer/SelfPlay_Queue_Size', queue_size, step)
                # Log weights and gradients for each layer
                for name, param in model.named_parameters():
                    writer.add_histogram(f'Gradients/{name}', param.grad, step)
                    writer.add_histogram(f'Weights/{name}', param.data, step)
                print(f"Step {step+1}/{training_hyperparams['total_training_steps']} | Loss: {loss.item():.4f} | Step Time: {duration*1000:.2f}ms | Fetch Time: {fetch_duration*1000:.2f}ms")

            if (step + 1) % training_hyperparams['model_update_freq_steps'] == 0:
                print(f"\nUpdating C++ model at step {step+1}...")
                model_bytes, _ = save_model_with_metadata(
                    model,
                    step=step + 1,
                    current_loss=loss,
                    game_config=game_config,
                    self_play_config=self_play_config,
                    training_hyperparams=training_hyperparams
                )
                set_model(alphazero_lib, model_bytes, len(model_bytes), metadata_json.encode('utf-8'))

            step += 1

        print("\nTraining finished")

        # --- Save the final trained model with embedded metadata ---
        # Use the TensorBoard log directory name to create a unique model filename,
        # linking the model to its training run logs.
        final_model_path = os.path.join("models", f"{os.path.basename(writer.log_dir)}.pt")
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        print(f"\nSaving final trained model to {final_model_path}...")

        model_bytes, metadata_json = save_model_with_metadata(
            model,
            step=step,
            current_loss=loss,
            game_config=game_config,
            self_play_config=self_play_config,
            training_hyperparams=training_hyperparams
        )

        print("Bundling metadata with the model:")
        print(metadata_json)

        # Save the final model bytes to a file.
        with open(final_model_path, "wb") as f:
            f.write(model_bytes)
        print("Model saved successfully.")

        print("\nFetching final inference batch size histogram from C++...")
        try:
            # Define the C-function signature
            c_get_histo = alphazero_lib.get_inference_histogram
            c_get_histo.restype = ctypes.c_int
            c_get_histo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]

            # First, call with no buffer to get the required size.
            required_size = c_get_histo(None, 0)
            
            if required_size > 0:
                # Allocate a buffer of the correct size and get the data.
                histo_data = np.zeros(required_size, dtype=np.uintp)
                histo_ptr = histo_data.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t))
                c_get_histo(histo_ptr, required_size)
                
                # The data is a histogram of counts per batch size.
                # We log this directly to get a distribution plot in TensorBoard.
                final_step = step if 'step' in locals() and step is not None else 0
                log_histogram_as_image(writer, 'Performance/Inference_Batch_Size_Histogram', histo_data, final_step)
                print(f"Logged final inference batch size histogram ({required_size} data points).")
        except Exception as e:
            print(f"Failed to get and log final histogram: {e}")
        
        writer.close()
        print("TensorBoard writer closed.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # This block runs whether the training loop succeeded or failed.
        # --- C++ Resource Cleanup ---
        # This is crucial to prevent crashes on exit by telling the C++ library
        # to shut down its background threads before the interpreter exits.
        print("\nCleaning up C++ resources...")
        cleanup_func = alphazero_lib.cleanup_resources
        cleanup_func.restype = None
        cleanup_func.argtypes = []
        cleanup_func()
        print("C++ cleanup complete.")
