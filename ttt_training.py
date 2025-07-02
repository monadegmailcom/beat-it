import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import ctypes
import io
from torch.utils.tensorboard import SummaryWriter

G_SIZE = 18  # 2 planes * 9 cells
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
        ctypes.c_int8, # current_player
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
        config['current_player'],
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

def print_board(state_vector, player_index):
    """
    Prints a 3x3 Tic-Tac-Toe board from a serialized state vector.
    The state is always from the perspective of the current player.
    """
    board = [' '] * 9
    current_player_symbol = 'X' if player_index == 0 else 'O'
    opponent_symbol = 'O' if player_index == 0 else 'X'

    # First 9 elements are the current player's pieces
    for i in range(9):
        if state_vector[i] == 1.0:
            board[i] = current_player_symbol
    
    # Next 9 elements are the opponent's pieces
    for i in range(9):
        if state_vector[i + 9] == 1.0:
            board[i] = opponent_symbol

    print(f"Turn to move: {current_player_symbol}")
    print("-------------")
    print(f"| {board[0]} | {board[1]} | {board[2]} |")
    print("-------------")
    print(f"| {board[3]} | {board[4]} | {board[5]} |")
    print("-------------")
    print(f"| {board[6]} | {board[7]} | {board[8]} |")
    print("-------------")

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
    try:
        lib_path = os.path.join('obj', 'libalphazero.so')
        alphazero_lib = ctypes.CDLL(lib_path)
        print(f"Successfully loaded shared library from: {lib_path}")

        # --- Game and Network Configuration ---
        game_config = {
            'board_size': 3,
            'num_actions': 9,
            'input_channels': 2, # Player pieces, Opponent pieces
            'num_res_blocks': 1, # A smaller ResNet for a simple game like TTT
            'res_block_channels': 64
        }
        learning_rate = 0.0005
        # Training loop hyperparameters
        training_iterations = 100
        games_per_iteration = 100 # Generate more data before training
        epochs_per_iteration = 5 # Train for fewer epochs on the larger dataset
        batch_size = 64

        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Model
        model = AlphaZeroCNN(**game_config).to(device)

        # --- TensorBoard Setup ---
        # This will create a 'runs' directory to store the logs
        writer = SummaryWriter('runs/ttt_alphazero_experiment_3')
        # Log the model graph
        writer.add_graph(model, torch.randn(1, G_SIZE).to(device))

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Configuration for the self-play run
        base_self_play_config = {
            'c_base': 19652.0,
            'c_init': 1.25,
            'dirichlet_alpha': 0.3,
            'dirichlet_epsilon': 0.25,
            'simulations': 100,
            'opening_moves': 1
        }

        for iteration in range(training_iterations):
            print(f"\n===== Training Iteration {iteration + 1}/{training_iterations} =====")
            
            # Update the model for the next self-play phase
            model.eval()
            scripted_model = torch.jit.script(model)
            buffer = io.BytesIO()
            torch.jit.save(scripted_model, buffer)
            model_bytes = buffer.getvalue()
            set_model(alphazero_lib, model_bytes, len(model_bytes))

            # Self-Play Phase: Generate data from multiple games
            print("--- Generating self-play data ---")
            # Use lists to accumulate data from each game. This is memory efficient.
            iteration_states, iteration_policies, iteration_values, iteration_indices = [], [], [], []

            for i in range(games_per_iteration):
                # Alternate the starting player to get more diverse game data
                self_play_config = base_self_play_config.copy()
                self_play_config['current_player'] = i % 2 

                new_data = get_selfplay_data_from_cpp(alphazero_lib, self_play_config)

                if new_data:
                    # Copy the data from the C++ view into a new Python-managed array
                    # and append it to the list. This is the essential, safe copy point.
                    iteration_states.append(new_data['game_states'].copy())
                    iteration_policies.append(new_data['policy_targets'].copy())
                    iteration_values.append(new_data['value_targets'].copy())
                    iteration_indices.append(new_data['player_indices'].copy())
  
            if not iteration_states:
                print("No data generated in this iteration. Skipping training.")
                continue

            # After collecting all data, concatenate the lists of arrays into single,
            # large NumPy arrays. This is far more efficient than appending in a loop.
            all_states = np.concatenate(iteration_states)
            all_policies = np.concatenate(iteration_policies)
            all_values = np.concatenate(iteration_values)
            all_player_indices = np.concatenate(iteration_indices)
            print(f"Generated {len(all_states)} total positions from {games_per_iteration} games.")

            # 2. Training Phase
            print("--- Training the model ---")
            states_tensor = torch.from_numpy(all_states).float()
            policy_targets_tensor = torch.from_numpy(all_policies).float()
            value_targets_tensor = torch.from_numpy(all_values).float()

            dataset = torch.utils.data.TensorDataset(states_tensor, policy_targets_tensor, value_targets_tensor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            value_loss_fn = nn.MSELoss()

            for epoch in range(epochs_per_iteration):
                model.train()
                total_epoch_loss, total_epoch_policy_loss, total_epoch_value_loss = 0, 0, 0
                for batch_states, batch_target_policies, batch_target_values in dataloader:
                    batch_states, batch_target_policies, batch_target_values = \
                        batch_states.to(device), batch_target_policies.to(device), batch_target_values.to(device)

                    optimizer.zero_grad()
                    pred_values, pred_policy_logits = model(batch_states)

                    loss_policy = -torch.sum(batch_target_policies * F.log_softmax(pred_policy_logits, dim=1), dim=1).mean()
                    loss_value = value_loss_fn(pred_values.squeeze(-1), batch_target_values)
                    total_loss = loss_policy + loss_value

                    total_loss.backward()
                    optimizer.step()

                    total_epoch_loss += total_loss.item()
                    total_epoch_policy_loss += loss_policy.item()
                    total_epoch_value_loss += loss_value.item()

                avg_epoch_loss = total_epoch_loss / len(dataloader)
                avg_policy_loss = total_epoch_policy_loss / len(dataloader)
                avg_value_loss = total_epoch_value_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{epochs_per_iteration}], Avg Loss: {avg_epoch_loss:.4f}, Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}")

            # Log metrics to TensorBoard at the end of each training iteration
            # We use the iteration number as the global step for the x-axis.
            writer.add_scalar('Loss/Total', avg_epoch_loss, iteration)
            writer.add_scalar('Loss/Policy', avg_policy_loss, iteration)
            writer.add_scalar('Loss/Value', avg_value_loss, iteration)

        print("\nTraining finished.")

        # --- Save the final trained model to a file --- 
        final_model_path = "models/ttt_model_final.pt"
        print(f"\nSaving final trained model to {final_model_path}...")
        # We need to script it before saving, ensuring it's in eval mode
        final_scripted_model = torch.jit.script(model)
        final_scripted_model.save(final_model_path)
        print("Model saved successfully.")

        # Close the TensorBoard writer
        writer.close()

        # --- Compare final model predictions with MCTS targets ---
        print("\n--- Comparing final model predictions with MCTS targets ---")
        model.eval() # Set model to evaluation mode
        with torch.no_grad():
            # Let's compare the first few positions to see how well the model learned
            num_comparisons = min(len(states_tensor), 10) # Compare up to 10 positions
            for i in range(num_comparisons):
                state_tensor_for_model = states_tensor[i].unsqueeze(0).to(device)
                state_vector_for_print = states_tensor[i].numpy()
                player_index = all_player_indices[i]
                target_policy = policy_targets_tensor[i].numpy()
                target_value = value_targets_tensor[i].item()

                # Get model prediction
                pred_value, pred_policy_logits = model(state_tensor_for_model)
                pred_policy_probs = F.softmax(pred_policy_logits, dim=1).squeeze(0).cpu().numpy()
                pred_value = pred_value.item()

                print(f"\n========== Position {i+1} ==========")
                print_board(state_vector_for_print, player_index)
                print(f"Value Target: {target_value: .4f}  |  Predicted: {pred_value: .4f}")
                print("---------------------------------")
                print("Move | MCTS Target | NN Predicted ")
                print("---------------------------------")
                for j in range(game_config['num_actions']):
                    print(f" {j}   |   {target_policy[j]:.4f}    |  {pred_policy_probs[j]:.4f}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
