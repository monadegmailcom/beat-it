import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import ctypes
import io
import json
import time
import importlib
from typing import cast
from torch.utils.tensorboard import SummaryWriter
import argparse
from .utils import (
    ReplayBuffer, set_model, fetch_selfplay_data_from_cpp, MetricLogger,
    TrainingHyperparameters, create_inference_model_bundle, save_checkpoint,
    log_histogram_as_image, DataPointers, split_and_add_data,
    train_buffer_metadata_file
)

scheduler_state_file = 'scheduler_state.pt'
optimizer_state_file = 'optimizer_state.pt'
metadata_file = 'metadata.json'
train_buffer_data_file = 'train_buffer_data.npz'
validation_buffer_data_file = 'validation_buffer_data.npz'

# 2. Training Setup
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="AlphaZero Training for Tic-Tac-Toe")
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to a model checkpoint (.pt file) to resume \
                             training from.')
    parser.add_argument('--game', type=str, default='ttt',
                        help='The game to train on (e.g., "ttt", "uttt").')
    args = parser.parse_args()
    session_handle = None

    lib_path = os.path.join('obj', 'libalphazero.so')
    alphazero_lib = ctypes.CDLL(lib_path)
    print(f"Successfully loaded shared library from: {lib_path}")
    writer = None  # Define writer in the outer scope
    try:
        # --- Initial Setup ---
        start_step = 0
        log_dir = None
        architecture_changed = False
        extra_files_to_load = {}

        # --- Dynamically Load Game Configuration ---
        print(f"Loading configuration for game: {args.game}")
        try:
            game_module = importlib.import_module(
                f".{args.game}", package=__package__)
        except ImportError:
            print(f"Error: Could not find configuration module for game"
                  f" '{args.game}'.")
            exit(1)

        # Group all training hyperparameters into a single dictionary for easy
        # logging.
        training_hyperparams: TrainingHyperparameters = {
            'learning_rate': 0.0005,
            'weight_decay': 1e-4,  # L2 regularization strength
            'batch_size': 64,
            'log_freq_steps': 100,
            'total_training_steps': 200000,
            'model_update_freq_steps': 500,
            'checkpoint_freq_steps': 1000,
            'replay_buffer_size': 200000,
            'min_replay_buffer_size': 1000,
            'target_replay_ratio': 4.0,
            'validation_split_percentage': 0.05,  # 5% of data for validation
            'validation_freq_steps': 500,  # Run validation every 500 steps
            'lr_schedule_milestones': [100000, 150000],  # Steps to decay LR
            'lr_schedule_gamma': 0.1,  # LR decay factor
        }

        # Configuration for the self-play run
        self_play_config = {  # This is now only for metadata logging
            # Oversubscribe threads to hide I/O latency
            'threads': 3,
            'selfplay_threads': 20,
            'min_batch_size': 1,
            'c_base': 19652.0,
            'c_init': 1.25,
            'dirichlet_alpha': 0.3,
            'dirichlet_epsilon': 0.25,
            'simulations': 400,
            'opening_moves': 5
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
        model = game_module.model.to(device)

        # --- Game and Network Configuration ---
        game_config = game_module.game_config
        G_SIZE = game_config['input_channels'] * game_config['board_size'] \
            * game_config['board_size']
        P_SIZE = game_config['num_actions']
        basename = game_module.basename

        # Replay Buffer
        train_buffer_size = int(
            training_hyperparams['replay_buffer_size'] *
            (1 - training_hyperparams['validation_split_percentage']))
        validation_buffer_size = \
            training_hyperparams['replay_buffer_size'] - train_buffer_size

        replay_buffer = ReplayBuffer(
            train_buffer_size, G_SIZE, P_SIZE, device)
        validation_buffer = ReplayBuffer(
            validation_buffer_size, G_SIZE, P_SIZE, device)

        # --- Resume from Checkpoint Logic ---
        if args.resume_from:
            print(f"Attempting to resume training from: {args.resume_from}")
            if os.path.exists(args.resume_from):
                # Load the scripted model and extract extra files in one go.
                extra_files_to_load = {
                    optimizer_state_file: b'',
                    metadata_file: b'',
                    scheduler_state_file: b'',
                    train_buffer_data_file: b'',
                    train_buffer_metadata_file: b'',
                    validation_buffer_data_file: b'',
                    'validation_buffer_metadata.json': b''
                }
                old_model_scripted = torch.jit.load(
                    args.resume_from, map_location=device,
                    _extra_files=extra_files_to_load)

                # Load replay buffers
                if extra_files_to_load[train_buffer_data_file]:
                    print("Found training replay buffer in checkpoint.")
                    replay_buffer.load_from_bytes(
                        extra_files_to_load[train_buffer_data_file],
                        extra_files_to_load[train_buffer_metadata_file]
                    )
                else:
                    print("No training replay buffer found in checkpoint. "
                          "Starting with an empty one.")

                if extra_files_to_load[validation_buffer_data_file]:
                    print("Found validation replay buffer in checkpoint.")
                    validation_buffer.load_from_bytes(
                        extra_files_to_load[validation_buffer_data_file],
                        extra_files_to_load['validation_buffer_metadata.json']
                    )
                else:
                    print("No validation replay buffer found in checkpoint. "
                          "Starting with an empty one.")

                # Load metadata and configs
                metadata = json.loads(extra_files_to_load['metadata.json'])
                old_game_config = metadata['game_config']
                new_game_config = game_module.game_config

                # Create a new model instance with the current configuration.
                model = game_module.model.to(device)
                old_state_dict = old_model_scripted.state_dict()
                new_state_dict = model.state_dict()

                # Check for architecture changes by comparing state dict keys
                # and shapes. This is more robust than just checking config
                # values.
                architecture_changed = False
                if set(old_state_dict.keys()) != set(new_state_dict.keys()):
                    architecture_changed = True
                    print("Detected architecture change: layer names differ.")
                else:
                    for key in old_state_dict:
                        if old_state_dict[key].shape \
                             != new_state_dict[key].shape:
                            architecture_changed = True
                            print(f"Detected shape mismatch for layer "
                                  f"'{key}': "
                                  f"old={old_state_dict[key].shape}, "
                                  f"new={new_state_dict[key].shape}")
                            break

                # Smartly transfer weights
                if architecture_changed:
                    print(
                        "Model architecture changed. "
                        "Attempting to transfer compatible weights.")
                    for name, param in old_state_dict.items():
                        if name in new_state_dict and \
                           new_state_dict[name].shape == param.shape:
                            new_state_dict[name].copy_(param)
                    model.load_state_dict(new_state_dict)
                    print("Transferred compatible weights to the new model.")
                else:
                    print("Checkpoint architecture matches. "
                          "Loading weights directly.")
                    model.load_state_dict(old_state_dict)

                # Load configurations from checkpoint, then overwrite with
                # current script's settings
                loaded_hyperparams = metadata.get('hyperparameters', {})
                loaded_self_play_config = metadata.get('self_play_config', {})

                loaded_hyperparams.update(training_hyperparams)
                loaded_self_play_config.update(self_play_config)

                training_hyperparams = cast(
                    TrainingHyperparameters, loaded_hyperparams)
                self_play_config = loaded_self_play_config

                print("Loaded and merged configurations. Script settings "
                      "take precedence.")

                # Load metadata for step count
                start_step = metadata.get('training_steps', 0)

                print(f"Resuming from step {start_step}. "
                      "Optimizer state loaded.")

                run_name = os.path.basename(os.path.dirname(args.resume_from))
                log_dir = os.path.join("runs", run_name)
            else:
                print(f"Warning: Checkpoint file not found at "
                      f"{args.resume_from}. Starting a new run.")

        # Optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_hyperparams['learning_rate'],
            weight_decay=training_hyperparams['weight_decay'])

        # Scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=training_hyperparams['lr_schedule_milestones'],
            gamma=training_hyperparams['lr_schedule_gamma']
        )

        # If resuming, load the optimizer and scheduler state now that they are
        # created
        if args.resume_from and os.path.exists(args.resume_from):
            # Conditionally load optimizer state
            if architecture_changed:
                print("Warning: Model architecture changed. "
                      "Optimizer state will not be loaded.")
            else:
                optimizer_state_buffer = io.BytesIO(
                    extra_files_to_load['optimizer_state.pt'])
                optimizer.load_state_dict(
                    torch.load(optimizer_state_buffer))
                # Also load scheduler state if it exists
                if extra_files_to_load[scheduler_state_file]:
                    scheduler_state_buffer = io.BytesIO(
                        extra_files_to_load[scheduler_state_file])
                    scheduler.load_state_dict(
                        torch.load(scheduler_state_buffer))

        # --- TensorBoard Setup ---
        if log_dir is None:
            base_log_dir = 'runs/' + basename
            if not os.path.exists(base_log_dir):
                log_dir = base_log_dir
            else:
                counter = 1
                while os.path.exists(f"{base_log_dir}_{counter}"):
                    counter += 1
                log_dir = f"{base_log_dir}_{counter}"
        else:
            print(f"Resuming TensorBoard logs in: {log_dir}")

        checkpoint_dir = os.path.join("models", os.path.basename(log_dir))
        os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")

        writer = SummaryWriter(log_dir)
        # Log the model graph
        writer.add_graph(model, torch.randn(1, G_SIZE).to(device))

        # --- C++ Session and Initial Model Setup ---
        print("Creating C++ session and setting initial model...")

        # Prepare C++ function handles
        alphazero_lib.create_session.restype = ctypes.c_void_p
        alphazero_lib.destroy_session.argtypes = [ctypes.c_void_p]
        session_handle = alphazero_lib.create_session()
        if not session_handle:
            raise RuntimeError("Failed to create C++ session.")

        # --- Initial Model Setup ---
        c_set_model_func = getattr(
            alphazero_lib, game_module.set_model_func_name)
        c_set_model_func.restype = ctypes.c_int
        c_set_model_func.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
            ctypes.c_char_p, ctypes.c_int32
        ]

        c_fetch_data_func = getattr(
            alphazero_lib, game_module.fetch_data_func_name)
        c_fetch_data_func.restype = ctypes.c_int
        c_fetch_data_func.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(DataPointers), ctypes.c_uint32
        ]

        model_bytes, metadata_json = create_inference_model_bundle(
            model,
            step=0,
            current_loss=None,
            game_config=game_config,
            self_play_config=self_play_config,
            training_hyperparams=training_hyperparams
        )
        set_model(
            session_handle, c_set_model_func, model_bytes, len(model_bytes),
            metadata_json.encode('utf-8'))

        loss = None  # Initialize loss to a default value
        step = start_step

        print(f"Starting training loop from step {start_step} up to "
              f"{training_hyperparams['total_training_steps']} steps...")
        # 1. Fetch a small batch of new data to keep the buffer fresh.
        num_positions_to_fetch = max(
            1,

            int(training_hyperparams['batch_size']
                / training_hyperparams['target_replay_ratio']))

        # Initialize accumulators for averaging metrics over a logging window
        total_duration_in_window = 0.0
        total_selfplay_duration_in_window = 0.0
        total_loss_in_window = 0.0
        loss_policy_in_window = 0.0
        loss_value_in_window = 0.0
        steps_in_window = 0
        logger = MetricLogger(writer)

        while step < training_hyperparams['total_training_steps']:
            fetch_start_time = time.time()
            # This call blocks until the C++ workers have produced enough
            #  games.
            new_data, queue_size = fetch_selfplay_data_from_cpp(
                session_handle, c_fetch_data_func, num_positions_to_fetch,
                G_SIZE, P_SIZE)
            fetch_duration = time.time() - fetch_start_time

            if new_data:
                split_and_add_data(
                    new_data, replay_buffer, validation_buffer,
                    training_hyperparams['validation_split_percentage'])

            # Check if the buffer is large enough to start training.
            # This elegantly combines the pre-filling and training phases.
            if len(replay_buffer) < training_hyperparams[
                    'min_replay_buffer_size']:
                # Log progress occasionally during pre-fill
                if (step + 1) % 10 == 0:
                    print(f"Pre-filling replay buffer... {len(replay_buffer)}"
                          f"/{training_hyperparams['min_replay_buffer_size']}")
                # Skip the training part of the loop until the buffer is ready.
                # We still increment step to avoid an infinite loop if
                # something goes wrong.
                step += 1
                continue

            # 2. Perform one training step once the buffer is ready.
            model.train()
            start_time = time.time()

            batch_states, batch_target_policies, batch_target_values = \
                replay_buffer.sample(training_hyperparams['batch_size'])

            optimizer.zero_grad()
            pred_values, pred_policy_logits = model(batch_states)
            value_loss_fn = nn.MSELoss()

            loss_policy = -torch.sum(batch_target_policies * F.log_softmax(
                pred_policy_logits, dim=1), dim=1).mean()
            loss_value = value_loss_fn(
                pred_values.squeeze(-1), batch_target_values)
            loss = loss_policy + loss_value

            loss.backward()
            optimizer.step()
            scheduler.step()  # Update the learning rate
            duration = time.time() - start_time

            logger.update(
                loss_total=loss.item(),
                loss_policy=loss_policy.item(),
                loss_value=loss_value.item(),
                step_time_ms=duration * 1000,
                selfplay_time_ms=fetch_duration * 1000
            )

            if (step + 1) % training_hyperparams['log_freq_steps'] == 0:
                # Log weights and gradients for each layer
                for name, param in model.named_parameters():
                    writer.add_histogram(f'Gradients/{name}', param.grad, step)
                    writer.add_histogram(f'Weights/{name}', param.data, step)
                logger.log_and_reset(
                    step, training_hyperparams['total_training_steps'],
                    len(replay_buffer), queue_size,
                    optimizer.param_groups[0]['lr'])

            # --- Periodic Validation Step ---
            if (step + 1) % training_hyperparams['validation_freq_steps'] == 0:
                if len(validation_buffer) > training_hyperparams['batch_size']:
                    print("\nRunning validation...")
                    model.eval()  # Set model to evaluation mode
                    with torch.no_grad():  # Disable gradient calculations
                        val_batch_states, val_batch_policies, val_batch_values\
                         = validation_buffer.sample(
                            training_hyperparams['batch_size'])

                        pred_values, pred_policy_logits = model(
                            val_batch_states)

                        val_loss_policy = -torch.sum(
                            val_batch_policies * F.log_softmax(
                                pred_policy_logits, dim=1),
                            dim=1).mean()
                        val_loss_value = value_loss_fn(
                            pred_values.squeeze(-1), val_batch_values)
                        val_loss_total = val_loss_policy + val_loss_value

                        writer.add_scalar(
                            'Loss/Validation_Total',
                            val_loss_total.item(), step)
                        writer.add_scalar(
                            'Loss/Validation_Policy',
                            val_loss_policy.item(), step)
                        writer.add_scalar(
                            'Loss/Validation_Value',
                            val_loss_value.item(), step)
                        print(
                            f"Validation Loss: {val_loss_total.item():.4f}\n")
                    model.train()  # Set model back to training mode

            if (step + 1) % training_hyperparams['model_update_freq_steps']\
                    == 0:
                print(f"\nUpdating C++ model at step {step+1}...")
                model_bytes, metadata_json = create_inference_model_bundle(
                    model,
                    step=step + 1,
                    current_loss=loss,
                    game_config=game_config,
                    self_play_config=self_play_config,
                    training_hyperparams=training_hyperparams
                )
                set_model(
                    session_handle, c_set_model_func, model_bytes, len(model_bytes),
                    metadata_json.encode('utf-8'))

            # Periodically save a checkpoint
            if (step + 1) % training_hyperparams['checkpoint_freq_steps'] == 0:
                print(f"\nSaving checkpoint at step {step+1} to "
                      f"{checkpoint_path}...")
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    step=step + 1,
                    current_loss=loss,
                    game_config=game_config,
                    self_play_config=self_play_config,
                    training_hyperparams=training_hyperparams,
                    path=checkpoint_path,
                    train_buffer=replay_buffer,
                    validation_buffer=validation_buffer
                )

            step += 1

        print("\nTraining finished")

        # --- Save the final trained model with embedded metadata ---
        if writer:
            # The log_dir is like 'runs/ttt_alphazero_experiment_6'
            final_model_path = os.path.join(
                "models", os.path.basename(log_dir), "final_model.pt")
            print(f"\nSaving final trained model to {final_model_path}...")
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                step=step,
                current_loss=loss,
                game_config=game_config,
                self_play_config=self_play_config,
                training_hyperparams=training_hyperparams,
                path=final_model_path,
                train_buffer=replay_buffer,
                validation_buffer=validation_buffer
            )

        print("\nFetching final inference batch size histogram from C++...")
        try:
            # Define the C-function signature
            c_get_histo = alphazero_lib.get_inference_histogram
            c_get_histo.restype = ctypes.c_int
            c_get_histo.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_int
            ]

            # First, call with no buffer to get the required size.
            required_size = c_get_histo(None, 0)

            if required_size > 0:
                # Allocate a buffer of the correct size and get the data.
                histo_data = np.zeros(required_size, dtype=np.uintp)
                histo_ptr = histo_data.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_size_t))
                c_get_histo(session_handle, histo_ptr, required_size)

                # The data is a histogram of counts per batch size.
                # We log this directly to get a distribution plot in
                # TensorBoard.
                if writer:
                    final_step = step if 'step' in locals() and step is \
                        not None else 0
                    log_histogram_as_image(
                        writer, 'Performance/Inference_Batch_Size_Histogram',
                        histo_data, final_step)
                print(f"Logged final inference batch size histogram "
                      f"({required_size} data points).")
        except Exception as e:
            print(f"Failed to get and log final histogram: {e}")

        if writer:
            writer.close()
            print("TensorBoard writer closed.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # This block runs whether the training loop succeeded or failed.
        # --- C++ Resource Cleanup ---
        if session_handle:
            print("\nCleaning up C++ session...")
            alphazero_lib.destroy_session(session_handle)
            print("C++ session cleanup complete.")
