import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import ctypes
import io
import json
import time
import importlib
from typing import cast
from torch.utils.tensorboard import SummaryWriter
import argparse
import shutil
from .utils import (
    ReplayBuffer, set_model, fetch_selfplay_data_from_cpp, MetricLogger,
    TrainingHyperparameters, create_inference_model_bundle, save_checkpoint,
    DataPointers, split_and_add_data, GameType, CppStats,
    train_buffer_metadata_file, Hyperparameters, evaluate_models,
    pause_session, resume_session
)

scheduler_state_file = 'scheduler_state.pt'
optimizer_state_file = 'optimizer_state.pt'
metadata_file = 'metadata.json'
train_buffer_data_file = 'train_buffer_data.npz'
validation_buffer_data_file = 'validation_buffer_data.npz'



# 2. Training Setup



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="AlphaZero Training")
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to a model checkpoint (.pt file) to resume \
                             training from.')
    parser.add_argument('--game', type=str,
                        help='The game to train on (e.g., "ttt", "uttt").')
    args = parser.parse_args()
    
    # Map string game name to GameType enum
    try:
        game_type = GameType[args.game.upper()]
    except KeyError:
        print(f"Error: Invalid game type '{args.game}'. Available types: {[e.name for e in GameType]}")
        exit(1)

    session_handle = None

    possible_paths = [
        os.path.join('build', 'libalphazero.dylib'),
        os.path.join('build', 'libalphazero.so'),
    ]
    lib_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if lib_path is None:
        raise FileNotFoundError(f"Could not find libalphazero shared library. Checked: {possible_paths}")
    alphazero_lib = ctypes.CDLL(lib_path)
    print(f"Successfully loaded shared library from: {lib_path}")
    writer = None  # Define writer in the outer scope
    try:
        # --- Initial Setup ---
        start_step = 0
        log_dir = None
        architecture_changed = False
        extra_files_to_load = {}

        # --- Load Game Configuration from JSON ---
        print(f"Loading configuration for game: {args.game}")
        try:
            game_module = importlib.import_module(
                f".{args.game}", package=__package__)
            config_path = os.path.join(
                os.path.dirname(__file__), f"{args.game}_config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            game_config = config['game_config']
            training_hyperparams = cast(
                TrainingHyperparameters, config['training_hyperparams'])
            self_play_config = config['self_play_config']

        except (ImportError, FileNotFoundError, KeyError) as e:
            print(f"Error: Could not load configuration for game "
                  f"'{args.game}': {e}")
            exit(1)

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
        model = game_module.create_model(game_config).to(device)

        # --- Game and Network Configuration ---
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

                # Load metadata
                metadata = json.loads(extra_files_to_load['metadata.json'])

                # Create a new model instance with the current configuration.
                model = game_module.create_model(game_config).to(device)
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

                # The new configs from the JSON file take precedence
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
                    torch.load(optimizer_state_buffer, map_location=device))
                # Also load scheduler state if it exists
                if extra_files_to_load[scheduler_state_file]:
                    scheduler_state_buffer = io.BytesIO(
                        extra_files_to_load[scheduler_state_file])
                    scheduler.load_state_dict(
                        torch.load(scheduler_state_buffer, map_location=device))

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


        c_fetch_data_func = alphazero_lib.fetch_selfplay_data
        c_fetch_data_func.restype = None
        c_fetch_data_func.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(DataPointers),
            ctypes.c_uint32,
            ctypes.POINTER(CppStats), ctypes.POINTER(CppStats),
            ctypes.POINTER(CppStats)
        ]

        c_set_model_func = alphazero_lib.set_model
        c_set_model_func.restype = None
        c_set_model_func.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_uint32,
            ctypes.POINTER(CppStats), ctypes.POINTER(CppStats),
            ctypes.POINTER(CppStats)
        ]

        model_bytes, metadata_json = create_inference_model_bundle(
            model,
            step=0,
            current_loss=None,
            game_config=game_config,
            self_play_config=self_play_config,
            training_hyperparams=training_hyperparams
        )

        c_evaluate_func = alphazero_lib.evaluate_models
        # restype/argtypes are set inside evaluate_models wrapper

        # Populate Hyperparameters struct
        hp = Hyperparameters(self_play_config)

        print("Creating C++ session...")
        # Prepare C++ function handles
        alphazero_lib.create_session.restype = ctypes.c_void_p
        # GameType (int), model_data (char*), model_len (uint32), hp (POINTER)
        alphazero_lib.create_session.argtypes = [
            ctypes.c_int32, ctypes.c_char_p, ctypes.c_uint32, ctypes.POINTER(Hyperparameters)
        ]
        alphazero_lib.destroy_session.argtypes = [ctypes.c_int32, ctypes.c_void_p]

        # Call create_session
        session_handle = alphazero_lib.create_session(
            ctypes.c_int32(game_type.value),
            model_bytes,
            len(model_bytes),
            ctypes.byref(hp)
        )
        if not session_handle:
            raise RuntimeError("Failed to create C++ session.")

        # Destroy session requires/takes game_type in the updated API?
        # Checking lib_interface.cpp: 
        # void destroy_session( GameType game_type, void* session)
        # Yes.

        # Note: We don't need to call set_model immediately anymore because create_session does it.
        # But we DO need to keep c_set_model_func for later updates.

        # --- Print Final Configuration ---
        # Combine all configs into one dictionary for printing to ensure all
        # settings (from file and checkpoint) are clear.
        final_config = {
            "game_config": game_config,
            "training_hyperparams": training_hyperparams,
            "self_play_config": self_play_config
        }
        print("\n" + "="*80)
        print("Starting training with the following merged configuration:")
        print(json.dumps(final_config, indent=4))
        print("="*80 + "\n")

        loss = None  # Initialize loss to a default value
        step = start_step

        previous_checkpoint_bytes = None
        # If we just loaded a model, strictly speaking it's the "current" one.
        # But for the first checkpoint we create, we can compare against this loaded one if we had its bytes.
        # However, we don't easily have its bytes unless we re-serialized it. 
        # For simplicity, we start 'previous' as None, so the first match happens at the SECOND checkpoint.
        # OR: we can generate bytes for the initial model now.
        previous_checkpoint_path = None
        if args.resume_from and os.path.exists(args.resume_from):
             previous_checkpoint_path = args.resume_from
        else:
             # Save initial random model as 'checkpoint_prev.pt' so the first evaluation has a baseline
             initial_prev_path = os.path.join(checkpoint_dir, "checkpoint_prev.pt")
             print(f"Saving initial random model to {initial_prev_path} for first evaluation...")
             save_checkpoint(
                model,
                optimizer,
                scheduler,
                step=step,
                current_loss=torch.tensor(0.0), # Dummy loss
                game_config=game_config,
                self_play_config=self_play_config,
                training_hyperparams=training_hyperparams,
                path=initial_prev_path,
                train_buffer=None,
                validation_buffer=None
             )
             previous_checkpoint_path = initial_prev_path


        # 1. Fetch a small batch of new data to keep the buffer fresh.
        num_positions_to_fetch = max(
            1,
            int(training_hyperparams['batch_size']
                / training_hyperparams['target_replay_ratio']))

        print(f"Starting training loop from step {start_step} up to "
              f"{training_hyperparams['total_training_steps']} steps."
              f" Fetch {num_positions_to_fetch} positions at a time.")

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
            new_data, stats = fetch_selfplay_data_from_cpp(
                session_handle, c_fetch_data_func, game_type,
                num_positions_to_fetch, G_SIZE, P_SIZE)
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
                selfplay_time_ms=fetch_duration * 1000,
                inference_batch_size=stats['inference_batch_size'],
                inference_time=stats['inference_time'],
                allocator_size=stats['allocator_size']
            )

            if (step + 1) % training_hyperparams['log_freq_steps'] == 0:
                # Log weights and gradients for each layer
                for name, param in model.named_parameters():
                    writer.add_histogram(f'Gradients/{name}', param.grad, step)
                    writer.add_histogram(f'Weights/{name}', param.data, step)
                logger.log_and_reset(
                    step, training_hyperparams['total_training_steps'],
                    len(replay_buffer),
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
                    session_handle, c_set_model_func, game_type, model_bytes)

            # Periodically save a checkpoint
            if (step + 1) % training_hyperparams['checkpoint_freq_steps'] == 0:
                # Rotate checkpoints to allow evaluation against previous version
                prev_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_prev.pt")
                if os.path.exists(checkpoint_path):
                    shutil.copy2(checkpoint_path, prev_checkpoint_path)
                    previous_checkpoint_path = prev_checkpoint_path
                    print(f"Backed up previous checkpoint to {prev_checkpoint_path}")
                
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

                # --- Evaluation Match ---
                print(f"Running evaluation match at step {step+1}...")
                
                # Use paths for evaluation
                current_model_path = checkpoint_path
                
                if previous_checkpoint_path and os.path.exists(previous_checkpoint_path):
                    # Pause self-play to free up GPU resources
                    print("Pausing self-play for evaluation...")
                    pause_session(session_handle, alphazero_lib, game_type)
                    
                    # Release training resources from GPU
                    print("Releasing model/optimizer from GPU...")
                    model.cpu()
                    # We can't easily move optimizer state to CPU and back without full reload or manual iteration.
                    # Since we just saved a checkpoint, it's safest and easiest to just reload everything after.
                    del optimizer
                    del scheduler
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    eval_games_dir = os.path.join(log_dir, "games")
                    os.makedirs(eval_games_dir, exist_ok=True)
                    eval_log_path = os.path.join(eval_games_dir, f"eval_step_{step+1}.jsonl")
                    num_eval_games = training_hyperparams.get('evaluation_games', 100)

                    # Pass paths to the models
                    current_model_path = checkpoint_path
                    
                    print(f"Starting evaluation (in-process) vs {os.path.basename(previous_checkpoint_path)}...")
                    
                    # Read models as bytes for the in-process call
                    with open(current_model_path, 'rb') as f:
                        current_model_bytes = f.read()
                    with open(previous_checkpoint_path, 'rb') as f:
                        prev_model_bytes = f.read()
                        
                    # Reconstruct Hyperparameters struct
                    hp_struct = Hyperparameters(self_play_config)
                    
                    try:
                        eval_start = time.time()
                        eval_result = evaluate_models(
                            None, # No session handle needed for eval
                            c_evaluate_func,
                            game_type,
                            current_model_bytes,
                            prev_model_bytes,
                            hp_struct,
                            num_eval_games,
                            eval_log_path,
                            os.path.basename(log_dir),
                            step+1
                        )
                        eval_duration = time.time() - eval_start
                        print(f"Evaluation took {eval_duration:.2f}s")
                        
                        total_games = eval_result.wins_p1 + eval_result.wins_p2 + eval_result.draws
                        if total_games > 0:
                            win_rate_p1 = eval_result.wins_p1 / total_games
                            win_rate_p2 = eval_result.wins_p2 / total_games
                            draw_rate = eval_result.draws / total_games
                            print(f"Evaluation Result (Current vs Previous):")
                            print(f"  Current Wins: {eval_result.wins_p1} ({win_rate_p1:.1%})")
                            print(f"  Previous Wins: {eval_result.wins_p2} ({win_rate_p2:.1%})")
                            print(f"  Draws:        {eval_result.draws} ({draw_rate:.1%})")
                            
                            if writer:
                                writer.add_scalar('Evaluation/WinRates/Current', win_rate_p1, step+1)
                                writer.add_scalar('Evaluation/WinRates/Previous', win_rate_p2, step+1)
                                writer.add_scalar('Evaluation/WinRates/Draw', draw_rate, step+1)
                        else:
                            print("Evaluation finished with 0 games.")
                            
                    except Exception as e:
                        print(f"Evaluation failed: {e}")

                    # Restore Training State from Checkpoint
                    print("Restoring training state from checkpoint...")
                    checkpoint = torch.jit.load(checkpoint_path) 
                    # Note: We need the Python model, not the JIT model.
                    # We saved the Python model state in the 'optimizer_state.pt' etc logic? 
                    # Wait, save_checkpoint saves JIT model with extra files.
                    # It DOES NOT save the python source code/architecture pickling in a way we can just 'load'.
                    # We have the 'model' object instance still (just CPU moved).
                    # We should just load state dict if possible?
                    # Actually, we moved 'model' to CPU. It's still valid. We just need to move it back.
                    # But optimizer was deleted. We need to reload optimizer state.
                    
                    # Move model back
                    model.to(device)
                    model.train()
                    
                    # Re-instantiate optimizer/scheduler
                    optimizer = optim.Adam(
                        model.parameters(),
                        lr=training_hyperparams['learning_rate'],
                        weight_decay=training_hyperparams['weight_decay'])
                    
                    scheduler = optim.lr_scheduler.MultiStepLR(
                        optimizer,
                        milestones=training_hyperparams['lr_schedule_milestones'],
                        gamma=training_hyperparams['lr_schedule_gamma'])

                    # Load states from the saved checkpoint extra files
                    # We saved optimizer_state.pt in extra_files.
                    # To load it, we load the JIT model and access extra files.
                    # We can reused the 'checkpoint' loaded above? No, torch.jit.load loads ScriptModule.
                    # extra_files map needs to be passed to load.
                    
                    extra_files = {
                        'optimizer_state.pt': '', 
                        'scheduler_state.pt': ''
                    }
                    torch.jit.load(checkpoint_path, _extra_files=extra_files)
                    
                    # extra_files values are already bytes, so directly wrap in BytesIO
                    optimizer.load_state_dict(torch.load(io.BytesIO(extra_files['optimizer_state.pt']), map_location=device)) 
                    
                    scheduler.load_state_dict(torch.load(io.BytesIO(extra_files['scheduler_state.pt']), map_location=device))
                    
                    print("Training state restored.")
                    
                    print("Resuming self-play...")
                    resume_session(session_handle, alphazero_lib, game_type)
                else:
                    if previous_checkpoint_path:
                         print(f"Previous checkpoint path {previous_checkpoint_path} not found. Skipping eval.")
                    else:
                         print("No previous checkpoint to evaluate against yet.")

                # Update previous path for next time
                previous_checkpoint_path = checkpoint_path


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
            alphazero_lib.destroy_session(ctypes.c_int32(game_type.value), session_handle)
            print("C++ session cleanup complete.")
