# Daily Training Workflow

This is your linear, zero-fluff playbook for testing locally, building, and deploying to RunPod. It is optimized for SSH and Tmux to keep you in the terminal.

> [!NOTE]
> If you have not yet set up your Github Token or RunPod Template, reference `RUNPOD_DEPLOYMENT.md` first.

---

## 1. Code & Build
### On your Mac
1. Find issues, modify C++ or Python code.
2. Test locally to ensure no syntax errors.

### On your WSL Ubuntu Machine
1. Pull the latest github changes or mount the directory.
2. Build the Intel/AMD compatible x86 image and push to GitHub:
   ```bash
   # Build the image
   docker build -t ghcr.io/monadegmailcom/beat-it-runpod:latest .
   
   # Push it to the registry
   docker push ghcr.io/monadegmailcom/beat-it-runpod:latest
   ```

---

## 2. Allocate the RunPod
Instead of relying on RunPod's UI to stop/start the container, we launch it purely as a persistent sandbox and control the execution from SSH.

1. Go to the RunPod Pods UI.
2. Deploy a new Spot GPU using your `beat-it-runpod` template.
3. Click **Customize Deployment**, enter `sleep infinity` in the Start Command field, and Deploy.
4. *Optional*: If you need to upload a `checkpoint.pt` for the first time on a new volume, use the RunPod Web Terminal to upload it to `/workspace/models/checkpoint.pt`.

---

## 3. Connect via SSH & Tmux
Forward the ports to your local Mac so TensorBoard and Optuna feel native.

1. Get the SSH endpoint from the RunPod UI (e.g., `root@x.x.x.x -p 10000`).
2. Run this on your **Mac Terminal**:
   ```bash
   ssh -L 6006:localhost:6006 -L 8080:localhost:8080 root@x.x.x.x -p 10000 -i ~/.ssh/id_ed25519
   ```
3. Once connected, start a Tmux session to ensure training survives network disconnects:
   ```bash
   tmux new -s training
   ```
   *(To reconnect later if you drop, run `tmux attach -t training`)*

---

## 4. Optimize Hyperparameters (Optuna)
Inside your Tmux session on RunPod:

1. Launch the entrypoint script in `optuna` mode:
   ```bash
   export RUN_MODE=optuna
   export OPTUNA_MODE=train # or 'match'
   /app/runpod_entrypoint.sh
   ```
2. Open your Mac's browser to [http://localhost:8080](http://localhost:8080) to watch the Optuna Dashboard.
3. Once satisfied, stop the run `Ctrl+C`.

---

## 5. Configure & Train
Once you have the optimal numbers from Optuna, apply them and start the main training loop.

1. Still in Tmux, edit your configuration file:
   ```bash
   nano /app/runs/uttt_config.json
   # or vim /app/runs/uttt_config.json
   ```
2. Launch the entrypoint script in `train` mode:
   ```bash
   export RUN_MODE=train
   /app/runpod_entrypoint.sh
   ```
3. Open your Mac's browser to [http://localhost:6006](http://localhost:6006) to watch TensorBoard.
4. Detach from Tmux using `Ctrl+B`, then `D`. Your training is now safely running in the cloud!
