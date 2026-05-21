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
2. Build the Intel/AMD compatible x86 image and push to GitHub (only needed when dependencies or C++ code changes):
   ```bash
   # Build the image
   docker build -t ghcr.io/monadegmailcom/beat-it-runpod:latest .
   
   # Push it to the registry
   docker push ghcr.io/monadegmailcom/beat-it-runpod:latest
   ```

> [!TIP]
> **Avoid Rebuilds:** If you only changed Python scripts or the entrypoint, you don't need to rebuild. Ensure `AUTO_UPDATE=1` is set in your RunPod environment variables; the pod will pull latest code on boot.

---

## 2. Allocate the RunPod
The container is designed to be "Idle-by-Default," meaning it will start, initialize background services (SSH, TensorBoard), and then simply sleep until you are ready.

1. Go to the RunPod Pods UI.
2. Deploy a new Spot GPU using your `beat-it-runpod` template.
3. **No custom commands are required** in the RunPod UI. You can optionally set the environment variable `AUTO_UPDATE=1` if you want it to pull latest code on boot.
4. Deploy and wait for the SSH endpoint to appear.
5. *Optional*: If you need to upload a `checkpoint.pt` for the first time on a new volume, use the RunPod Web Terminal to upload it to `/workspace/models/checkpoint.pt`.

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

1. Configure the environment to persist across restarts by writing to the `.env` file:
   ```bash
   echo "export RUN_MODE=optuna" > /workspace/runs/.env
   echo "export OPTUNA_MODE=train" >> /workspace/runs/.env
   ```
2. Launch the entrypoint script:
   ```bash
   /app/runpod_entrypoint.sh
   ```
3. Open your Mac's browser to [http://localhost:8080](http://localhost:8080) to watch the Optuna Dashboard.
4. Once satisfied, stop the run `Ctrl+C`.

---

## 5. Configure & Train
Once you have the optimal numbers from Optuna, apply them and start the main training loop.

1. Still in Tmux, edit your configuration file:
   ```bash
   nano /workspace/runs/uttt_config.json
   ```
2. Update the persistent environment to training mode:
   ```bash
   echo "export RUN_MODE=train" > /workspace/runs/.env
   ```
3. Launch the entrypoint script:
   ```bash
   /app/runpod_entrypoint.sh
   ```
   *(Since `RUN_MODE=train` is now in `/workspace/runs/.env`, the training will automatically resume if the Pod is preempted and restarts!)*

4. Open your Mac's browser to [http://localhost:6006](http://localhost:6006) to watch TensorBoard.
5. Detach from Tmux using `Ctrl+B`, then `D`. Your training is now safely running and protected against Spot restarts!
