# RunPod Cheat Sheet & Troubleshooting Guide

This guide compiles all the critical, hard-earned knowledge and terminal workarounds we implemented for running your AlphaZero training pipeline on RunPod. Keep this file handy in VS Code whenever you deploy or configure a new Pod.

---

## 🚨 1. Avoid Data Loss: The Network Volume Rule
**Before stopping or terminating a Pod, always verify if you have a persistent volume attached.**
* **The Warning:** If you click "Stop" and see *"You do not have a volume configured. ALL DATA will be lost!"*, your data is ephemeral.
* **The Fix:**
  1. Go to the **Storage** section of your RunPod dashboard.
  2. Create a **Network Volume** (e.g., 50 GB) in the same data center region.
  3. When deploying your GPU, scroll down to **Storage** and **mount your Network Volume** to `/workspace`.
  4. Once booted, your persistent data will live safely in `/workspace/runs` and `/workspace/models`.

---

## ✏️ 2. File Editors (Vim & Nano)
We have **permanently solved this** by baking `vim` and `nano` directly into your project's [Dockerfile](file:///Users/monade/source/beat-it/Dockerfile).
* **Next build:** The next time you build and push your Docker image to GitHub, both editors will be pre-installed out of the box!
* **Manual Install (If ever needed on a raw container):**
  If you ever spin up a raw Ubuntu container and find editors missing, run this to install them:
  ```bash
  apt-get update && apt-get install -y vim nano
  ```

---

## 🌐 3. Forcing IPv4 in Apt-Get (Network Quirk Fix)
If `apt-get update` hangs or throws `Network is unreachable` errors, the container's network is trying to connect to Ubuntu mirrors over **IPv6**, which RunPod does not route. 
* **The Fix:** Force `apt-get` to use **IPv4 only** using the `-o` configuration flag:
  ```bash
  apt-get -o Acquire::ForceIPv4=true update && apt-get -o Acquire::ForceIPv4=true install -y vim nano
  ```

---

## 🔑 4. GitHub Authentication inside the Pod
When you try to pull or update code using `git fetch` inside your SSH session, it will fail with `Permission denied (publickey)` unless authenticated. Use one of these two seamless paths:

### Path A: Switch to HTTPS (Easiest if the repo is Public)
If your repository is public, you don't need any SSH keys! Tell Git to use HTTPS:
```bash
git remote set-url origin https://github.com/monadegmailcom/beat-it.git
```
Now, commands like `git fetch` and your container's `AUTO_UPDATE=1` boot system will work flawlessly out of the box.

### Path B: SSH Agent Forwarding (Best for Private Repos)
Instead of copying your private SSH keys onto the remote cloud container, let your local Mac securely sign the authentication requests.
1. On your **local Mac terminal**, add your GitHub SSH key to your agent:
   ```bash
   ssh-add -K ~/.ssh/id_ed25519
   ```
2. SSH into your RunPod using the **`-A`** flag:
   ```bash
   ssh -A root@x.x.x.x -p [PORT] -i ~/.ssh/id_ed25519
   ```
3. Now, `git fetch` inside the container will work instantly by securely leveraging your Mac's active keys!

---

## 🚀 5. Starting the Optuna Optimization
When running Optuna trials or training manually in your SSH terminal, follow these steps to avoid environment or port conflicts:

### Step A: Kill conflicting processes
If TensorBoard is already holding the active ports in the background, kill them:
```bash
pkill -f tensorboard
```

### Step B: Sync with latest code from Git
If you pushed fresh updates from your Mac:
```bash
cd /app
git fetch --all
git reset --hard origin/main
```

### Step C: Load Environment Variables on Login
Add this to your remote profile so your interactive SSH shells are always perfectly synced with the container paths:
```bash
echo "export BASE_RUNS_DIR=/workspace/runs" >> /root/.bashrc
echo "export BASE_MODELS_DIR=/workspace/models" >> /root/.bashrc
echo "source /workspace/runs/.env 2>/dev/null" >> /root/.bashrc
```

### Step D: Run the script
Launch the entrypoint from any directory:
```bash
/app/runpod_entrypoint.sh
```

---

## 💾 6. Downloading Data for Local Analysis
To download your Optuna database or model checkpoints to your local Mac, run this from your **local Mac terminal**:
```bash
# Download runs (including optuna.db)
scp -P [PORT_NUMBER] -i ~/.ssh/id_ed25519 -r root@204.12.201.59:/workspace/runs ./runs_backup

# Download models
scp -P [PORT_NUMBER] -i ~/.ssh/id_ed25519 -r root@204.12.201.59:/workspace/models ./models_backup
```
