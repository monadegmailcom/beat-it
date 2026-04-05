# RunPod Deployment & Publishing Guide

## GitHub Container Registry (GHCR)

You can absolutely use GitHub Container Registry (GHCR) for free with your `monade@gmail.com` GitHub account!
- **If your repository is Public**: Storage and bandwidth on GHCR is 100% free and unlimited.
- **If your repository is Private**: You get 500MB of storage and 1GB of bandwidth completely free every month.

## Step 1: Authenticate with GitHub

To push images from your Windows WSL compiling machine (or Mac) to GHCR, you need to authenticate Docker with GitHub.

1. Go to your GitHub profile settings -> **Developer settings** -> **Personal access tokens** -> **Tokens (classic)**.
2. Generate a new token with the `write:packages` and `read:packages` scopes.
3. On your compiling machine terminal, log in to Docker using that token:
   ```bash
   echo "YOUR_GITHUB_TOKEN" | docker login ghcr.io -u monadegmailcom --password-stdin
   ```

## Step 2: Build the Image for x86_64

Because RunPod instances run on Intel/AMD GPUs, you must ensure the image is built using the standard x86 architecture. If you are building this on WSL on an Intel Window's machine, it does this by default:
   
```bash
# Build the image
docker build -t ghcr.io/monadegmailcom/beat-it-runpod:latest .
```

*Note: Replace `monadegmailcom` with your actual GitHub handle (e.g., `monade`)*.

## Step 3: Push to GitHub Container Registry

Once the Docker engine is finished compiling the C++ dependencies and downloading PyTorch, upload your image to the web:
```bash
docker push ghcr.io/monadegmailcom/beat-it-runpod:latest
```

## Step 4: Make the Image Public (Optional but Recommended)

By default, the pushed image might be marked as private on your GitHub profile.
1. Go to your GitHub profile -> **Packages**.
2. Click on `beat-it-runpod`.
3. Go to **Package Settings** and change visibility to **Public**. *(This allows RunPod to easily pull the image without requiring complicated secret-key setups in RunPod).*

## Step 5: Create a RunPod Template

To save time and ensure your persistent storage paths are correct, create a reusable template:

1. In the RunPod dashboard, go to **Templates** -> **New Template**.
2. Set the following fields:
   - **Container Image:** `ghcr.io/monadegmailcom/beat-it-runpod:latest`
   - **Container Disk:** `25 GB`
   - **Volume Disk:** `30 GB`
   - **Volume Mount Path:** `/workspace`
   - **Expose HTTP Ports:** `6006, 8080`
   - **Expose TCP Ports:** `22`
3. Add these **Environment Variables**:
   - `BASE_RUNS_DIR` = `/workspace/runs`
   - `BASE_MODELS_DIR` = `/workspace/models`
4. Save the template.

## Step 6: Deploy and Upload Checkpoint

1. Go to **Pods** -> **Deploy** and select a Spot GPU (e.g., RTX 3090).
2. Choose your new custom template from the dropdown.
3. Click **Customize Deployment**, enter `sleep infinity` into the **Start Command** field, and click **Deploy**. *(This keeps the pod awake but idle so you can securely upload files).*
4. Once the pod is running, click **Connect** -> **Web Terminal**.
5. Create the required directories by running:
   ```bash
   mkdir -p /workspace/models /workspace/runs
   ```
6. Use the **Upload File** button in the Web Terminal to upload your `checkpoint.pt` file.
7. Move the uploaded file to the persistent models folder:
   ```bash
   mv checkpoint.pt /workspace/models/checkpoint.pt
   ```

## Step 7: Start Optuna or Training

Now that your checkpoint is safely on the persistent volume, you can switch the pod out of sleep mode.

1. Close the terminal, go to your Pods dashboard, and click **Edit Pod** (the gear/hamburger menu).
2. Clear out the **Start Command** field completely (this forces the pod to use your Dockerfile's native `runpod_entrypoint.sh`).
3. Add a new Environment Variable `RUN_MODE` and set it to `train` or `optuna`.
   *(If using optuna, also add `OPTUNA_MODE` set to `train` or `match`).*
4. Click **Save**.

RunPod will automatically restart your container. It will pick up your checkpoint and start the C++ engine! You can click the **6006** port button for TensorBoard or **8080** for Optuna.

---

## Running Hyperparameter Optimization (Optuna)

Instead of the default `train` mode, the Docker Entrypoint has been structurally rewritten to easily pivot into an optimization mode. You can visualize the search by navigating to `http://localhost:8080` or using the proxy link in the RunPod UI.

### In Docker Locally (Mac/Windows)
To dynamically invoke these from your debugging script, supply the environment configurations directly:

**Optimize Inter-game parallel throughput (Training)**
```bash
RUN_MODE=optuna OPTUNA_MODE=train ./test_runpod_mac.sh
```

**Optimize Intra-game search parallelism (Match Evaluation)**
```bash
RUN_MODE=optuna OPTUNA_MODE=match ./test_runpod_mac.sh
```

### On RunPod Cloud
In the **Edit Pod** "Environment Variables" section:
- Create a Variable named `RUN_MODE` and set it to `optuna`
- Create a Variable named `OPTUNA_MODE` and set it to either `train` or `match`

*RunPod will natively execute the Optuna logic! Since the `optuna.db` file drops directly into the persistent `/workspace/runs` volume alongside the tensorboard logs, you can pick up exactly where you left off organically across different pods!*
