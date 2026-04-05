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

## Step 5: Deploy on RunPod

1. Go to your RunPod dashboard and rent a GPU Pod.
2. The UI will default to a "Runpod Pytorch" template. Click **Customize Deployment**.
3. Under **Container Image**, replace the default text with your image URL:
   `ghcr.io/monadegmailcom/beat-it-runpod:latest`
4. Under **Volume Mounts**, ensure your persistent pod volume is mapped to:
   - `/app/models`
   - `/app/runs`
5. Under **Expose HTTP Ports**, add `6006, 8080` so you can securely click the web link directly in RunPod to view your Tensorboard and Optuna data traces live!

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
In the exact same "Environment Variables" section of the Pod Configuration wizard where you setup the container:
- Create a Variable named `RUN_MODE` and set it to `optuna`
- Create a Variable named `OPTUNA_MODE` and set it to either `train` or `match`

*RunPod will natively execute the Optuna logic! Since the `optuna.db` file drops directly into the persistent `/app/runs` volume alongside the tensorboard logs, you can pick up exactly where you left off organically across different pods!*
