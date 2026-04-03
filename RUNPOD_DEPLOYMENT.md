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
   echo "YOUR_GITHUB_TOKEN" | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
   ```

## Step 2: Build the Image for x86_64

Because RunPod instances run on Intel/AMD GPUs, you must ensure the image is built using the standard x86 architecture. If you are building this on WSL on an Intel Window's machine, it does this by default:
   
```bash
# Build the image and strictly target x86_64
docker build --platform linux/amd64 -t ghcr.io/YOUR_GITHUB_USERNAME/beat-it-runpod:latest .
```

*Note: Replace `YOUR_GITHUB_USERNAME` with your actual GitHub handle (e.g., `monade`)*.

## Step 3: Push to GitHub Container Registry

Once the Docker engine is finished compiling the C++ dependencies and downloading PyTorch, upload your image to the web:
```bash
docker push ghcr.io/YOUR_GITHUB_USERNAME/beat-it-runpod:latest
```

## Step 4: Make the Image Public (Optional but Recommended)

By default, the pushed image might be marked as private on your GitHub profile.
1. Go to your GitHub profile -> **Packages**.
2. Click on `beat-it-runpod`.
3. Go to **Package Settings** and change visibility to **Public**. *(This allows RunPod to easily pull the image without requiring complicated secret-key setups in RunPod).*

## Step 5: Deploy on RunPod

1. Go to your RunPod dashboard and rent a GPU Pod.
2. In the setup wizard under **Container Image**, just paste your image URL:
   `ghcr.io/YOUR_GITHUB_USERNAME/beat-it-runpod:latest`
3. Under **Volume Mounts**, ensure your persistent pod volume is mapped to:
   - `/app/models`
   - `/app/runs`
4. Under **Expose HTTP Ports**, add `6006` so you can click the web link directly in RunPod to view your Tensorboard logs.

RunPod will now fetch your image directly from GitHub, spin it up, restore your most recent checkpoint from the volume, and begin actively training!
