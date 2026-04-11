# RunPod & GitHub Infrastructure Setup

This guide contains the **one-time setup** steps required to link your local compilation environment to GitHub Container Registry (GHCR) and to configure the RunPod Template. 

> [!TIP]
> If you have already completed these steps, refer to `DAILY_WORKFLOW.md` for your daily development and training loops.

## Step 1: Authenticate with GitHub Container Registry (GHCR)

You can use GitHub Container Registry (GHCR) for free with your `monade@gmail.com` GitHub account. To push images from your Windows WSL compiling machine (or Mac) to GHCR, you need to authenticate Docker.

1. Go to your GitHub profile settings -> **Developer settings** -> **Personal access tokens** -> **Tokens (classic)**.
2. Generate a new token with the `write:packages` and `read:packages` scopes.
3. On your compiling machine terminal, log in to Docker using that token:
   ```bash
   echo "YOUR_GITHUB_TOKEN" | docker login ghcr.io -u monadegmailcom --password-stdin
   ```

## Step 2: Make the Image Public (Optional but Recommended)

By default, any pushed image might be marked as private on your GitHub profile. To avoid needing complicated RunPod registry secrets:
1. After your first push, go to your GitHub profile -> **Packages**.
2. Click on `beat-it-runpod`.
3. Go to **Package Settings** and change visibility to **Public**.

## Step 3: Create a RunPod Template

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

---
**All Set!** Proceed to `DAILY_WORKFLOW.md` for your workflow loops.
