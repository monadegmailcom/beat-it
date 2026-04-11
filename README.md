# next steps
- fix training setup
  - configuration file working?
  - loading and saving models working?
  - checkpoints working?
  - tensorboard working?
- fix additional statistics for tensorboard
- add arena games old model vs new model
- setup docker container
- starting training on kaggle
- training on tristans machine?
- add gpu + powersupply to linux server

# beat-it
ki engine for two player games

## 📚 Documentation Quick Links
- **[Daily Training Workflow](DAILY_WORKFLOW.md)**: Your step-by-step cheatsheet for building, deploying, optimizing, and training on RunPod via Tmux/SSH.
- **[RunPod & GitHub Infrastructure](INFRASTRUCTURE_SETUP.md)**: One-time setup instructions for GitHub Container Registry and RunPod templates.
- **[Optuna Tradeoffs](OPTUNA_TRADEOFF.md)**: Guide on how to interpret optimization metrics.

## Requirements on mac
- install prerequisites with homebrew
  - brew install libomp
- install python venv with in project dir (Python 3.12 or 3.11 is recommended, as 3.13 has breaking changes)
  - brew install python@3.12
  - python3.12 -m venv .venv
  - source .venv/bin/activate
  - `pip install -r train/requirements.txt`
  - on mac make the gatekeeper happy
    - xattr -d com.apple.quarantine /Users/wrqpjzc/source/libtorch/lib/*.dylib
  - regenerate compile_commands.json with bear -- make

## Requirements on linux (ubuntu 25.04)
- sudo apt-get install libboost-all-dev
- ubuntu has python 3.13 installed, but this version is broken with pytorch
- pytorch recommends python 3.12
- ubuntu 25.04 does not have python 3.12 as an apt package, repo deadsnakes/ppa neither
- so install pyenv:
  - sudo apt update
  -  sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl
  - curl https://pyenv.run | bash
  - edit ~/.profile as suggested by the command above
  - exec $SHELL
- pyenv install -v 3.12.9
- pyenv local 3.12.9
- python --version
- python -m venv .venv
- . .venv/bin/activate
- `pip install -r train/requirements.txt`

## setup google compute engine (debian based)
- install gemini
  - `sudo npm install -g @google/gemini-cli`
  - return url to authorize api key in browser
- install newer gcc
  - `gcc --version` gives 10.2.1 (no c++ 23 support)
  - add testing repo, standard apt repo does not have it
    - `echo "deb http://deb.debian.org/debian testing main" | sudo tee                                │
 │   /etc/apt/sources.list.d/testing.list`
  - apt pinning
    - `echo 'Package: *\nPin: release a=stable\nPin-Priority: 900\n\nPackage: *\nPin:                 │
 │   release a=testing\nPin-Priority: 300\n' | sudo tee                                             │
 │   /etc/apt/preferences.d/99-testing-pref`
    - `sudo apt-get update`
    - `sudo apt-get install -t testing g++-12 -y`
    - `apt --fix-broken install`
  - install boost
    - `sudo apt-get install -y libboost-json-dev`
  - setup venv and install torch in it
    - `pip install -r train/requirements.txt`
  - to upload file with the ssh-in-browser tool you may retry to update outdated ssh keys


## Docker Setup
```bash
# Install Lima and Docker CLI
brew install lima docker
# Start a Linux instance with Docker support in the lima config
limactl start docker-rootful.yaml
# Configure docker CLI to use Lima
export DOCKER_HOST="unix://$(limactl list docker --format '{{.Dir}}/sock/docker.sock')"
```

### Quick Start (Local CPU)
```bash
docker build -t beat-it:cpu .
docker run -it -p 6006:6006 -v $(pwd)/runs:/app/runs beat-it:cpu
```

### Start on training machine (GPU)
- goto to pytorch.org and configure in the link section on the page buttom, e.g. for windows:
  - Stable(2.9.1) -> Windows -> LibTorch -> C++/Java -> CUDA 12.6 -> Link 
```bash
docker build -t beat-it:gpu .
```

## How to Run

### On the target machine interactive shell in docker
```bash
docker run -it --rm \
  -p 6006:6006 \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/models:/app/models \
  beat-it:gpu
```

### Natively on bare metal

-  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```
- **Configure training parameter in uttt_config.json**
    - get number of cpu cores: `nproc`

-  **Build and run the training:**
    ```bash
    ./build.sh && python -m train.main --game uttt
    ```
    *(Note: The build script handles CMake configuration and compilation. If you prefer manual steps, see `build.sh` content.)*
-  **Monitor with TensorBoard (in a separate terminal inside the virtual environment):**
    ```bash
    tensorboard --logdir=runs
    ```
    - open tensorboard website from another machine
      - ssh -L 8888:localhost:6006 x10dri
      - open browser at http://localhost:8888

- **Export linux Window Manager**
  - install novnc and websockify: `sudo apt install novnc websockify`
  - start desktop session on linux: `vncserver --localhost`
  - note display number, e.g. `:1`
  - start web bridge (no ssh tunnel required): ` websockify -D --web /usr/share/novnc/ 6080 localhost:5901`
  - open in chrome: `http://x10dri:6080/vnc.html`

## Inspect model metadata file
```bash
unzip -p models/ttt_alphazero_experiment_6/final_model.pt final_model/extra/metadata.json | jq
```

## Resume training
### Resume training from checkpoint**
  ```bash
  python -m train.main --game uttt --resume_from experiment_name/checkpoint.pt 2>&1 | tee -a output.log
  ```

### Extend training steps**
  - overwrite configurations in train/main.py and train/uttt.py
  - resume training in point 1.

## Monitor machine
- monitor cpu, memory and gpu usage: `watch 'ps -p $(pgrep python) -o %cpu,%mem ; nvidia-smi'`
- `sensors`

## Hyperparameter tuning & Cloud Training
> [!NOTE]
> The detailed cloud hyperparameter tuning and model training processes have been streamlined. 
> Please follow the **[Daily Training Workflow](DAILY_WORKFLOW.md)** for execution.
 
# Lock-free techniques
- use boost::lockfree::queue for a producer-consumer-pattern
  - a failed mutex lock slows down a thread for orders of magnitude
  - note: a lock-free queue has a fixed maximal size
- use semaphores for maintaining resources
  - a resource in this context is
    - the number of free slots in a lock-free queue
    - the number of requests to be processed
    - the number of notifications to deliver
  - a semaphore can be used to let a thread sleep until a resource is free without busy-spinning.
- a lock-free queue with a semaphore should be a high performance replacement for mutex and condition variables.
- exit a blocking semaphore acquire by sending a "poison-pill" dummy value.
   
## Further potential optimizations
- make virtual_loss a hyperparameter
- try Int8 arithmetic for gpu
- use subnormal number optimization
```cpp
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <xmmintrin.h> // For SSE intrinsics to control floating-point behavior
#endif
  ```
-  in the code before using libtorch
   ```cpp
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    // Enable Flush-to-Zero and Denormals-are-Zero modes for this inference thread.
    // This can significantly speed up inference, especially with untrained
    // models that might produce many subnormal floating-point values.
    _mm_setcsr(_mm_getcsr() | 0x8040);
#endif
```
## vscode & neovim vscode extension
- fix keystroke repeat to reach vscode in bash: 
  - `defaults write com.microsoft.VSCode ApplePressAndHoldEnabled -bool false`
- toggle side bar: `⌘B`
- gemini chat: `⌘I`
- switch file: `⌘P`
- switch group: `⌘[n]` 
