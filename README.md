# beat-it
ki engine for two player games

## Requirements on mac
- install prerequisites with homebrew
  - brew install libomp
- install python venv with in project dir (Python 3.12 or 3.11 is recommended, as 3.13 has breaking changes)
  - brew install python@3.12
  - python3.12 -m venv .venv
  - source .venv/bin/activate
  - pip3 install torch torchvision tensorboard matplotlib flake8 mypy
  - on mac make the gatekeeper happy
    - xattr -d com.apple.quarantine /Users/wrqpjzc/source/libtorch/lib/*.dylib

## Requirements on linux (ubuntu 25.04)
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
- pip3 install torch torchvision tensorboard matplotlib flake8 mypy

## How to Run

-  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```
- **Configure training in train/main.py**
    - get number of cpu cores: `nproc`

-  **Build and run the training:**
    ```bash
    make shared && python -m train.main --game uttt
    ```
-  **Monitor with TensorBoard (in a separate terminal inside the virtual environment):**
    ```bash
    tensorboard --logdir=runs
    ```
    - open tensorboard website from another machine
      - ssh -L 8888:localhost:6006
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
1. **Resume training from checkpoint**
  ```bash
  python -m train.main --game ttt --resume_from runs/models/ttt_alphazero_experiment_2/checkpoint.pt
  ```
2. **Extend training steps**
  - extract metadata.json :
  ```bash
  mkdir -p checkpoint/extra
  unzip -p runs/models/ttt_alphazero_experiment_2/checkpoint.pt checkpoint/extra/metadata.json > checkpoint/extra/metadata.json
  ```
  - edit metadata.json and replace hyperparameter "total_training_steps" with a higher value
  - replace metadata.json in the model file:
  ```bash
  zip -u runs/models/ttt_alphazero_experiment_2/checkpoint.pt checkpoint/extra/metadata.json
  ```
  - resume training in point 1.

## Todos

- add weight_decay parameter to adam optimizer as hyperparameter
- use optuna for hyperparameter optimization

## Further potential optimizations

1. use subnormal number optimization
```cpp
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <xmmintrin.h> // For SSE intrinsics to control floating-point behavior
#endif
  ```
   2. in the code before using libtorch
   ```cpp
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    // Enable Flush-to-Zero and Denormals-are-Zero modes for this inference thread.
    // This can significantly speed up inference, especially with untrained
    // models that might produce many subnormal floating-point values.
    _mm_setcsr(_mm_getcsr() | 0x8040);
#endif
````
