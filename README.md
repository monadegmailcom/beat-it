# beat-it
ki engine for two player games
- install prerequisites with homebrew
  - brew install libomp
- install python venv with in project dir (Python 3.12 or 3.11 is recommended, as 3.13 has breaking changes)
  - brew install python@3.12
  - python3.12 -m venv .venv
  - source .venv/bin/activate
  - pip3 install torch torchvision tensorboard
on the mac make the gatekeeper happy
    - xattr -d com.apple.quarantine /Users/wrqpjzc/source/libtorch/lib/*.dylib

## How to Run

1.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```
2.  **Build and run the training:**
    ```bash
    make shared && python ttt_training.py
    ```
3.  **Monitor with TensorBoard (in a separate terminal inside the virtual environment):**
    ```bash
    tensorboard --logdir=runs
    ```
