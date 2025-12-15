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

## Patch paths for local machine
- create patch:
  - `diff -u /home/monade/source/beat-it/Makefile /home/monade/source/beat-it/Makefile.modified > /home/monade/source/beat-it/libtorch.patch`
- run patch
  - `patch < libtorch.patch`

## How to Run

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

## Hyperparameter tuning

- before starting a training you should optimize hyperparameters for the number of parallel self plays, virtual loss selfplay threads and minimal nn evaluation batch size.
- `make shared && python -m train.opt_selfplay \
    --model_path models/uttt_alphazero_experiment_2/final_model.pt \
    --n_trials 100 \
    --simulations_per_move 800 \
    --number_of_games 20\
    --study_name my_name`
- `optuna-dashboard sqlite:///db.sqlite3`
- open browser at the output url
- find the sweet spot of 
  - intra game parallelism (selfplay threads)
  - inter game parallelism (parallel game play)
  - gpu utilization (batch_size)
  - well-guided search (shannon entropy of root-children-visit distribution)
- "flying blind" -> high entropy
- well-guided search -> low entropy
- find the pareto front by simultaneously
  - maximizing throughput
  - minimizing shannon entropy
 
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
   
## Todos
- regularly evaluate game play in a match
  - add match function to shared lib
  - every time a new checkpoint model is saved start a match (100 games) against the previous model and output stats to console.
- set up docker for ease of deployment
  - optimizing selfplay
  - training
  - interactive shell into docker
  - questions
    - is the gpu accessible from docker?
    - how to transfer files to/from docker
    - how to access tensorboard and optuna-dashboard

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