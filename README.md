# beat-it
ki engine for two player games
- install prerequisites with homebrew
  - brew install libomp
- install python venv with in project dir (Python 3.12 or 3.11 is recommended, as 3.13 has breaking changes)
  - brew install python@3.12
  - python3.12 -m venv .venv
  - source .venv/bin/activate
  - pip3 install torch torchvision tensorboard matplotlib
   the mac make the gatekeeper happy
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
## Inspect model metadata file
```bash
unzip -p models/ttt_alphazero_experiment_6/final_model.pt final_model/extra/metadata.json | jq
```

## Resume training
1. **Resume training from checkpoint**
  ```bash
  python ttt_training.py --resume_from runs/models/ttt_alphazero_experiment_2/checkpoint.pt
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
- make `play_match` parallel

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
   