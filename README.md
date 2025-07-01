# beat-it
ki engine for two player games
- install prerequisites with homebrew
  - brew install libomp
- install python venv with in project dir
  - python3 -m venv .venv
  - source .venv/bin/activate
  - pip3 install torch torchvision
on the mac make the gatekeeper happy
    - xattr -d com.apple.quarantine /Users/wrqpjzc/source/libtorch/lib/*.dylib
