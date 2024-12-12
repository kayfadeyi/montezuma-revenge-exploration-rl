#!/bin/bash

# Go to the root directory of the project
cd "$(dirname "$0")/.."

<<<<<<< HEAD
# Run training script
python3 train.py
=======
# Go to the root directory of the project
cd "$(dirname "$0")/.."

# Add the current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run training
python3 train.py
>>>>>>> 6ef4dbf (Complete implementation with working environment setup, training, and evaluation)
