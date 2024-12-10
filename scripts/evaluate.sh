#!/bin/bash

# Activate virtual environment if needed
# source venv/bin/activate

# Run evaluation
python src/evaluate.py --model_path checkpoints/model_best.pt --episodes 10 --render