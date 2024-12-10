# Montezuma's Revenge with Curiosity-Driven Exploration

This repository implements a Deep Reinforcement Learning agent for Montezuma's Revenge using curiosity-driven exploration. The implementation combines several advanced techniques:

- Double DQN with Dueling Architecture
- Prioritized Experience Replay
- Curiosity-Driven Exploration
- Frame Stacking and Preprocessing

## Installation

```bash
# Clone the repository
git clone https://github.com/kayfadeyi/montezuma-revenge-exploration-rl.git
cd montezuma-revenge-exploration-rl

# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Run training script
bash scripts/train.sh
```

The training script will:
- Create necessary directories for checkpoints and logs
- Train the agent using curiosity-driven exploration
- Save periodic checkpoints and the best model

### Evaluation

```bash
# Run evaluation script
bash scripts/evaluate.sh
```

The evaluation script will:
- Load a trained model
- Run evaluation episodes
- Generate plots and metrics
- Save results in the evaluation_results directory

## Project Structure

```
.
├── src/
│   ├── models/          # Neural network architectures
│   ├── utils/           # Helper functions and preprocessing
│   ├── memory/          # Experience replay implementations
│   ├── exploration/     # Curiosity-driven exploration
│   ├── train.py         # Training script
│   └── evaluate.py      # Evaluation script
├── scripts/             # Shell scripts for running experiments
├── checkpoints/         # Saved models
└── evaluation_results/  # Evaluation metrics and plots
```

## Implementation Details

### DQN Architecture
- Convolutional layers for processing 84x84 grayscale frames
- Dueling architecture separating value and advantage streams
- Double Q-learning for more stable training

### Curiosity Module
- Forward dynamics model predicting next state features
- Intrinsic rewards based on prediction error
- Normalized reward combination

### Experience Replay
- Prioritized experience replay using sum tree
- Importance sampling for unbiased updates
- Priority annealing during training

## License

MIT
