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
- Prioritised experience replay using sum tree
- Importance sampling for unbiased updates
- Priority annealing during training

### References
Van Hasselt, H., Guez, A. and Silver, D., 2016. Deep reinforcement learning with double Q-learning. Proceedings of the AAAI Conference on Artificial Intelligence, 30(1), pp.2094-2100.

Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M. and Freitas, N., 2016. Dueling network architectures for deep reinforcement learning. International Conference on Machine Learning, pp.1995-2003.

Pathak, D., Agrawal, P., Efros, A.A. and Darrell, T., 2017. Curiosity-driven exploration by self-supervised prediction. International Conference on Machine Learning, pp.2778-2787.

Burda, Y., Edwards, H., Storkey, A. and Klimov, O., 2019. Exploration by random network distillation. International Conference on Learning Representations.

Schaul, T., Quan, J., Antonoglou, I. and Silver, D., 2016. Prioritized experience replay. International Conference on Learning Representations.

Ecoffet, A., Huizinga, J., Lehman, J., Stanley, K.O. and Clune, J., 2019. Go-explore: a new approach for hard-exploration problems. arXiv preprint arXiv:1901.10995.

Kulkarni, T.D., Narasimhan, K., Saeedi, A. and Tenenbaum, J., 2016. Hierarchical deep reinforcement learning: Integrating temporal abstraction and intrinsic motivation. Advances in Neural Information Processing Systems, 29, pp.3675-3683.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G. and Petersen, S., 2015. Human-level control through deep reinforcement learning. Nature, 518(7540), pp.529-533.

## License

MIT
