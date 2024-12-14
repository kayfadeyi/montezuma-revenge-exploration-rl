# Montezuma's Revenge Deep Reinforcement Learning

This project implements an advanced Deep Reinforcement Learning solution for Montezuma's Revenge using curiosity-driven exploration and modern RL techniques.

## Features

- Double DQN with Dueling Architecture
- Prioritized Experience Replay
- Curiosity-Driven Exploration
- Frame Stacking and Preprocessing
- Advanced Reward Shaping
- Comprehensive Logging and Checkpointing

## Prerequisites

- Python 3.11 or newer
- macOS or Linux
- At least 8GB RAM
- (Optional) NVIDIA GPU with CUDA support

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kayfadeyi/montezuma-revenge-exploration-rl.git
   cd montezuma-revenge-exploration-rl
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install gymnasium[atari]>=1.0.0
   pip install torch>=1.9.0
   pip install opencv-python>=4.5.3
   pip install numpy>=1.19.5
   pip install tqdm>=4.62.0
   pip install matplotlib>=3.4.3
   pip install ale-py>=0.10.1
   pip install shimmy>=0.2.1
   pip install "AutoROM[accept-rom-license]"
   ```

## Project Structure

```plaintext
montezuma-revenge-exploration-rl/
├── montezuma_rl/
│   ├── models/          # Neural network architectures
│   ├── utils/           # Helper functions and preprocessing
│   ├── memory/          # Experience replay implementations
│   ├── exploration/     # Curiosity-driven exploration
├── checkpoints/         # Saved models
├── logs/                # Training logs
├── train.py             # Training script
├── evaluate.py          # Evaluation script
└── requirements.txt     # Project dependencies
└── videos/              # Evaluation videos
```

## Running the Project

### Testing Installation

Verify the environment setup:
```bash
python test_play.py
```
This should open a window showing Montezuma's Revenge with random actions.

### Training

Start training:
```bash
python train.py
```

The training process includes:

- Automatic checkpointing every 100 episodes
- Progress logging
- Early stopping conditions
- Performance metrics tracking
- Curiosity-driven exploration
- Adaptive reward scaling

#### Training Parameters

Configurable in `train.py`:

- `num_episodes`: Number of episodes (default: 2000)
- `batch_size`: Training batch size (default: 32)
- `gamma`: Discount factor (default: 0.99)
- `initial_exploration`: Starting exploration rate (default: 1.0)
- `final_exploration`: Final exploration rate (default: 0.01)
- `exploration_steps`: Number of exploration steps (default: 2000000)
- `target_update_frequency`: Target update frequency (default: 5000)
- `learning_rate`: Learning rate (default: 0.0001)
- `min_exploration_ratio`: Minimum exploration ratio (default: 0.9)

### Evaluation

To evaluate a trained model:
```bash
python evaluate.py --model_path checkpoints/model_best.pt
```

## Monitoring Progress

### Training Logs

Check the `logs/` directory for detailed training logs. Logs include:
- Rewards
- Steps
- Exploration rate

### Checkpoints

- Best model saved to `checkpoints/model_best.pt`
- Regular checkpoints saved every 100 episodes
- Checkpoints include:
  - Model state
  - Optimizer state
  - Training statistics

### Early Stopping

Training will automatically stop if:
- No improvement for 1000 episodes
- Reaches reward threshold (100)
- Completes maximum episodes

## Customization

### Network Architecture

- Modify `models/dqn.py` for network changes
- Adjust layer sizes and activation functions

### Exploration Strategy

- Modify `exploration/curiosity.py`
- Adjust intrinsic reward scaling
- Change feature encoding

### Memory Management

- Adjust buffer size in `memory/prioritized_replay.py`
- Modify prioritization strategy
- Change sampling parameters

### Training Parameters

- Adjust learning rate
- Modify batch size
- Change exploration schedule

## Troubleshooting

### Common Issues and Solutions

#### ROM Loading Errors
```bash
python -m AutoROM --accept-license
```

#### Out of Memory

- Reduce batch size
- Decrease replay buffer size
- Use CPU if GPU memory is limited

#### Environment Creation Fails
```bash
python test_env.py
```

#### Training Stability

- Adjust learning rate
- Modify exploration parameters
- Change network architecture

### Evaluation

To evaluate a trained model, use the evaluate.py script. Here are the basic usage patterns:

#### Basic Evaluation
```bash
# Basic evaluation with rendering
python evaluate.py --model_path checkpoints/model_best.pt --render

# Run specific number of episodes without rendering
python evaluate.py --model_path checkpoints/model_best.pt --episodes 20

# Run multiple episodes with rendering
python evaluate.py --model_path checkpoints/model_best.pt --episodes 20 --render
```

#### Evaluation Parameters

- `--model_path`: Path to checkpoint file (required)
- `--episodes`: Number of evaluation episodes (default: 10)
- `--render`: Flag to enable visualization of gameplay (optional)

#### Evaluation Metrics

The evaluation provides the following metrics:

- Episode rewards and steps
- Unique positions visited per episode
- Average reward across all episodes
- Average steps across all episodes
- Maximum reward achieved
- Maximum steps reached in any episode

The evaluation results are printed to the console after completion, showing both per-episode statistics and overall performance metrics.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This implementation combines techniques from:

- "Human-level control through deep reinforcement learning" (DQN)
- "Deep Reinforcement Learning with Double Q-learning" (Double DQN)
- "Dueling Network Architectures for Deep Reinforcement Learning" (Dueling DQN)
- "Curiosity-driven Exploration by Self-supervised Prediction" (Intrinsic Rewards)
- "Prioritized Experience Replay" (PER)

## References

1. Van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. *Proceedings of the AAAI Conference on Artificial Intelligence, 30*(1), 2094-2100.
2. Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. *International Conference on Machine Learning*, 1995-2003.
3. Pathak, D., Agrawal, P., Efros, A.A., & Darrell, T. (2017). Curiosity-driven exploration by self-supervised prediction. *International Conference on Machine Learning*, 2778-2787.
4. Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2019). Exploration by random network distillation. *International Conference on Learning Representations*.
5. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized experience replay. *International Conference on Learning Representations*.
6. Ecoffet, A., Huizinga, J., Lehman, J., Stanley, K.O., & Clune, J. (2019). Go-explore: a new approach for hard-exploration problems. *arXiv preprint arXiv:1901.10995*.
7. Kulkarni, T.D., Narasimhan, K., Saeedi, A., & Tenenbaum, J. (2016). Hierarchical deep reinforcement learning: Integrating temporal abstraction and intrinsic motivation. *Advances in Neural Information Processing Systems, 29*, 3675-3683.
8. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G., & Petersen, S. (2015). Human-level control through deep reinforcement learning. *Nature, 518*(7540), 529-533.

