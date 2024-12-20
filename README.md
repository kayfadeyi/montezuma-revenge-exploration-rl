# Montezuma's Revenge Deep Reinforcement Learning

This project implements an advanced Deep Reinforcement Learning solution for Montezuma's Revenge, leveraging curiosity-driven exploration, enhanced GPU optimisation, and modern RL techniques.

## Features

- Double DQN with Dueling Architecture
- Prioritised Experience Replay
- Intrinsic Curiosity Modules for Exploration
- Frame Stacking and Preprocessing
- Advanced Reward Shaping
- Comprehensive Logging, Checkpointing, and Visualisations

## Prerequisites

- Python 3.11 or newer
- macOS or Linux
- At least 8GB RAM
- NVIDIA GPU with CUDA support (tested on AWS p3.2xlarge with Tesla V100)

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
   pip install -r requirements.txt
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
├── train_enhanced.py    # Training script
├── evaluate.py          # Evaluation script
├── results/             # Training results and visualisations
└── requirements.txt     # Project dependencies
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
python train_enhanced.py
```

The training process includes:

- Automatic checkpointing every 100 episodes
- Progress logging
- Graph generation for training metrics
- Intrinsic and extrinsic reward analysis

#### Training Parameters

Configurable in `train_enhanced.py`:

- `num_episodes`: Number of episodes (default: 50,000)
- `batch_size`: Training batch size (default: 32)
- `gamma`: Discount factor (default: 0.99)
- `epsilon_decay_steps`: Exploration decay steps (default: 200,000)
- `target_update_frequency`: Target update frequency (default: 5,000)
- `learning_rate`: Learning rate (default: 0.0001)

### Evaluation

To evaluate a trained model:
```bash
python evaluate.py --model_path checkpoints/model_best.pt
```

## Monitoring Progress

### Training Logs

Check the `logs/` directory for detailed training logs. Logs include:
- Rewards (extrinsic and intrinsic)
- Steps
- Exploration rate

### Results and Graphs

Training results are saved in the `results/` directory:

1. **Cumulative Rewards Over Episodes:** Tracks total rewards, highlighting improvement phases.
2. **Frames Per Reward:** Visualises efficiency gains during training.
3. **State Coverage Heatmaps:** Illustrates exploration density at early and late phases.
4. **Intrinsic vs Extrinsic Rewards:** Shows the dynamics of exploration and gameplay learning.

All graphs are generated automatically at the end of training.

### Checkpoints

- Best model saved to `checkpoints/model_best.pt`
- Regular checkpoints saved every 100 episodes
- Checkpoints include:
  - Model state
  - Optimiser state
  - Training statistics

### Early Stopping

Training stops if:
- No improvement for 5,000 episodes
- Maximum reward threshold is reached
- Completes all episodes

## ESG Implications

This project incorporates ESG considerations:

**Environmental:**
- GPU optimisation reduces energy usage.
- Potential to explore renewable-powered cloud setups in future work.

**Social:**
- Makes RL research accessible through affordable cloud resources.
- Contributions to open-source RL frameworks for broader benefits.

**Governance:**
- Adheres to ethical AI standards in research and development.
- Transparent sharing of methodologies and results.

## Troubleshooting

### Common Issues and Solutions

#### Out of Memory

- Reduce batch size.
- Decrease replay buffer size.

#### Environment Creation Fails

Ensure proper installation of Atari ROMs using:
```bash
AutoROM --accept-license
```

#### Training Stability

- Adjust learning rate.
- Modify exploration parameters.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This implementation combines techniques from:

- "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
- "Curiosity-driven Exploration by Self-supervised Prediction" (Pathak et al., 2017)
- "Prioritised Experience Replay" (Schaul et al., 2016)
- "Go-Explore: A New Approach for Hard-Exploration Problems" (Ecoffet et al., 2021)

## References

1. Van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. *Proceedings of the AAAI Conference on Artificial Intelligence, 30*(1), 2094-2100.
2. Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. *International Conference on Machine Learning*, 1995-2003.
3. Pathak, D., Agrawal, P., Efros, A.A., & Darrell, T. (2017). Curiosity-driven exploration by self-supervised prediction. *International Conference on Machine Learning*, 2778-2787.
4. Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2019). Exploration by random network distillation. *International Conference on Learning Representations*.
5. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized experience replay. *International Conference on Learning Representations*.
6. Ecoffet, A., Huizinga, J., Lehman, J., Stanley, K.O., & Clune, J. (2019). Go-explore: a new approach for hard-exploration problems. *arXiv preprint arXiv:1901.10995*.
7. Kulkarni, T.D., Narasimhan, K., Saeedi, A., & Tenenbaum, J. (2016). Hierarchical deep reinforcement learning: Integrating temporal abstraction and intrinsic motivation. *Advances in Neural Information Processing Systems, 29*, 3675-3683.
8. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G., & Petersen, S. (2015). Human-level control through deep reinforcement learning. *Nature, 518*(7540), 529-533.
9. Pathak, D., Agrawal, P., Efros, A.A., & Darrell, T. (2017). Curiosity-driven exploration by self-supervised prediction. *International Conference on Machine Learning*, 2778-2787.

