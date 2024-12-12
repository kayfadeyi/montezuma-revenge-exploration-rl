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

## Create and activate a virtual environment:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

## Install dependencies:
pip install gymnasium[atari]>=0.29.1
pip install torch>=1.9.0
pip install opencv-python>=4.5.3
pip install numpy>=1.19.5
pip install tqdm>=4.62.0
pip install matplotlib>=3.4.3
pip install ale-py>=0.8.0
pip install shimmy>=0.2.1
pip install "AutoROM[accept-rom-license]"

## Download and install ROMs:
python -m AutoROM --accept-license

## Project Structure
montezuma-revenge-exploration-rl/
├── montezuma_rl/
│   ├── models/          # Neural network architectures
│   ├── utils/           # Helper functions and preprocessing
│   ├── memory/          # Experience replay implementations
│   ├── exploration/     # Curiosity-driven exploration
├── checkpoints/         # Saved models
├── logs/               # Training logs
├── train.py           # Training script
├── evaluate.py        # Evaluation script
└── requirements.txt   # Project dependencies

## Running the Project
## Testing Installation

## Verify environment setup:
python test_play.py
## This should open a window showing Montezuma's Revenge with random actions.

## Training

## Start training:
python train.py

## The training includes:

## Automatic checkpointing every 100 episodes
## Progress logging
## Early stopping conditions
## Performance metrics tracking
## Curiosity-driven exploration
## Adaptive reward scaling

## Training parameters (configurable in train.py):

## num_episodes: Number of episodes (default: 2000)
## batch_size: Training batch size (default: 32)
## gamma: Discount factor (default: 0.99)
## initial_exploration: Starting exploration rate (default: 1.0)
## final_exploration: Final exploration rate (default: 0.01)

## Evaluation
## To evaluate a trained model:
python evaluate.py --model_path checkpoints/model_best.pt

## Monitoring Progress

## Training Logs


## Check logs/ directory for detailed training logs
## Logs include rewards, steps, exploration rate


## Checkpoints


## Best model saved to checkpoints/model_best.pt
## Regular checkpoints saved every 100 episodes
## Checkpoints include:

## Model state
## Optimizer state
## Training statistics




## Early Stopping
## Training will automatically stop if:


## No improvement for 1000 episodes
## Reaches reward threshold (100)
## Completes maximum episodes

## Customization
## The project is modular and can be customized:

## Network Architecture


## Modify models/dqn.py for network changes
## Adjust layer sizes, activation functions


## Exploration Strategy


## Modify exploration/curiosity.py
## Adjust intrinsic reward scaling
Change feature encoding


## Memory Management


## Adjust buffer size in memory/prioritized_replay.py
## Modify prioritization strategy
## Change sampling parameters


## Training Parameters


## Adjust learning rate
## Modify batch size
## Change exploration schedule

## Troubleshooting
## Common issues and solutions:

## ROM Loading Errors
##  Reinstall ROMs
python -m AutoROM --accept-license

## Out of Memory


## Reduce batch size
## Decrease replay buffer size
## Use CPU if GPU memory is limited


## Environment Creation Fails
##  Verify environment
python test_env.py

## Training Stability


## Adjust learning rate
## Modify exploration parameters
## Change network architecture

## License
## This project is licensed under the MIT License - see the LICENSE file for details.
## Acknowledgments
## This implementation combines techniques from:

## "Human-level control through deep reinforcement learning" (DQN)
## "Deep Reinforcement Learning with Double Q-learning" (Double DQN)
## "Dueling Network Architectures for Deep Reinforcement Learning" (Dueling DQN)
## "Curiosity-driven Exploration by Self-supervised Prediction" (Intrinsic Rewards)
## "Prioritized Experience Replay" (PER)

### References
Van Hasselt, H., Guez, A. and Silver, D., 2016. Deep reinforcement learning with double Q-learning. Proceedings of the AAAI Conference on Artificial Intelligence, 30(1), pp.2094-2100.

Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M. and Freitas, N., 2016. Dueling network architectures for deep reinforcement learning. International Conference on Machine Learning, pp.1995-2003.

Pathak, D., Agrawal, P., Efros, A.A. and Darrell, T., 2017. Curiosity-driven exploration by self-supervised prediction. International Conference on Machine Learning, pp.2778-2787.

Burda, Y., Edwards, H., Storkey, A. and Klimov, O., 2019. Exploration by random network distillation. International Conference on Learning Representations.

Schaul, T., Quan, J., Antonoglou, I. and Silver, D., 2016. Prioritized experience replay. International Conference on Learning Representations.

Ecoffet, A., Huizinga, J., Lehman, J., Stanley, K.O. and Clune, J., 2019. Go-explore: a new approach for hard-exploration problems. arXiv preprint arXiv:1901.10995.

Kulkarni, T.D., Narasimhan, K., Saeedi, A. and Tenenbaum, J., 2016. Hierarchical deep reinforcement learning: Integrating temporal abstraction and intrinsic motivation. Advances in Neural Information Processing Systems, 29, pp.3675-3683.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G. and Petersen, S., 2015. Human-level control through deep reinforcement learning. Nature, 518(7540), pp.529-533.

