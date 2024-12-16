import logging
import os
import shutil
from collections import deque
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from ale_py import ALEInterface
from gymnasium.wrappers import FrameStackObservation as FrameStack
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from tqdm import tqdm
import objgraph

from montezuma_rl.exploration.curiosity import CuriosityModule
from montezuma_rl.memory.prioritized_replay import PrioritizedReplayBuffer
from montezuma_rl.models.dqn import DuelingDQN
from montezuma_rl.utils.reward_combination import RewardCombiner


def calculate_step_bonus(steps):
    """Calculate progressive bonus for longer episodes with increased rewards"""
    if steps > 300:
        bonus = 0.2 * (steps / 300)
        if steps > 700:
            bonus += 0.4
        if steps > 1000:
            bonus += 0.6
        return bonus
    return 0.0


def load_checkpoint(path):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        logging.info(f"Loaded checkpoint from {path}")
        return checkpoint
    return None


def optimize_memory():
    """Free memory allocated by PyTorch."""
    import torch
    from torch.cuda import memory_allocated, memory_reserved

    torch.cuda.empty_cache()  # Clear cache to free unused memory
    allocated = memory_allocated()
    reserved = memory_reserved()
    logging.info(f"Memory allocated: {allocated / 1e6:.2f} MB")
    logging.info(f"Memory reserved: {reserved / 1e6:.2f} MB")


class TrainModel:
    def __init__(self, atari_environment, checkpoint_epoch=100, video_epoch=50):
        self.atari_environment = atari_environment
        self.checkpoint_epoch = checkpoint_epoch
        self.video_epoch = video_epoch

    def backup_checkpoint(self, src_path):
        """Create a backup of checkpoint file."""
        backup_dir = f'checkpoints/{self.atari_environment}/backups'
        if os.path.exists(src_path):
            Path(backup_dir).mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{backup_dir}/checkpoint_backup_{timestamp}.pt"
            shutil.copy2(src_path, backup_path)
            logging.info(f"Created checkpoint backup: {backup_path}")

    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(f'logs/{self.atari_environment}')
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(f'{log_dir}/training_{timestamp}.log'),
                logging.StreamHandler()
            ]
        )

    def train(self,
              num_episodes=10000,
              batch_size=32,
              gamma=0.99,
              initial_exploration=1.0,
              final_exploration=0.85,
              exploration_steps=2000000,
              target_update_frequency=10000,
              learning_rate=0.0001,
              min_exploration_ratio=0.9):
        """
        Training loop with optimizations for memory efficiency.
        """
        self.setup_logging()
        objgraph.show_growth(limit=10)

        # Backup existing checkpoint
        checkpoint_dir = f'checkpoints/{self.atari_environment}'
        checkpoint_path = f'{checkpoint_dir}/training_state.pt'
        self.backup_checkpoint(checkpoint_path)

        logging.info("Starting training with optimized parameters.")
        logging.info(f"Episodes: {num_episodes}, Batch size: {batch_size}, Gamma: {gamma}")
        logging.info(f"Exploration: {initial_exploration} -> {final_exploration} over {exploration_steps} steps")
        logging.info(f"Target update frequency: {target_update_frequency}")

        # Create directories
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        try:
            env = gym.make(self.atari_environment, render_mode='rgb_array', frameskip=1)  # No video during training
            env = AtariPreprocessing(
                env,
                noop_max=10,
                frame_skip=4,
                terminal_on_life_loss=True,
                screen_size=84,
                grayscale_obs=True,
                grayscale_newaxis=False
            )
            env = FrameStack(env, stack_size=4)
            env = RecordEpisodeStatistics(env)  # Only record statistics
            env = RecordVideo(env, video_folder=f"videos/{self.atari_environment}/training", name_prefix="training",
                              episode_trigger=lambda x: x % 250 == 0)

            logging.info("Environment created successfully")

        except Exception as e:
            logging.error(f"Failed to create environment: {e}")
            raise

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

        # Initialize networks
        online_net = DuelingDQN(num_actions=env.action_space.n).to(device)
        target_net = DuelingDQN(num_actions=env.action_space.n).to(device)
        target_net.load_state_dict(online_net.state_dict())

        curiosity = CuriosityModule(
            state_shape=(4, 84, 84),
            action_dim=env.action_space.n,
            device=device,
            learning_rate=learning_rate * 15
        )
        reward_combiner = RewardCombiner(beta=0.8)

        # Optimized replay buffer size
        memory = PrioritizedReplayBuffer(50000)  # Reduce size to save memory
        optimizer = torch.optim.Adam(online_net.parameters(), lr=learning_rate)

        # Training metrics
        best_reward = float('-inf')
        total_steps = 0
        recent_rewards = deque(maxlen=100)

        # Load checkpoint if exists
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            online_net.load_state_dict(checkpoint['online_net_state_dict'])
            target_net.load_state_dict(checkpoint['target_net_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            total_steps = checkpoint['total_steps']
            logging.info(f"Resuming from checkpoint at step {total_steps}")

        patience = 1500  # Early stopping patience
        episodes_without_improvement = 0

        logging.info("Starting training loop")
        for episode in tqdm(range(num_episodes), desc="Training"):
            state, info = env.reset()
            episode_reward = 0
            done = False
            truncated = False

            objgraph.show_growth(limit=10)
            if episode % 50 == 0:
                objgraph.show_backrefs(
                    objgraph.by_type('Tensor')[0],
                    filename=f'backrefs_episode_{episode}.png'
                )

            while not (done or truncated):
                # Compute epsilon for exploration
                epsilon = max(
                    final_exploration,
                    initial_exploration * (1 - total_steps / exploration_steps)
                )

                # Action selection
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = online_net(state_tensor).argmax().item()
                    del state_tensor  # Free memory

                # Environment step
                next_state, reward, done, truncated, info = env.step(action)

                # Compute intrinsic reward
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                intrinsic_reward = curiosity.compute_intrinsic_reward(
                    state_tensor,
                    torch.tensor([action]).to(device),
                    next_state_tensor
                )
                del state_tensor, next_state_tensor  # Free memory

                # Combine rewards
                combined_reward = reward_combiner.combine_rewards(
                    np.array([reward]),
                    intrinsic_reward.cpu().numpy()
                )[0]

                # Store transition in replay buffer
                memory.push(state, action, combined_reward, next_state, done)
                state = next_state
                episode_reward += reward
                total_steps += 1

                # Optimize network if memory has enough samples
                if len(memory) > batch_size:
                    batch, indices, weights = memory.sample(batch_size)
                    state_batch = torch.FloatTensor(np.array([t.state for t in batch])).to(device)
                    action_batch = torch.LongTensor([t.action for t in batch]).to(device)
                    reward_batch = torch.FloatTensor([t.reward for t in batch]).to(device)
                    next_state_batch = torch.FloatTensor(np.array([t.next_state for t in batch])).to(device)
                    done_batch = torch.FloatTensor([t.done for t in batch]).to(device)

                    # Compute Q values
                    current_q = online_net(state_batch).gather(1, action_batch.unsqueeze(1))
                    with torch.no_grad():
                        next_actions = online_net(next_state_batch).max(1)[1]
                        next_q = target_net(next_state_batch).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                        expected_q = reward_batch + gamma * next_q * (1 - done_batch)

                    # Compute loss
                    loss = (torch.tensor(weights, device=device) *
                            (current_q.squeeze() - expected_q.detach()) ** 2).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
                    optimizer.step()

                    priorities = (current_q.squeeze() - expected_q).abs().detach().cpu().numpy()
                    memory.update_priorities(indices, priorities)

                    del state_batch, action_batch, reward_batch, next_state_batch, done_batch  # Free memory

                # Update target network periodically
                if total_steps % target_update_frequency == 0:
                    target_net.load_state_dict(online_net.state_dict())
                    logging.info("Updated target network")
                    optimize_memory()  # Free memory

            # Track rewards and log
            recent_rewards.append(episode_reward)
            avg_reward = np.mean(list(recent_rewards))

            logging.info(f'Episode {episode} - Reward: {episode_reward:.2f}, '
                         f'Steps: {total_steps}, Avg Reward (100): {avg_reward:.2f}')

            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save({
                    'online_net_state_dict': online_net.state_dict(),
                    'target_net_state_dict': target_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'total_steps': total_steps
                }, f'{checkpoint_dir}/best_model.pt')
                logging.info("New best model saved.")

            if episodes_without_improvement >= patience:
                logging.info("Stopping early due to no improvement.")
                break

        env.close()
        logging.info("Training completed.")


if __name__ == '__main__':
    tm_breakout = TrainModel('ALE/Breakout-v5')
    tm_breakout.train()
