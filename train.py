import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
from datetime import datetime
import os
import shutil
from collections import deque

from montezuma_rl.models.dqn import DuelingDQN
from montezuma_rl.memory.prioritized_replay import PrioritizedReplayBuffer
from montezuma_rl.exploration.curiosity import CuriosityModule
from montezuma_rl.utils.reward_combination import RewardCombiner

# UPDATED: More aggressive step bonus calculation
def calculate_step_bonus(steps):
    """Calculate progressive bonus for longer episodes with increased rewards"""
    if steps > 300:  # Lowered threshold from 500
        bonus = 0.2 * (steps / 300)  # Doubled base bonus
        if steps > 700:  # Lowered threshold from 1000
            bonus += 0.4  # Doubled bonus
        if steps > 1000:  # Lowered threshold from 1500
            bonus += 0.6  # Doubled bonus
        return bonus
    return 0.0

def backup_checkpoint(src_path, backup_dir='checkpoints/backups'):
    """Create a backup of checkpoint file"""
    if os.path.exists(src_path):
        Path(backup_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{backup_dir}/checkpoint_backup_{timestamp}.pt"
        shutil.copy2(src_path, backup_path)
        logging.info(f"Created checkpoint backup: {backup_path}")

def load_checkpoint(path):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        logging.info(f"Loaded checkpoint from {path}")
        return checkpoint
    return None

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(f'logs/training_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )

def train(num_episodes=1500,  # Reduced from 2000 for faster iteration
          batch_size=32,
          gamma=0.99,
          initial_exploration=1.0,
          final_exploration=0.85,
          exploration_steps=2000000,
          target_update_frequency=5000,
          learning_rate=0.0001,
          min_exploration_ratio=0.9,
          checkpoint_path='checkpoints/training_state.pt'):
    """
    Enhanced training with aggressive exploration and rewards
    """
    setup_logging()
    
    # Backup existing checkpoint
    backup_checkpoint(checkpoint_path)
    
    logging.info("Starting training with enhanced aggressive parameters:")
    logging.info(f"Episodes: {num_episodes}, Batch size: {batch_size}, Gamma: {gamma}")
    logging.info(f"Exploration: {initial_exploration} -> {final_exploration} over {exploration_steps} steps")
    logging.info(f"Target update frequency: {target_update_frequency}")
    logging.info(f"Min exploration ratio: {min_exploration_ratio}")

    # Create directories
    Path('checkpoints').mkdir(exist_ok=True)

    # Initialize tracking with enhanced history
    position_history = set()  # Track unique positions
    last_steps = deque(maxlen=10)  # Increased from 5 for better momentum tracking
    position_streak = 0  # Track consecutive new positions

    try:
        logging.info("Creating environment: MontezumaRevenge-v4")
        env = gym.make('MontezumaRevenge-v4',
                      frameskip=4,
                      render_mode=None)
        
        env = AtariPreprocessing(env, 
                               grayscale_obs=True,
                               frame_skip=1,
                               scale_obs=True)
        env = FrameStack(env, num_stack=4)
        logging.info("Environment created successfully")
        
    except Exception as e:
        logging.error(f"Failed to create environment: {e}")
        raise

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Initialize networks
    online_net = DuelingDQN().to(device)
    target_net = DuelingDQN().to(device)
    target_net.load_state_dict(online_net.state_dict())

    # Enhanced curiosity module with higher learning rate
    curiosity = CuriosityModule(
        state_shape=(4, 84, 84),
        action_dim=env.action_space.n,
        device=device,
        learning_rate=learning_rate * 15  # Increased multiplier
    )
    reward_combiner = RewardCombiner(beta=0.8)

    # Larger replay buffer
    memory = PrioritizedReplayBuffer(200000)
    optimizer = torch.optim.Adam(online_net.parameters(), lr=learning_rate)

    # Training metrics
    best_reward = float('-inf')
    episode_rewards = []
    total_steps = 0
    start_episode = 0
    recent_rewards = deque(maxlen=100)
    exploration_actions = deque(maxlen=1000)

    # Load checkpoint if exists
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint:
        online_net.load_state_dict(checkpoint['online_net_state_dict'])
        target_net.load_state_dict(checkpoint['target_net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode']
        total_steps = checkpoint['total_steps']
        best_reward = checkpoint.get('best_reward', float('-inf'))
        episode_rewards = checkpoint.get('episode_rewards', [])
        logging.info(f"Resuming from episode {start_episode}")

    # Enhanced early stopping with longer patience
    patience = 1500  # Increased from 1000
    episodes_without_improvement = 0
    min_reward_threshold = 100

    logging.info("Starting enhanced training loop")
    for episode in tqdm(range(start_episode, num_episodes), desc="Training"):
        state, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_random_actions = 0
        done = False
        truncated = False
        positions_this_episode = set()

        while not (done or truncated):
            # Enhanced exploration strategy
            epsilon = max(
                final_exploration,
                initial_exploration * (1 - total_steps / exploration_steps)
            )
            
            # Action selection with tracking
            if np.random.random() < epsilon:
                action = env.action_space.sample()
                episode_random_actions += 1
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = online_net(state_tensor).argmax().item()

            exploration_actions.append(1 if np.random.random() < epsilon else 0)

            # Environment step
            next_state, reward, done, truncated, info = env.step(action)

            # Enhanced reward calculation
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            
            # UPDATED: Enhanced position-based reward
            if 'x' in info and 'y' in info:
                position = (info['x'], info['y'])
                if position not in position_history:
                    position_history.add(position)
                    positions_this_episode.add(position)
                    position_streak += 1
                    combined_reward = 0.4  # Doubled position bonus
                    
                    # Bonus for finding multiple new positions
                    if len(positions_this_episode) % 3 == 0:  # Changed from 5 to 3
                        combined_reward += 0.5
                    
                    # Streak bonus
                    if position_streak >= 3:
                        combined_reward += 0.3 * position_streak
                else:
                    position_streak = 0

            # Intrinsic reward
            intrinsic_reward = curiosity.compute_intrinsic_reward(
                state_tensor,
                torch.tensor([action]).to(device),
                next_state_tensor
            )

            # Combine rewards
            combined_reward = reward_combiner.combine_rewards(
                np.array([reward]),
                intrinsic_reward.cpu().numpy()
            )[0]

            # UPDATED: More aggressive step bonus
            step_bonus = calculate_step_bonus(episode_steps)
            combined_reward += step_bonus

            # UPDATED: Enhanced momentum bonus
            if len(last_steps) > 0:
                avg_steps = sum(last_steps) / len(last_steps)
                if episode_steps > avg_steps:
                    momentum_bonus = 0.2 * (episode_steps - avg_steps) / avg_steps
                    combined_reward += momentum_bonus

            # Store transition and update counters
            memory.push(state, action, combined_reward, next_state, done or truncated)
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            # Enhanced training
            if len(memory) > batch_size:
                batch, indices, weights = memory.sample(batch_size)
                
                # Prepare batch
                state_batch = torch.FloatTensor(np.array([t.state for t in batch])).to(device)
                action_batch = torch.LongTensor([t.action for t in batch]).to(device)
                reward_batch = torch.FloatTensor([t.reward for t in batch]).to(device)
                next_state_batch = torch.FloatTensor(np.array([t.next_state for t in batch])).to(device)
                done_batch = torch.FloatTensor([t.done for t in batch]).to(device)

                # Compute Q values with double Q-learning
                current_q = online_net(state_batch).gather(1, action_batch.unsqueeze(1))
                with torch.no_grad():
                    next_actions = online_net(next_state_batch).max(1)[1]
                    next_q = target_net(next_state_batch).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    expected_q = reward_batch + gamma * next_q * (1 - done_batch)

                # Compute loss with prioritized replay
                loss = (torch.tensor(weights, device=device) * 
                       (current_q.squeeze() - expected_q.detach()) ** 2).mean()

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
                optimizer.step()

                # Update priorities
                priorities = (current_q.squeeze() - expected_q).abs().detach().cpu().numpy()
                memory.update_priorities(indices, priorities)

            # More frequent target updates
            if total_steps % target_update_frequency == 0:
                target_net.load_state_dict(online_net.state_dict())
                logging.info("Updated target network")

        # Episode completion
        last_steps.append(episode_steps)
        recent_rewards.append(episode_reward)
        avg_reward = np.mean(list(recent_rewards))
        exploration_ratio = np.mean(list(exploration_actions))

        # Enhanced logging
        logging.info(f'Episode {episode} - Reward: {episode_reward:.2f}, '
                    f'Steps: {episode_steps}, Epsilon: {epsilon:.3f}, '
                    f'Avg Reward (100): {avg_reward:.2f}, '
                    f'Exploration Ratio: {exploration_ratio:.2f}, '
                    f'New Positions: {len(positions_this_episode)}')

        if episode_reward > 0:
            logging.info(f"Non-zero reward achieved: {episode_reward}")

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            episodes_without_improvement = 0
            torch.save({
                'episode': episode,
                'online_net_state_dict': online_net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_reward': best_reward,
                'episode_rewards': episode_rewards,
                'total_steps': total_steps
            }, 'checkpoints/model_best.pt')
            logging.info(f"New best reward: {best_reward:.2f}")
        else:
            episodes_without_improvement += 1

        # Regular checkpointing with backup
        if episode % 100 == 0:
            backup_checkpoint(checkpoint_path)  # Backup before saving new checkpoint
            torch.save({
                'episode': episode,
                'online_net_state_dict': online_net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode_rewards': episode_rewards,
                'total_steps': total_steps,
                'best_reward': best_reward
            }, checkpoint_path)
            logging.info(f"Saved checkpoint at episode {episode}")

        # UPDATED: More aggressive exploration reset
        if exploration_ratio < min_exploration_ratio and episode > 500:
            logging.info("Restarting exploration due to low exploration ratio")
            epsilon = initial_exploration
            position_streak = 0  # Reset streak on exploration reset

        # Enhanced early stopping
        if episodes_without_improvement >= patience:
            logging.info("Stopping early due to no improvement")
            break

        if avg_reward >= min_reward_threshold:
            logging.info("Stopping early due to reaching reward threshold")
            break

    env.close()
    logging.info("Training completed")

if __name__ == '__main__':
    train()
