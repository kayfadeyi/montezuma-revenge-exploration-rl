import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
from datetime import datetime
import os

from montezuma_rl.models.dqn import DuelingDQN
from montezuma_rl.memory.prioritized_replay import PrioritizedReplayBuffer
from montezuma_rl.exploration.curiosity import CuriosityModule
from montezuma_rl.utils.reward_combination import RewardCombiner

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

def train(num_episodes=2000,
         batch_size=32,
         gamma=0.99,
         initial_exploration=1.0,
         final_exploration=0.01,
         exploration_steps=500000,
         checkpoint_path='checkpoints/training_state.pt'):
   """
   Train the agent on Montezuma's Revenge.
   
   Args:
       num_episodes: Maximum number of episodes to train
       batch_size: Size of batches for training
       gamma: Discount factor for future rewards
       initial_exploration: Starting value of epsilon for exploration
       final_exploration: Final value of epsilon for exploration
       exploration_steps: Number of steps to decay epsilon over
       checkpoint_path: Path to save/load checkpoint
   """
   setup_logging()
   logging.info("Starting training with parameters:")
   logging.info(f"Episodes: {num_episodes}, Batch size: {batch_size}, Gamma: {gamma}")
   logging.info(f"Exploration: {initial_exploration} -> {final_exploration} over {exploration_steps} steps")

   # Create directories for saving results
   Path('checkpoints').mkdir(exist_ok=True)

   # Create environment with proper wrappers
   try:
       logging.info("Creating environment: MontezumaRevenge-v4")
       env = gym.make('MontezumaRevenge-v4',
                     frameskip=4,
                     render_mode=None)  
       
       # Add preprocessing wrappers
       env = AtariPreprocessing(env, 
                              grayscale_obs=True,
                              frame_skip=1,  # Already handled by env
                              scale_obs=True)
       env = FrameStack(env, num_stack=4)
       logging.info("Environment created successfully")
       
   except Exception as e:
       logging.error(f"Failed to create environment: {e}")
       raise

   # Initialize device
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   logging.info(f"Using device: {device}")

   # Initialize networks
   online_net = DuelingDQN().to(device)
   target_net = DuelingDQN().to(device)
   target_net.load_state_dict(online_net.state_dict())

   # Initialize curiosity module
   curiosity = CuriosityModule(
       state_shape=(4, 84, 84),
       action_dim=env.action_space.n,
       device=device
   )
   reward_combiner = RewardCombiner(beta=0.2)

   # Initialize memory and optimizer
   memory = PrioritizedReplayBuffer(100000)
   optimizer = torch.optim.Adam(online_net.parameters(), lr=0.00025)

   # Training metrics
   best_reward = float('-inf')
   episode_rewards = []
   total_steps = 0
   start_episode = 0

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

   # Training parameters
   patience = 1000  # Episodes to wait for improvement
   episodes_without_improvement = 0
   min_reward_threshold = 100  # Early success threshold

   # Training loop
   logging.info("Starting training loop")
   for episode in tqdm(range(start_episode, num_episodes), desc="Training"):
       state, info = env.reset()
       episode_reward = 0
       episode_steps = 0
       done = False
       truncated = False

       while not (done or truncated):
           # Calculate epsilon for exploration with faster decay
           epsilon = max(
               final_exploration,
               initial_exploration * (1 - total_steps / (exploration_steps/2))
           )
           
           # Select action
           if np.random.random() < epsilon:
               action = env.action_space.sample()
           else:
               with torch.no_grad():
                   state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                   action = online_net(state_tensor).argmax().item()

           # Take step in environment
           next_state, reward, done, truncated, info = env.step(action)

           # Compute intrinsic reward
           state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
           next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
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

           # Store transition
           memory.push(state, action, combined_reward, next_state, done or truncated)
           
           # Update counters
           state = next_state
           episode_reward += reward
           episode_steps += 1
           total_steps += 1

           # Train if enough samples
           if len(memory) > batch_size:
               batch, indices, weights = memory.sample(batch_size)
               
               # Prepare batch
               state_batch = torch.FloatTensor(np.array([t.state for t in batch])).to(device)
               action_batch = torch.LongTensor([t.action for t in batch]).to(device)
               reward_batch = torch.FloatTensor([t.reward for t in batch]).to(device)
               next_state_batch = torch.FloatTensor(np.array([t.next_state for t in batch])).to(device)
               done_batch = torch.FloatTensor([t.done for t in batch]).to(device)

               # Compute current Q values
               current_q = online_net(state_batch).gather(1, action_batch.unsqueeze(1))

               # Compute next Q values (Double DQN)
               with torch.no_grad():
                   next_actions = online_net(next_state_batch).max(1)[1]
                   next_q = target_net(next_state_batch).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                   expected_q = reward_batch + gamma * next_q * (1 - done_batch)

               # Compute loss and update priorities
               loss = (torch.tensor(weights, device=device) * 
                      (current_q.squeeze() - expected_q.detach()) ** 2).mean()

               # Optimize
               optimizer.zero_grad()
               loss.backward()
               torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
               optimizer.step()

               # Update priorities in replay buffer
               priorities = (current_q.squeeze() - expected_q).abs().detach().cpu().numpy()
               memory.update_priorities(indices, priorities)

           # Update target network periodically
           if total_steps % 10000 == 0:
               target_net.load_state_dict(online_net.state_dict())
               logging.info("Updated target network")

       # Episode completed
       episode_rewards.append(episode_reward)
       avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0

       # Early stopping checks
       if episode_reward > best_reward:
           best_reward = episode_reward
           episodes_without_improvement = 0
           # Save best model
           torch.save({
               'episode': episode,
               'online_net_state_dict': online_net.state_dict(),
               'target_net_state_dict': target_net.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
               'best_reward': best_reward,
               'episode_rewards': episode_rewards,
               'total_steps': total_steps
           }, 'checkpoints/model_best.pt')
           logging.info(f"New best reward: {best_reward:.2f} - Saved checkpoint")
       else:
           episodes_without_improvement += 1

       # Regular checkpoint saving
       if episode % 100 == 0:
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

       logging.info(f'Episode {episode} - Reward: {episode_reward:.2f}, '
                   f'Steps: {episode_steps}, Epsilon: {epsilon:.3f}, '
                   f'Avg Reward (100): {avg_reward:.2f}')

       # Check early stopping conditions
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
