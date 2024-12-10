import gym
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from models.dqn import DuelingDQN
from memory.prioritized_replay import PrioritizedReplayBuffer
from utils.preprocessing import FrameStacker
from exploration.curiosity import CuriosityModule
from utils.reward_combination import RewardCombiner

def train(env_name='MontezumaRevenge-v0', 
         num_episodes=10000,
         batch_size=32,
         gamma=0.99,
         initial_exploration=1.0,
         final_exploration=0.01,
         exploration_steps=1000000):

    # Create directories
    Path('checkpoints').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)

    # Initialize environment and preprocessing
    env = gym.make(env_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frame_stacker = FrameStacker(device=device)

    # Initialize networks
    online_net = DuelingDQN().to(device)
    target_net = DuelingDQN().to(device)
    target_net.load_state_dict(online_net.state_dict())

    # Initialize curiosity module
    curiosity = CuriosityModule(state_shape=(4, 84, 84), action_dim=env.action_space.n, device=device)
    reward_combiner = RewardCombiner(beta=0.2)

    # Initialize memory and optimizers
    memory = PrioritizedReplayBuffer(100000)
    optimizer = torch.optim.Adam(online_net.parameters())

    # Training loop
    total_steps = 0
    best_reward = float('-inf')
    
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        state = frame_stacker.reset()
        episode_reward = 0
        done = False

        while not done:
            # Select action with epsilon-greedy
            epsilon = final_exploration + (initial_exploration - final_exploration) * \
                     max(0, (exploration_steps - total_steps)) / exploration_steps
            
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = online_net(state.unsqueeze(0)).argmax().item()

            # Take step in environment
            next_state, reward, done, info = env.step(action)
            next_state = frame_stacker.add_frame(next_state)

            # Compute intrinsic reward
            intrinsic_reward = curiosity.compute_intrinsic_reward(
                state.unsqueeze(0),
                torch.tensor([action]).to(device),
                next_state.unsqueeze(0)
            )

            # Combine rewards
            combined_reward = reward_combiner.combine_rewards(
                np.array([reward]),
                intrinsic_reward.cpu().numpy()
            )[0]

            # Store transition
            memory.push(state, action, combined_reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1

            # Train if enough samples
            if len(memory) > batch_size:
                batch, indices, weights = memory.sample(batch_size)
                state_batch = torch.stack([t.state for t in batch])
                action_batch = torch.tensor([t.action for t in batch], device=device)
                reward_batch = torch.tensor([t.reward for t in batch], device=device, dtype=torch.float32)
                next_state_batch = torch.stack([t.next_state for t in batch])
                done_batch = torch.tensor([t.done for t in batch], device=device, dtype=torch.float32)

                # Compute Q values
                current_q = online_net(state_batch).gather(1, action_batch.unsqueeze(1))
                next_actions = online_net(next_state_batch).max(1)[1].detach()
                next_q = target_net(next_state_batch).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                expected_q = reward_batch + gamma * next_q * (1 - done_batch)

                # Compute loss and update priorities
                loss = (torch.tensor(weights, device=device) * \
                       (current_q.squeeze() - expected_q) ** 2).mean()

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update priorities in replay buffer
                priorities = (current_q.squeeze() - expected_q).abs().detach().cpu().numpy()
                memory.update_priorities(indices, priorities)

            # Update target network periodically
            if total_steps % 10000 == 0:
                target_net.load_state_dict(online_net.state_dict())

        # Save model if best reward achieved
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save({
                'online_net': online_net.state_dict(),
                'target_net': target_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'episode': episode,
                'total_steps': total_steps,
                'best_reward': best_reward
            }, 'checkpoints/model_best.pt')

        # Save periodic checkpoint
        if episode % 100 == 0:
            torch.save({
                'online_net': online_net.state_dict(),
                'target_net': target_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'episode': episode,
                'total_steps': total_steps
            }, f'checkpoints/model_episode_{episode}.pt')

        print(f'Episode {episode}, Reward: {episode_reward}, Steps: {total_steps}, Epsilon: {epsilon:.3f}')

if __name__ == '__main__':
    train()