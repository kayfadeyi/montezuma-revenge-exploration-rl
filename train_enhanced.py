import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import random
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from dqn import DQN
from replay_memory import PrioritizedReplayBuffer
from intrinsic_motivation import IntrinsicCuriosityModule

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configurations
ENV_NAME = 'ALE/MontezumaRevenge-v5'
NUM_EPISODES = 50000
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = 200000
TARGET_UPDATE_FREQUENCY = 5000
RESULTS_DIR = "./results"

# Create environment
env = gym.make(ENV_NAME, render_mode='rgb_array', frameskip=1)
num_actions = env.action_space.n

# Neural Networks
policy_net = DQN(env.observation_space.shape, num_actions).to(device)
target_net = DQN(env.observation_space.shape, num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Optimizer and replay buffer
optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
replay_buffer = PrioritizedReplayBuffer(capacity=100000)
curiosity_module = IntrinsicCuriosityModule(env.observation_space.shape, num_actions).to(device)

# Epsilon-greedy schedule
def epsilon_by_step(step):
    return EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-step / EPSILON_DECAY_STEPS)

# Tracking data for analysis
rewards = []
extrinsic_rewards = []
intrinsic_rewards = []
frames_per_reward = []
state_visits = defaultdict(int)
cumulative_reward = 0
steps = 0

# Training loop
for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    state = torch.FloatTensor(np.array(state, dtype=np.float32) / 255.0).to(device)
    episode_reward = 0
    episode_intrinsic_reward = 0
    episode_extrinsic_reward = 0
    episode_frames = 0

    done = False
    while not done:
        epsilon = epsilon_by_step(steps)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(state.unsqueeze(0)).argmax().item()

        next_state, reward, done, _, _ = env.step(action)
        next_state = torch.FloatTensor(np.array(next_state, dtype=np.float32) / 255.0).to(device)

        # Track state visits for heatmap
        state_tuple = tuple(next_state.cpu().numpy().flatten())
        state_visits[state_tuple] += 1

        intrinsic_reward = curiosity_module.compute_intrinsic_reward(state, next_state, action)
        total_reward = reward + intrinsic_reward

        replay_buffer.push(state.cpu().numpy(), action, total_reward, next_state.cpu().numpy(), done)
        episode_reward += total_reward
        episode_extrinsic_reward += reward
        episode_intrinsic_reward += intrinsic_reward

        if len(replay_buffer) > BATCH_SIZE:
            batch = replay_buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = [torch.FloatTensor(x).to(device) for x in batch]

            # Compute Q-learning target
            with torch.no_grad():
                next_q_values = target_net(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * GAMMA * next_q_values

            # Compute Q-value predictions
            q_values = policy_net(states)
            actions = actions.long().unsqueeze(1)
            predicted_q_values = q_values.gather(1, actions).squeeze(1)

            # Compute loss
            loss = nn.MSELoss()(predicted_q_values, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state
        steps += 1
        episode_frames += 1

        if steps % TARGET_UPDATE_FREQUENCY == 0:
            target_net.load_state_dict(policy_net.state_dict())

    rewards.append(episode_reward)
    extrinsic_rewards.append(episode_extrinsic_reward)
    intrinsic_rewards.append(episode_intrinsic_reward)
    frames_per_reward.append(episode_frames / (episode_extrinsic_reward + 1e-6))  # Avoid division by zero
    cumulative_reward += episode_extrinsic_reward

    print(f"Episode {episode}, Total Reward: {episode_reward:.2f}, Extrinsic Reward: {episode_extrinsic_reward:.2f}, Intrinsic Reward: {episode_intrinsic_reward:.2f}")

# Save results
os.makedirs(RESULTS_DIR, exist_ok=True)
np.save(os.path.join(RESULTS_DIR, "rewards.npy"), rewards)
np.save(os.path.join(RESULTS_DIR, "extrinsic_rewards.npy"), extrinsic_rewards)
np.save(os.path.join(RESULTS_DIR, "intrinsic_rewards.npy"), intrinsic_rewards)
np.save(os.path.join(RESULTS_DIR, "frames_per_reward.npy"), frames_per_reward)

# Generate graphs
plt.figure(figsize=(10, 6))
plt.plot(rewards, label="Total Rewards")
plt.plot(extrinsic_rewards, label="Extrinsic Rewards")
plt.plot(intrinsic_rewards, label="Intrinsic Rewards")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Training Rewards")
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "rewards_plot.png"))
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(frames_per_reward, label="Frames Per Reward")
plt.xlabel("Episodes")
plt.ylabel("Frames Per Reward")
plt.title("Frames Per Reward Over Training")
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "frames_per_reward_plot.png"))
plt.show()

# Heatmap generation
heatmap_data = np.zeros((84, 84))  # Assuming grid size of the game
for state, visits in state_visits.items():
    x, y = np.clip(np.array(state[:2]) * 84, 0, 83).astype(int)  # Scale and clip coordinates
    heatmap_data[x, y] += visits

plt.figure(figsize=(10, 6))
plt.imshow(heatmap_data, cmap="hot", interpolation="nearest")
plt.colorbar(label="State Visit Frequency")
plt.title("State Coverage Heatmap")
plt.savefig(os.path.join(RESULTS_DIR, "state_coverage_heatmap.png"))
plt.show()

