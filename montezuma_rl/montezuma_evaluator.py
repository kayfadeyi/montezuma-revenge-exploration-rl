import json
import os
from collections import defaultdict
from datetime import datetime

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class StateHasher:
    def __init__(self, resize_shape=(42, 42)):
        self.resize_shape = resize_shape

    def hash_state(self, state):
        if len(state.shape) == 3:
            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, self.resize_shape)
        state = (state > 128).astype(np.uint8)
        return hash(state.tobytes())


class MontezumaEvaluator:
    def __init__(self, agent, environment, n_episodes=100, max_steps=4500):
        self.env = gym.make(environment, render_mode='rgb_array')
        self.agent = agent
        self.n_episodes = n_episodes
        self.max_steps = max_steps

        self.episode_rewards = []
        self.episode_lengths = []
        self.unique_states = set()
        self.room_visits = defaultdict(int)
        self.death_locations = []

        self.state_hasher = StateHasher()

    def evaluate(self):
        print(f'\nStarting evaluation over {self.n_episodes} episodes...')

        for episode in tqdm(range(self.n_episodes)):
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0

            while episode_steps < self.max_steps:
                action = self.agent.predict_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)

                episode_reward += reward
                episode_steps += 1

                state_hash = self.state_hasher.hash_state(state)
                self.unique_states.add(state_hash)

                if 'room' in info:
                    self.room_visits[info['room']] += 1

                if done and reward < 0:
                    self.death_locations.append((info.get('x', 0), info.get('y', 0)))

                if done:
                    break

                state = next_state

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)

    def get_metrics(self):
        return {
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'mean_episode_length': np.mean(self.episode_lengths),
            'unique_states_visited': len(self.unique_states),
            'unique_rooms_visited': len(self.room_visits),
            'total_rooms_visited': sum(self.room_visits.values()),
            'n_episodes': self.n_episodes
        }

    def save_results(self, save_dir='eval_results'):
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        metrics = self.get_metrics()
        with open(f'{save_dir}/metrics_{timestamp}.json', 'w') as f:
            json.dump(metrics, f, indent=4)

        self._plot_rewards(save_dir, timestamp)
        self._plot_state_visitation(save_dir, timestamp)
        self._plot_room_heatmap(save_dir, timestamp)

        print(f'\nResults saved to {save_dir}')

    def _plot_rewards(self, save_dir, timestamp):
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig(f'{save_dir}/rewards_{timestamp}.png')
        plt.close()

    def _plot_state_visitation(self, save_dir, timestamp):
        states = [len(self.unique_states[:i + 1]) for i in range(len(self.unique_states))]
        plt.figure(figsize=(10, 6))
        plt.plot(states)
        plt.title('Cumulative Unique States Visited')
        plt.xlabel('Episode')
        plt.ylabel('Number of Unique States')
        plt.savefig(f'{save_dir}/unique_states_{timestamp}.png')
        plt.close()

    def _plot_room_heatmap(self, save_dir, timestamp):
        rooms = sorted(self.room_visits.keys())
        visits = [self.room_visits[room] for room in rooms]
        plt.figure(figsize=(12, 6))
        plt.bar(rooms, visits)
        plt.title('Room Visitation Frequency')
        plt.xlabel('Room Number')
        plt.ylabel('Number of Visits')
        plt.savefig(f'{save_dir}/room_heatmap_{timestamp}.png')
        plt.close()
