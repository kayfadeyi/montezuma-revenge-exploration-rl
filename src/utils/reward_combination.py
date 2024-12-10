import numpy as np
import torch
from collections import deque

class RewardNormalizer:
    def __init__(self, clip_range=(-10, 10), epsilon=1e-8):
        self.mean = 0
        self.std = 1
        self.count = 0
        self.clip_range = clip_range
        self.epsilon = epsilon
        self.running_ms = RunningMeanStd()
    
    def __call__(self, x):
        self.running_ms.update(x)
        normalized = (x - self.running_ms.mean) / (np.sqrt(self.running_ms.var) + self.epsilon)
        return np.clip(normalized, self.clip_range[0], self.clip_range[1])

class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class RewardCombiner:
    def __init__(self, beta=0.2, reward_window=1000):
        self.beta = beta
        self.ext_normalizer = RewardNormalizer()
        self.int_normalizer = RewardNormalizer()
        
        self.ext_rewards = deque(maxlen=reward_window)
        self.int_rewards = deque(maxlen=reward_window)
        self.combined_rewards = deque(maxlen=reward_window)
    
    def combine_rewards(self, extrinsic_rewards, intrinsic_rewards):
        if torch.is_tensor(extrinsic_rewards):
            extrinsic_rewards = extrinsic_rewards.cpu().numpy()
        if torch.is_tensor(intrinsic_rewards):
            intrinsic_rewards = intrinsic_rewards.cpu().numpy()
        
        normalized_ext = self.ext_normalizer(extrinsic_rewards)
        normalized_int = self.int_normalizer(intrinsic_rewards)
        
        combined = normalized_ext + self.beta * normalized_int
        
        self.ext_rewards.extend(extrinsic_rewards)
        self.int_rewards.extend(intrinsic_rewards)
        self.combined_rewards.extend(combined)
        
        return combined
    
    def adjust_beta(self, new_beta):
        self.beta = new_beta
    
    def get_stats(self):
        return {
            'extrinsic_mean': np.mean(self.ext_rewards) if self.ext_rewards else 0,
            'extrinsic_std': np.std(self.ext_rewards) if self.ext_rewards else 0,
            'intrinsic_mean': np.mean(self.int_rewards) if self.int_rewards else 0,
            'intrinsic_std': np.std(self.int_rewards) if self.int_rewards else 0,
            'combined_mean': np.mean(self.combined_rewards) if self.combined_rewards else 0,
            'combined_std': np.std(self.combined_rewards) if self.combined_rewards else 0,
            'beta': self.beta
        }

def create_reward_schedule(initial_beta=0.5, final_beta=0.1, decay_steps=1000000):
    def beta_schedule(step):
        progress = min(1.0, step / decay_steps)
        return final_beta + (initial_beta - final_beta) * (1 - progress)
    return beta_schedule