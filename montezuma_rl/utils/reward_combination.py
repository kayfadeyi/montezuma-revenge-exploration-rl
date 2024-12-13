import numpy as np
import torch
from collections import deque

class RewardNormalizer:
    """Normalizes rewards using running mean and standard deviation"""
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
    """Tracks running mean and standard deviation"""
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
    """Combines intrinsic and extrinsic rewards with enhanced scaling"""
    def __init__(self, beta=0.8, reward_window=1000):  # Increased beta from 0.5 to 0.8
        self.beta = beta
        self.ext_normalizer = RewardNormalizer()
        self.int_normalizer = RewardNormalizer()
        
        # Keep track of raw rewards for monitoring
        self.ext_rewards = deque(maxlen=reward_window)
        self.int_rewards = deque(maxlen=reward_window)
        self.combined_rewards = deque(maxlen=reward_window)
    
    def combine_rewards(self, extrinsic_rewards, intrinsic_rewards):
        """
        Combine extrinsic and intrinsic rewards with enhanced scaling
        
        Args:
            extrinsic_rewards: numpy array or torch tensor of extrinsic rewards
            intrinsic_rewards: numpy array or torch tensor of intrinsic rewards
        
        Returns:
            Combined normalized rewards
        """
        # Convert to numpy if needed
        if torch.is_tensor(extrinsic_rewards):
            extrinsic_rewards = extrinsic_rewards.cpu().numpy()
        if torch.is_tensor(intrinsic_rewards):
            intrinsic_rewards = intrinsic_rewards.cpu().numpy()
        
        # Normalize rewards
        normalized_ext = self.ext_normalizer(extrinsic_rewards)
        normalized_int = self.int_normalizer(intrinsic_rewards)
        
        # Enhanced combination with higher beta
        combined = normalized_ext + self.beta * normalized_int
        
        # Store raw rewards for monitoring
        self.ext_rewards.extend(extrinsic_rewards)
        self.int_rewards.extend(intrinsic_rewards)
        self.combined_rewards.extend(combined)
        
        return combined
    
    def adjust_beta(self, new_beta):
        """Adjust the influence of intrinsic rewards"""
        self.beta = new_beta
    
    def get_stats(self):
        """Return statistics about recent rewards"""
        stats = {
            'extrinsic_mean': np.mean(self.ext_rewards) if self.ext_rewards else 0,
            'extrinsic_std': np.std(self.ext_rewards) if self.ext_rewards else 0,
            'intrinsic_mean': np.mean(self.int_rewards) if self.int_rewards else 0,
            'intrinsic_std': np.std(self.int_rewards) if self.int_rewards else 0,
            'combined_mean': np.mean(self.combined_rewards) if self.combined_rewards else 0,
            'combined_std': np.std(self.combined_rewards) if self.combined_rewards else 0,
            'beta': self.beta
        }
        return stats

def create_reward_schedule(initial_beta=0.8, final_beta=0.2, decay_steps=1000000):
    """
    Create a schedule for decaying beta over time
    
    Args:
        initial_beta: Starting value for beta (increased from 0.5)
        final_beta: Final value for beta
        decay_steps: Number of steps over which to decay beta
    
    Returns:
        Function that returns beta value for a given step
    """
    def beta_schedule(step):
        progress = min(1.0, step / decay_steps)
        return final_beta + (initial_beta - final_beta) * (1 - progress)
    return beta_schedule
