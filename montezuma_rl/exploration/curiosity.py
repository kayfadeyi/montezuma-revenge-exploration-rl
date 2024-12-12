import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FeatureEncoder(nn.Module):
    """Encodes states into latent feature representations"""
    def __init__(self, input_channels=4, feature_dim=512):
        super(FeatureEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate feature size after convolutions
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        self.fc = nn.Linear(linear_input_size, feature_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        return features

class ForwardDynamics(nn.Module):
    """Predicts next state features given current state features and action"""
    def __init__(self, feature_dim=512, action_dim=18, hidden_dim=512):  # Changed action_dim to 18
        super(ForwardDynamics, self).__init__()
        
        self.fc1 = nn.Linear(feature_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, feature_dim)
        self.action_dim = action_dim  # Store action_dim
        
    def forward(self, state_features, action):
        # Convert action to one-hot if discrete
        if action.dim() == 1:
            action = F.one_hot(action.long(), num_classes=self.action_dim).float()
        
        x = torch.cat([state_features, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        next_features = self.fc3(x)
        return next_features

class CuriosityModule:
    """Implements curiosity-driven exploration"""
    def __init__(self, state_shape, action_dim, device="cuda"):
        self.device = device
        self.feature_dim = 512
        self.action_dim = action_dim
        
        # Initialize networks
        self.encoder = FeatureEncoder(input_channels=state_shape[0], 
                                    feature_dim=self.feature_dim).to(device)
        self.forward_dynamics = ForwardDynamics(feature_dim=self.feature_dim,
                                              action_dim=action_dim).to(device)
        
        # Optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        self.forward_opt = torch.optim.Adam(self.forward_dynamics.parameters(), lr=1e-4)
        
        # Normalize intrinsic rewards
        self.reward_normalizer = RunningNormalizer()
        
    def compute_intrinsic_reward(self, state, action, next_state):
        """Compute intrinsic reward based on prediction error"""
        with torch.no_grad():
            # Extract features
            state_features = self.encoder(state)
            next_state_features = self.encoder(next_state)
            
            # Predict next state features
            predicted_features = self.forward_dynamics(state_features, action)
            
            # Compute prediction error (intrinsic reward)
            prediction_error = F.mse_loss(predicted_features, next_state_features, 
                                        reduction='none').mean(dim=1)
            
            # Normalize reward
            normalized_reward = self.reward_normalizer(prediction_error.cpu().numpy())
            
        return torch.FloatTensor(normalized_reward).to(self.device)

class RunningNormalizer:
    """Normalizes rewards using running statistics"""
    def __init__(self, epsilon=1e-8):
        self.mean = 0
        self.std = 1
        self.count = 0
        self.epsilon = epsilon
        
    def __call__(self, x):
        self.count += 1
        if self.count == 1:
            self.mean = np.mean(x)
            self.std = np.std(x)
        else:
            old_mean = self.mean
            self.mean = old_mean + (np.mean(x) - old_mean) / self.count
            self.std = np.sqrt(self.std**2 + ((x - old_mean) * (x - self.mean)).mean())
            
        return (x - self.mean) / (self.std + self.epsilon)
