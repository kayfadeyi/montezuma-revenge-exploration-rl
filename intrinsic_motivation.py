import torch
import torch.nn as nn

class IntrinsicCuriosityModule(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(IntrinsicCuriosityModule, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512)
        )
        self.forward_model = nn.Linear(512 + num_actions, 512)
        self.inverse_model = nn.Linear(512 * 2, num_actions)

    def compute_intrinsic_reward(self, state, next_state, action):
        state_features = self.feature_extractor(state)
        next_state_features = self.feature_extractor(next_state)
        predicted_next_features = self.forward_model(torch.cat([state_features, action], dim=1))
        intrinsic_reward = (next_state_features - predicted_next_features).pow(2).mean()
        return intrinsic_reward.item()

