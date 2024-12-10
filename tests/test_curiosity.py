import unittest
import torch
from src.exploration.curiosity import CuriosityModule

class TestCuriosity(unittest.TestCase):
    def setUp(self):
        self.state_shape = (4, 84, 84)
        self.action_dim = 18
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.curiosity = CuriosityModule(self.state_shape, self.action_dim, self.device)

    def test_intrinsic_reward_shape(self):
        batch_size = 32
        state = torch.randn(batch_size, *self.state_shape).to(self.device)
        action = torch.randint(0, self.action_dim, (batch_size,)).to(self.device)
        next_state = torch.randn(batch_size, *self.state_shape).to(self.device)

        reward = self.curiosity.compute_intrinsic_reward(state, action, next_state)
        self.assertEqual(reward.shape, (batch_size,))

    def test_feature_encoder(self):
        batch_size = 1
        state = torch.randn(batch_size, *self.state_shape).to(self.device)
        features = self.curiosity.encoder(state)
        self.assertEqual(features.shape, (batch_size, self.curiosity.feature_dim))

    def test_forward_dynamics(self):
        batch_size = 1
        state_features = torch.randn(batch_size, self.curiosity.feature_dim).to(self.device)
        action = torch.randint(0, self.action_dim, (batch_size,)).to(self.device)
        next_features = self.curiosity.forward_dynamics(state_features, action)
        self.assertEqual(next_features.shape, (batch_size, self.curiosity.feature_dim))

if __name__ == '__main__':
    unittest.main()