import unittest
import torch
from src.models.dqn import DuelingDQN

class TestDuelingDQN(unittest.TestCase):
    def setUp(self):
        self.model = DuelingDQN(input_channels=4, num_actions=18)
        self.batch_size = 32
        self.input_shape = (4, 84, 84)

    def test_model_output_shape(self):
        x = torch.randn(self.batch_size, *self.input_shape)
        output = self.model(x)
        self.assertEqual(output.shape, (self.batch_size, 18))

    def test_value_advantage_separation(self):
        # Test that value and advantage streams are properly separated
        x = torch.randn(1, *self.input_shape)
        value = self.model.value_stream(self.model.conv1(x).view(1, -1))
        advantage = self.model.advantage_stream(self.model.conv1(x).view(1, -1))
        
        self.assertEqual(value.shape, (1, 1))
        self.assertEqual(advantage.shape, (1, 18))

    def test_model_device_transfer(self):
        if torch.cuda.is_available():
            model = self.model.cuda()
            x = torch.randn(1, *self.input_shape).cuda()
            output = model(x)
            self.assertTrue(output.is_cuda)

if __name__ == '__main__':
    unittest.main()