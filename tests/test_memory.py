import unittest
import numpy as np
from src.memory.prioritized_replay import PrioritizedReplayBuffer

class TestPrioritizedReplay(unittest.TestCase):
    def setUp(self):
        self.capacity = 100
        self.buffer = PrioritizedReplayBuffer(self.capacity)

    def test_push(self):
        state = np.zeros((4, 84, 84))
        next_state = np.ones((4, 84, 84))
        self.buffer.push(state, 0, 1.0, next_state, False)
        self.assertEqual(len(self.buffer), 1)

    def test_sample(self):
        # Fill buffer with some transitions
        for i in range(10):
            state = np.zeros((4, 84, 84))
            next_state = np.ones((4, 84, 84))
            self.buffer.push(state, i % 4, float(i), next_state, False)

        batch, indices, weights = self.buffer.sample(5)
        self.assertEqual(len(batch), 5)
        self.assertEqual(len(indices), 5)
        self.assertEqual(len(weights), 5)

    def test_priority_update(self):
        state = np.zeros((4, 84, 84))
        next_state = np.ones((4, 84, 84))
        self.buffer.push(state, 0, 1.0, next_state, False)
        _, indices, _ = self.buffer.sample(1)
        self.buffer.update_priorities(indices, [2.0])

if __name__ == '__main__':
    unittest.main()