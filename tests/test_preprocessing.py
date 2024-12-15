import unittest
import numpy as np
import torch
from utils.preprocessing import FrameStacker


class TestFrameStacker(unittest.TestCase):
    def setUp(self):
        self.frame_stacker = FrameStacker(frame_size=(84, 84), stack_size=4)
        self.test_frame = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)

    def test_preprocess_frame(self):
        processed = self.frame_stacker.preprocess_frame(self.test_frame)
        self.assertEqual(processed.shape, (84, 84))
        self.assertEqual(processed.dtype, np.uint8)

    def test_stack_shape(self):
        state = self.frame_stacker.add_frame(self.test_frame)
        self.assertEqual(state.shape, (4, 84, 84))

    def test_normalization(self):
        state = self.frame_stacker.add_frame(self.test_frame)
        self.assertTrue(torch.all(state >= 0) and torch.all(state <= 1))

    def test_reset(self):
        state = self.frame_stacker.reset()
        self.assertEqual(state.shape, (4, 84, 84))
        self.assertTrue(torch.all(state == 0))


if __name__ == '__main__':
    unittest.main()
