import numpy as np
import cv2
from collections import deque
import torch

class FrameStacker:
    def __init__(self, frame_size=(84, 84), stack_size=4, device="cpu"):
        self.frame_size = frame_size
        self.stack_size = stack_size
        self.device = device
        self.frames = deque(maxlen=stack_size)
        
        for _ in range(stack_size):
            self.frames.append(
                np.zeros(frame_size, dtype=np.uint8)
            )
    
    def preprocess_frame(self, frame):
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, self.frame_size, 
                          interpolation=cv2.INTER_AREA)
        return frame
    
    def reset(self):
        for _ in range(self.stack_size):
            self.frames.append(
                np.zeros(self.frame_size, dtype=np.uint8)
            )
        return self.get_state()
    
    def add_frame(self, frame):
        processed = self.preprocess_frame(frame)
        self.frames.append(processed)
        return self.get_state()
    
    def get_state(self):
        state = np.stack(self.frames, axis=0)
        return torch.FloatTensor(state).to(self.device) / 255.0