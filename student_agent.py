import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from torchvision import transforms as T

# Import DuelingCNN and NoisyLinear from train.py
from train import DuelingCNN, NoisyLinear

# Agent class with checkpoint loading and preprocessing
class Agent(object):
    """Agent that acts using a loaded DQN model."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        
        # Initialize the model and device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DuelingCNN(in_c=4, n_actions=12, noisy_sigma_init=2.5).to(self.device)  # in_c=4 for 4-frame stacking
        self.model.eval()  # Set to evaluation mode for inference
        
        # Load the checkpoint
        checkpoint_path = 'checkpoints/rainbow_icm.pth'  # Hardcoded path
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            print(f"Successfully loaded model from {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            print("Using randomly initialized model instead.")

        # Preprocessing transform (grayscale and resize)
        self.transform = T.Compose([
            T.ToPILImage(),  # Convert numpy array to PIL Image
            T.Grayscale(),   # Convert to grayscale
            T.Resize((84, 90)),  # Resize to 84x90
            T.ToTensor()     # Convert to tensor (1, 84, 90)
        ])

        # Frame stack to maintain 4 frames
        self.frame_stack = deque(maxlen=4)
        # Initialize the frame stack with zeros
        dummy_frame = np.zeros((1, 84, 90), dtype=np.float32)
        for _ in range(4):
            self.frame_stack.append(dummy_frame)

    def act(self, observation):
        # Ensure the observation is in the correct format
        observation = np.ascontiguousarray(observation)  # Fix negative strides
        if observation.shape != (240, 256, 3):  # Expected raw shape
            raise ValueError(f"Expected observation shape (240, 256, 3), but got {observation.shape}")

        # Preprocess the observation (grayscale, resize)
        processed_frame = self.transform(observation)  # Shape: (1, 84, 90)
        processed_frame = processed_frame.numpy()  # Convert back to numpy for frame stack

        # Add to frame stack
        self.frame_stack.append(processed_frame)

        # Stack the frames to get shape (4, 84, 90)
        stacked_frames = np.concatenate(self.frame_stack, axis=0)  # Shape: (4, 84, 90)

        # Convert to tensor and add batch dimension
        obs_tensor = torch.tensor(stacked_frames, dtype=torch.float32).unsqueeze(0).to(self.device)  # Shape: (1, 4, 84, 90)

        # Compute Q-values and select action
        with torch.no_grad():
            q_values = self.model(obs_tensor)
            action = q_values.argmax(dim=1).item()

        # Clean up to save memory
        del obs_tensor, q_values, processed_frame, stacked_frames

        return action