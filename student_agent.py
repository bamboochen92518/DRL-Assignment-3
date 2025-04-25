import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import DuelingCNN and NoisyLinear from train.py
from train import DuelingCNN, NoisyLinear

# Agent class with checkpoint loading
class Agent(object):
    """Agent that acts using a loaded DQN model."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        
        # Initialize the model and device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DuelingCNN(in_c=4, n_actions=12, noisy_sigma_init=2.5).to(self.device)  # in_c=4 for 4-frame stacking
        self.model.eval()  # Set to evaluation mode for inference
        
        # Load the checkpoint
        checkpoint_path = 'checkpoints/rainbow_icm.pth'  # Hardcoded path (can be modified to be configurable)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            print(f"Successfully loaded model from {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            print("Using randomly initialized model instead.")

    def act(self, observation):
        # Convert observation to tensor and move to device
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Compute Q-values and select action
        with torch.no_grad():
            q_values = self.model(obs_tensor)
            action = q_values.argmax(dim=1).item()
        
        # Clean up to save memory
        del obs_tensor, q_values
        
        return action