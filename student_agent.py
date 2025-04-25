import gym
import torch
import numpy as np
from collections import deque
from torchvision import transforms as T
import argparse

# Import Agent, DuelingCNN, and NoisyLinear from train.py
from train import Agent, DuelingCNN, NoisyLinear

# Agent class that reuses the act method from train.py's Agent
class Agent(object):
    """Agent that acts using a loaded DQN model from train.py."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)

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

        # Define observation shape expected by train.py's Agent
        obs_shape = (4, 84, 90)  # 4-frame stack, 84x90
        n_actions = 12  # Matches self.action_space
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create arguments for train.py's Agent (matching train.py's defaults)
        args = argparse.Namespace(
            noisy_sigma_init=2.5,
            lr=0.00025,
            adam_eps=0.00015,
            icm_embed_dim=256,
            icm_lr=1e-4,
            icm_beta=0.2,
            icm_eta=0.01,
            buffer_capacity=10000,
            per_alpha=0.6,
            per_beta=0.4,
            per_beta_frames=2000000,
            n_step=5,
            gamma=0.9,
            batch_size=32,
            copy_network_freq=10000,
            skip_frames=4,
            max_episode_steps=3000,
            backward_penalty=0,
            stay_penalty=0,
            death_penalty=-100,
            num_episodes=100000,
            max_frames=44800000,
            eps_start=1.0,
            eps_end=0.01,
            eps_decay=0.9999,
            per_epsilon=0.1
        )

        # Instantiate train.py's Agent
        self.train_agent = Agent(obs_shape, n_actions, device, args)

        # Load the checkpoint into train.py's Agent
        checkpoint_path = 'checkpoints/rainbow_icm.pth'
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.train_agent.online.load_state_dict(checkpoint['model'])
            self.train_agent.target.load_state_dict(checkpoint['model'])
            self.train_agent.frame_idx = checkpoint.get('frame_idx', 0)
            print(f"Successfully loaded model from {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            print("Using randomly initialized model instead.")

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

        # Call the act method from train.py's Agent
        action = self.train_agent.act(stacked_frames)

        # Clean up to save memory
        del processed_frame, stacked_frames

        return action