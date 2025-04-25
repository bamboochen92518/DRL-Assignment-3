import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from torchvision import transforms as T
import gc

# Import DuelingCNN, NoisyLinear, and COMPLEX_MOVEMENT from train.py
from train import DuelingCNN, NoisyLinear, COMPLEX_MOVEMENT

# Define constants (adjust these to match train.py if needed)
CHECKPOINT_PATH = 'checkpoints/rainbow_icm.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SKIP_FRAMES = 4  # Matches train.py's args.skip_frames

# Agent class with checkpoint loading and preprocessing
class Agent(object):
    """Agent that acts using a loaded DQN model."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        
        # Initialize the model and device
        self.device = torch.device(DEVICE)
        self.model = DuelingCNN(in_c=4, n_actions=len(COMPLEX_MOVEMENT), noisy_sigma_init=2.5).to(self.device)
        self.model.eval()  # Set to evaluation mode for inference
        
        # Load the checkpoint
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint.get('model', checkpoint))
            print(f"Successfully loaded model from {CHECKPOINT_PATH}")
        except Exception as e:
            print(f"Failed to load checkpoint from {CHECKPOINT_PATH}: {e}")
            print("Using randomly initialized model instead.")

        # Preprocessing transform (grayscale and resize)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((84, 90)),
            T.ToTensor()
        ])

        # Frame stack to maintain 4 frames
        self.frame_stack = deque(maxlen=4)
        self.first = True  # Flag to initialize frame stack on first call

        # Skip frames logic
        self.skip_frames = SKIP_FRAMES - 1  # e.g., if SKIP_FRAMES=4, skip 3 frames
        self.skip_count = 0
        self.last_action = 0

        # Step counter for periodic garbage collection
        self.step_counter = 0

    def act(self, observation):
        # Increment step counter for garbage collection
        self.step_counter += 1

        # Ensure the observation is in the correct format
        observation = np.ascontiguousarray(observation)  # Fix negative strides
        if observation.shape != (240, 256, 3):  # Expected raw shape
            raise ValueError(f"Expected observation shape (240, 256, 3), but got {observation.shape}")

        # Preprocess the observation (grayscale, resize)
        processed_frame = self.transform(observation).squeeze(0).numpy()  # Shape: (84, 90)

        # Initialize frame stack on first call
        if self.first:
            self.frame_stack.clear()
            for _ in range(4):
                self.frame_stack.append(processed_frame)
            self.first = False

        # Skip frames logic: return last action if still skipping
        if self.skip_count > 0:
            self.skip_count -= 1
            return self.last_action

        # Add to frame stack and stack frames
        self.frame_stack.append(processed_frame)
        stacked_frames = np.stack(self.frame_stack, axis=0)  # Shape: (4, 84, 90)

        # Convert to tensor and add batch dimension
        obs_tensor = torch.tensor(stacked_frames, dtype=torch.float32).unsqueeze(0).to(self.device)  # Shape: (1, 4, 84, 90)

        # Compute Q-values and select action
        with torch.no_grad():
            q_values = self.model(obs_tensor)
            action = q_values.argmax(dim=1).item()

        # Update last action and reset skip count
        self.last_action = action
        self.skip_count = self.skip_frames

        # Clean up to save memory
        del obs_tensor, q_values, processed_frame, stacked_frames

        # Periodic garbage collection (every 50 steps, matching train.py)
        if self.step_counter % 50 == 0:
            gc.collect()

        return action

    def reset(self):
        """Reset the frame stack and skip count for a new episode."""
        self.first = True
        self.skip_count = 0
        self.last_action = 0
        self.step_counter = 0