import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Noisy Linear Layer (copied from training script)
class NoisyLinear(nn.Module):
    def __init__(self, in_f, out_f, sigma_init):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.weight_mu = nn.Parameter(torch.empty(out_f, in_f))
        self.weight_sigma = nn.Parameter(torch.empty(out_f, in_f))
        self.register_buffer('weight_epsilon', torch.empty(out_f, in_f))
        self.bias_mu = nn.Parameter(torch.empty(out_f))
        self.bias_sigma = nn.Parameter(torch.empty(out_f))
        self.register_buffer('bias_epsilon', torch.empty(out_f))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()
    def reset_parameters(self):
        bound = 1 / (self.in_f**0.5)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.sigma_init / (self.in_f**0.5))
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_sigma, self.sigma_init / (self.out_f**0.5))
    def reset_noise(self):
        f = lambda x: x.sign() * x.abs().sqrt()
        eps_in = f(torch.randn(self.in_f))
        eps_out = f(torch.randn(self.out_f))
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)
    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)

# Dueling CNN (copied from training script)
class DuelingCNN(nn.Module):
    def __init__(self, in_c, n_actions, noisy_sigma_init):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_c, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_c, 84, 90)
            feat_dim = self.features(dummy).shape[1]
        self.val_noisy = NoisyLinear(feat_dim, 512, sigma_init=noisy_sigma_init)
        self.val = NoisyLinear(512, 1, sigma_init=noisy_sigma_init)
        self.adv_noisy = NoisyLinear(feat_dim, 512, sigma_init=noisy_sigma_init)
        self.adv = NoisyLinear(512, n_actions, sigma_init=noisy_sigma_init)
    def forward(self, x):
        x = self.features(x / 255.0)
        v = F.relu(self.val_noisy(x))
        v = self.val(v)
        a = F.relu(self.adv_noisy(x))
        a = self.adv(a)
        return v + (a - a.mean(dim=1, keepdim=True))
    def reset_noise(self):
        for m in [self.val_noisy, self.val, self.adv_noisy, self.adv]:
            m.reset_noise()

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