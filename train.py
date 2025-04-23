import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym import Wrapper
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import namedtuple
import random
from tqdm import tqdm
import argparse
import cv2

# Compatibility Wrapper for Old Gym API
class ResetCompatibilityWrapper(Wrapper):
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        obs = np.array(obs, dtype=np.uint8)
        if len(obs.shape) == 3 and obs.shape[-1] == 4:  # RGBA to RGB
            obs = obs[..., :3]
        elif len(obs.shape) == 2:  # Grayscale to RGB
            obs = np.stack([obs] * 3, axis=-1)
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:  # Old API: (obs, reward, done, info)
            obs, reward, done, info = result
            obs = np.array(obs, dtype=np.uint8)
            if len(obs.shape) == 3 and obs.shape[-1] == 4:
                obs = obs[..., :3]
            elif len(obs.shape) == 2:
                obs = np.stack([obs] * 3, axis=-1)
            return obs, reward, done, False, info
        obs, reward, terminated, truncated, info = result
        obs = np.array(obs, dtype=np.uint8)
        if len(obs.shape) == 3 and obs.shape[-1] == 4:
            obs = obs[..., :3]
        elif len(obs.shape) == 2:
            obs = np.stack([obs] * 3, axis=-1)
        return obs, reward, terminated, truncated, info

# Resize and Grayscale Observation Wrapper
class ResizeObservation(Wrapper):
    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        self.shape = shape  # (height, width)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._resize_grayscale(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._resize_grayscale(obs)
        return obs, reward, terminated, truncated, info

    def _resize_grayscale(self, obs):
        # Resize (height, width, 3) to (new_height, new_width, 3)
        obs = cv2.resize(obs, self.shape[::-1], interpolation=cv2.INTER_AREA)
        # Convert to grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Add channel dimension: (height, width) -> (height, width, 1)
        obs = np.expand_dims(obs, axis=-1)
        return obs

# Noisy Linear Layer for Noisy Nets
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.constant_(self.weight_sigma, sigma_init)
        nn.init.uniform_(self.bias_mu, -0.1, 0.1)
        nn.init.constant_(self.bias_sigma, sigma_init)

    def forward(self, x):
        weight_epsilon = torch.randn_like(self.weight_sigma)
        bias_epsilon = torch.randn_like(self.bias_sigma)
        noisy_weight = self.weight_mu + self.weight_sigma * weight_epsilon
        noisy_bias = self.bias_mu + self.bias_sigma * bias_epsilon
        return torch.nn.functional.linear(x, noisy_weight, noisy_bias)

# Dueling Categorical DQN
class DuelingCategoricalDQN(nn.Module):
    def __init__(self, input_shape, num_actions=12, num_atoms=51, V_min=-10, V_max=10):
        super(DuelingCategoricalDQN, self).__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.V_min = V_min
        self.V_max = V_max
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc_value = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, num_atoms)
        )
        self.fc_advantage = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, num_actions * num_atoms)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size(0)
        conv_out = self.conv(x).view(batch_size, -1)
        value = self.fc_value(conv_out).view(batch_size, 1, self.num_atoms)
        advantage = self.fc_advantage(conv_out).view(batch_size, self.num_actions, self.num_atoms)
        probs = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return torch.softmax(probs, dim=-1)

# Prioritized Replay Buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def push(self, *args):
        transition = Transition(*args)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

# Training Function
def train_rainbow_dqn(env, policy_net, target_net, optimizer, memory, args):
    num_actions = env.action_space.n
    num_atoms = args.num_atoms
    V_min = args.V_min
    V_max = args.V_max
    batch_size = args.batch_size
    gamma = args.gamma
    n_step = args.n_step
    target_update = args.target_update
    warmup_steps = args.warmup_steps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    delta_z = (V_max - V_min) / (num_atoms - 1)
    z = torch.linspace(V_min, V_max, num_atoms).to(device)
    history_rewards = []
    steps_done = 0

    # Warmup phase: fill replay buffer with random actions
    print("Starting warmup phase...")
    state, info = env.reset()
    state = torch.tensor(np.array(state), dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0
    with tqdm(total=warmup_steps, desc="Warmup Steps") as pbar:
        while steps_done < warmup_steps:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0
            reward = np.clip(reward, -1, 1)  # Clip reward to [-1, 1]
            memory.push(state.cpu().numpy(), action, next_state.cpu().numpy(), reward, done)
            state = next_state
            steps_done += 1
            pbar.update(1)
            if done:
                state, info = env.reset()
                state = torch.tensor(np.array(state), dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0
    print("Warmup phase completed.")

    # Main training loop
    with tqdm(range(args.num_episodes), desc="Training Episodes", unit="episode") as pbar:
        for episode in pbar:
            state, info = env.reset()
            state = torch.tensor(np.array(state), dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0
            episode_reward = 0
            done = False

            while not done:
                with torch.no_grad():
                    state_batched = state.unsqueeze(0)  # Add batch dimension for network
                    probs = policy_net(state_batched)
                    q_values = (probs * z).sum(dim=-1)
                    action = q_values.argmax(dim=1).item()
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0
                reward = np.clip(reward, -1, 1)  # Clip reward to [-1, 1]
                # Optional reward shaping
                # shaped_reward = reward + info.get('x_pos', 0) * 0.01
                episode_reward += reward
                memory.push(state.cpu().numpy(), action, next_state.cpu().numpy(), reward, done)
                state = next_state
                steps_done += 1

                if len(memory.buffer) >= batch_size:
                    transitions, indices, weights = memory.sample(batch_size, beta=args.beta)
                    batch = Transition(*zip(*transitions))
                    states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
                    actions = torch.tensor(batch.action, device=device)
                    rewards = torch.tensor(batch.reward, device=device)
                    next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=device)
                    dones = torch.tensor(batch.done, device=device, dtype=torch.float32)

                    with torch.no_grad():
                        next_probs = target_net(next_states)
                        next_q = (next_probs * z).sum(dim=-1)
                        next_action = next_q.argmax(dim=1)
                        target_probs = next_probs[range(batch_size), next_action]
                        target_z = rewards.unsqueeze(1) + (gamma ** n_step) * z.unsqueeze(0) * (1 - dones).unsqueeze(1)
                        target_z = target_z.clamp(V_min, V_max)
                        b = (target_z - V_min) / delta_z
                        l = b.floor().long()
                        u = b.ceil().long()
                        m = torch.zeros(batch_size, num_atoms, device=device)

                        # Vectorized projection
                        offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long().unsqueeze(1).expand(batch_size, num_atoms).to(device)
                        l_clamped = l.clamp(0, num_atoms - 1)
                        u_clamped = u.clamp(0, num_atoms - 1)
                        m_flat = m.view(-1)

                        # Debug tensor types and devices
                        print(f"m_flat: dtype={m_flat.dtype}, device={m_flat.device}")
                        print(f"l_clamped + offset: dtype={(l_clamped + offset).dtype}, device={(l_clamped + offset).device}")
                        print(f"u_clamped + offset: dtype={(u_clamped + offset).dtype}, device={(u_clamped + offset).device}")
                        print(f"target_probs * (u.float() - b): dtype={(target_probs * (u.float() - b)).dtype}, device={(target_probs * (u.float() - b)).device}")
                        print(f"target_probs * (b - l.float()): dtype={(target_probs * (b - l.float())).dtype}, device={(target_probs * (b - l.float())).device}")

                        # Explicitly cast indices to long and ensure source is float
                        l_indices = (l_clamped + offset).view(-1).long()
                        u_indices = (u_clamped + offset).view(-1).long()
                        l_source = (target_probs * (u.float() - b)).view(-1).float()
                        u_source = (target_probs * (b - l.float())).view(-1).float()

                        m_flat.index_add_(0, l_indices, l_source)
                        m_flat.index_add_(0, u_indices, u_source)

                    probs = policy_net(states)[range(batch_size), actions]
                    loss = -(m * torch.log(probs + 1e-10)).sum(dim=-1)  # Shape: [batch_size]
                    weighted_loss = torch.tensor(weights, device=device) * loss
                    priorities = weighted_loss.abs().detach().cpu().numpy() + 1e-5  # Shape: [batch_size]
                    loss = weighted_loss.mean()  # Compute mean for optimization
                    memory.update_priorities(indices, priorities)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if steps_done % target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            history_rewards.append(episode_reward)
            pbar.set_postfix({"Reward": f"{episode_reward:.2f}"})
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(history_rewards[-100:]) if len(history_rewards) >= 100 else np.mean(history_rewards)
                print(f"Episode {episode + 1}/{args.num_episodes}, Avg Reward (last 100): {avg_reward:.2f}")
                try:
                    os.makedirs("checkpoints", exist_ok=True)
                    torch.save(policy_net.state_dict(), f"checkpoints/checkpoint_episode_{episode + 1}.pth")
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")
                evaluate_agent(env, policy_net, args)

    return policy_net

# Evaluation Function
def evaluate_agent(env, policy_net, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_atoms = args.num_atoms
    V_min = args.V_min
    V_max = args.V_max
    z = torch.linspace(V_min, V_max, num_atoms).to(device)

    for episode in tqdm(range(args.num_eval_episodes), desc="Evaluation Episodes"):
        state, info = env.reset()
        state = torch.tensor(np.array(state), dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                state_batched = state.unsqueeze(0)  # Add batch dimension for network
                probs = policy_net(state_batched)
                q_values = (probs * z).sum(dim=-1)
                action = q_values.argmax(dim=1).item()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = torch.tensor(np.array(next_state), dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0
            reward = np.clip(reward, -1, 1)  # Clip reward to [-1, 1]
            # Optional reward shaping
            # shaped_reward = reward + info.get('x_pos', 0) * 0.01
            total_reward += reward
        print(f"Evaluation Episode {episode + 1}, Reward: {total_reward:.2f}, x_pos: {info.get('x_pos', 0)}, coins: {info.get('coins', 0)}, time: {info.get('time', 400)}, flag_get: {info.get('flag_get', False)}")

# Main Function
def main():
    parser = argparse.ArgumentParser(description="Rainbow DQN for Super Mario Bros")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--warmup_steps", type=int, default=80000, help="Number of warmup steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--n_step", type=int, default=3, help="N-step return")
    parser.add_argument("--target_update", type=int, default=1000, help="Steps between target network updates")
    parser.add_argument("--num_atoms", type=int, default=51, help="Number of atoms in categorical DQN")
    parser.add_argument("--V_min", type=float, default=-10, help="Minimum value for categorical DQN")
    parser.add_argument("--V_max", type=float, default=10, help="Maximum value for categorical DQN")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate for Adam optimizer")
    parser.add_argument("--eps", type=float, default=1.5e-4, help="Epsilon for Adam optimizer")
    parser.add_argument("--memory_capacity", type=int, default=100000, help="Replay buffer capacity")
    parser.add_argument("--alpha", type=float, default=0.6, help="Prioritized replay alpha")
    parser.add_argument("--beta", type=float, default=0.4, help="Prioritized replay beta")
    parser.add_argument("--num_eval_episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--resize_shape", type=int, default=84, help="Size for resizing observations (resize_shape x resize_shape)")
    args = parser.parse_args()

    # Validate resize_shape
    if args.resize_shape <= 0:
        raise ValueError("resize_shape must be a positive integer")

    # Set up environment
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = ResetCompatibilityWrapper(env)
    env = ResizeObservation(env, shape=(args.resize_shape, args.resize_shape))

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Debug reset output
    result = env.reset()
    print(f"env.reset() output: {type(result)}, {len(result) if isinstance(result, tuple) else result.shape}")
    obs, info = result
    print(f"Observation: type={type(obs)}, shape={obs.shape if isinstance(obs, np.ndarray) else 'N/A'}, dtype={obs.dtype if isinstance(obs, np.ndarray) else 'N/A'}")

    # Initialize networks
    policy_net = DuelingCategoricalDQN((1, args.resize_shape, args.resize_shape), num_actions=env.action_space.n, num_atoms=args.num_atoms, V_min=args.V_min, V_max=args.V_max).to(device)
    target_net = DuelingCategoricalDQN((1, args.resize_shape, args.resize_shape), num_actions=env.action_space.n, num_atoms=args.num_atoms, V_min=args.V_min, V_max=args.V_max).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr, eps=args.eps)
    memory = PrioritizedReplayBuffer(capacity=args.memory_capacity, alpha=args.alpha)

    # Train the model
    print("Starting training...")
    policy_net = train_rainbow_dqn(env, policy_net, target_net, optimizer, memory, args)
    print("Training completed.")

    # Final evaluation
    print("Evaluating final model...")
    evaluate_agent(env, policy_net, args)
    env.close()

if __name__ == "__main__":
    main()