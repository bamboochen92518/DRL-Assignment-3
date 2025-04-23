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
import math
from dataclasses import dataclass

# Config for QNet
@dataclass
class Config:
    dueling: bool = True
    distributional_atom_size: int = 51
    distributional_v_min: float = -10.0
    distributional_v_max: float = 10.0

# NoisyLinear (from previous submission, with b_epsilon fix)
class NoisyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features, self.out_features, self.std_init = in_features, out_features, std_init

        self.w_mu = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.w_sigma = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.b_mu = torch.nn.Parameter(torch.Tensor(out_features))
        self.b_sigma = torch.nn.Parameter(torch.Tensor(out_features))

        self.register_buffer('w_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('b_epsilon', torch.Tensor(out_features))

        self.init_parameters()
        self.reset_noise()

    def forward(self, x):
        return torch.nn.functional.linear(x,
                                          self.w_mu + self.w_sigma * self.w_epsilon,
                                          self.b_mu + self.b_sigma * self.b_epsilon)

    def init_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.w_mu.data.uniform_(-mu_range, mu_range)
        self.b_mu.data.uniform_(-mu_range, mu_range)

        self.w_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.b_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _noise_func(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        in_epsilon = self._noise_func(self.in_features)
        out_epsilon = self._noise_func(self.out_features)
        self.w_epsilon.copy_(out_epsilon.ger(in_epsilon))
        self.b_epsilon.copy_(self._noise_func(self.out_features))

# QNet (with fixed NoisyLinear)
class QNet(torch.nn.Module):
    def __init__(self, n_observations, n_actions, config):
        super(QNet, self).__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.config = config

        self.layer1 = torch.nn.Linear(n_observations, 320)
        if self.config.dueling:
            self.layer_v = NoisyLinear(320, self.config.distributional_atom_size)
            self.layer_a = NoisyLinear(320, n_actions * self.config.distributional_atom_size)
        else:
            self.layer_q = NoisyLinear(320, n_actions * self.config.distributional_atom_size)

        if self.config.distributional_atom_size > 1:
            self.support_z = torch.linspace(self.config.distributional_v_min, self.config.distributional_v_max, self.config.distributional_atom_size)

    def forward(self, batch_state, return_dist=False):
        x = torch.relu(self.layer1(batch_state))
        if self.config.dueling:
            v = self.layer_v(x)
            a = self.layer_a(x)
            if self.config.distributional_atom_size <= 1:
                return v + a - a.mean()

            v = v.view(-1, 1, self.config.distributional_atom_size)
            a = a.view(-1, self.n_actions, self.config.distributional_atom_size)
            q_atoms = v + a - a.mean(dim=1, keepdim=True)
            dist = torch.softmax(q_atoms, dim=-1).clamp(min=1e-3)
        else:
            q = self.layer_q(x)
            if self.config.distributional_atom_size <= 1:
                return q
            q = q.view(-1, self.n_actions, self.config.distributional_atom_size)
            dist = torch.softmax(q, dim=-1).clamp(min=1e-3)

        if return_dist:
            return dist

        return torch.sum(dist * self.support_z, dim=2)

    def reset_noise(self):
        if self.config.dueling:
            self.layer_v.reset_noise()
            self.layer_a.reset_noise()
        else:
            self.layer_q.reset_noise()

    def zero_noise(self):
        if self.config.dueling:
            self.layer_v.w_epsilon.zero_()
            self.layer_v.b_epsilon.zero_()
            self.layer_a.w_epsilon.zero_()
            self.layer_a.b_epsilon.zero_()
        else:
            self.layer_q.w_epsilon.zero_()
            self.layer_q.b_epsilon.zero_()

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
        obs = cv2.resize(obs, self.shape[::-1], interpolation=cv2.INTER_AREA)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = np.expand_dims(obs, axis=-1)
        return obs

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
    state = torch.tensor(np.array(state), dtype=torch.float32, device=device).flatten(start_dim=1) / 255.0
    print(f"Warmup state shape: {state.shape}")
    with tqdm(total=warmup_steps, desc="Warmup Steps") as pbar:
        while steps_done < warmup_steps:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=device).flatten(start_dim=1) / 255.0
            reward = np.clip(reward, -1, 1)  # Clip reward to [-1, 1]
            memory.push(state.cpu().numpy(), action, next_state.cpu().numpy(), reward, done)
            state = next_state
            steps_done += 1
            pbar.update(1)
            if done:
                state, info = env.reset()
                state = torch.tensor(np.array(state), dtype=torch.float32, device=device).flatten(start_dim=1) / 255.0
    print("Warmup phase completed.")

    # Main training loop
    with tqdm(range(args.num_episodes), desc="Training Episodes", unit="episode") as pbar:
        for episode in pbar:
            policy_net.reset_noise()
            state, info = env.reset()
            state = torch.tensor(np.array(state), dtype=torch.float32, device=device).flatten(start_dim=1) / 255.0
            print(f"Episode {episode + 1} state shape: {state.shape}")
            episode_reward = 0
            done = False

            while not done:
                with torch.no_grad():
                    state_batched = state if state.dim() > 1 else state.unsqueeze(0)
                    print(f"state_batched shape: {state_batched.shape}")
                    q_values = policy_net(state_batched)
                    print(f"q_values shape: {q_values.shape}")
                    action = q_values.argmax(dim=1).item()
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=device).flatten(start_dim=1) / 255.0
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
                    states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device).flatten(start_dim=1)
                    actions = torch.tensor(batch.action, device=device)
                    rewards = torch.tensor(batch.reward, device=device)
                    next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=device).flatten(start_dim=1)
                    dones = torch.tensor(batch.done, device=device, dtype=torch.float32)
                    print(f"states shape: {states.shape}, next_states shape: {next_states.shape}")

                    with torch.no_grad():
                        next_probs = target_net(next_states, return_dist=True)
                        next_q = torch.sum(next_probs * z, dim=-1)
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

                        l_indices = (l_clamped + offset).view(-1).long()
                        u_indices = (u_clamped + offset).view(-1).long()
                        l_source = (target_probs * (u.float() - b)).view(-1).float()
                        u_source = (target_probs * (b - l.float())).view(-1).float()

                        m_flat.index_add_(0, l_indices, l_source)
                        m_flat.index_add_(0, u_indices, u_source)

                    probs = policy_net(states, return_dist=True)[range(batch_size), actions]
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
        policy_net.zero_noise()  # Deterministic evaluation
        state, info = env.reset()
        state = torch.tensor(np.array(state), dtype=torch.float32, device=device).flatten(start_dim=1) / 255.0
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                state_batched = state if state.dim() > 1 else state.unsqueeze(0)
                q_values = policy_net(state_batched)
                action = q_values.argmax(dim=1).item()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = torch.tensor(np.array(next_state), dtype=torch.float32, device=device).flatten(start_dim=1) / 255.0
            reward = np.clip(reward, -1, 1)  # Clip reward to [-1, 1]
            # Optional reward shaping
            # shaped_reward = reward + info.get('x_pos', 0) * 0.01
            total_reward += reward
        print(f"Evaluation Episode {episode + 1}, Reward: {total_reward:.2f}, x_pos: {info.get('x_pos', 0)}, coins: {info.get('coins', 0)}, time: {info.get('time', 400)}, flag_get: {info.get('flag_get', False)}")

# Main Function
def main():
    parser = argparse.ArgumentParser(description="Rainbow DQN for Super Mario Bros")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--warmup_steps", type=int, default=800, help="Number of warmup steps")
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
    config = Config(dueling=True, distributional_atom_size=args.num_atoms, distributional_v_min=args.V_min, distributional_v_max=args.V_max)
    policy_net = QNet(n_observations=args.resize_shape * args.resize_shape * 1, n_actions=env.action_space.n, config=config).to(device)
    target_net = QNet(n_observations=args.resize_shape * args.resize_shape * 1, n_actions=env.action_space.n, config=config).to(device)
    policy_net.support_z = policy_net.support_z.to(device)
    target_net.support_z = target_net.support_z.to(device)
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