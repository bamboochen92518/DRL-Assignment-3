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
from collections import namedtuple, defaultdict
import random
from tqdm import tqdm
import argparse
import cv2
import math
from gym.wrappers import FrameStack

# SumTree (from provided code)
class SumTree:
    def __init__(self, capacity):
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2

        self.tree = defaultdict(float)
        self.dataset = {}

        self.leaf_idx_offset = self.capacity - 1
        self.tree_leaf_pos = 0

        self.length = 0
        self.min_priority, self.max_priority = 1.0, 1.0

    def __len__(self):
        return self.length

    def add(self, data, priority=None):
        self.length = min(self.length + 1, self.capacity)

        if priority is None:
            priority = self.max_priority
        elif priority > self.max_priority:
            self.max_priority = priority
        if priority < self.min_priority:
            self.min_priority = priority

        leaf_real_idx = self.leaf_idx_offset + self.tree_leaf_pos
        self.dataset[leaf_real_idx] = data

        self.update_priority(leaf_real_idx, priority)

        self.tree_leaf_pos += 1
        self.tree_leaf_pos %= self.capacity

    def update_priority(self, leaf_real_idx, priority):
        self.tree[leaf_real_idx] = priority

        while True:
            parent_idx = (leaf_real_idx - 1) // 2
            if leaf_real_idx % 2 == 0:
                self.tree[parent_idx] = self.tree[leaf_real_idx - 1] + self.tree[leaf_real_idx]
            else:
                self.tree[parent_idx] = self.tree[leaf_real_idx] + self.tree[leaf_real_idx + 1]
            leaf_real_idx = parent_idx
            if parent_idx == 0:
                break

    def sample(self, batch_size, n_step=1):
        seg = self.tree[0] / batch_size
        batch_data_idx = []
        for i in range(batch_size):
            rnd_value = random.uniform(a=seg * i, b=seg * (i + 1))
            son_idx = 1
            while True:
                if self.tree[son_idx] < rnd_value:
                    rnd_value -= self.tree[son_idx]
                    son_idx += 1

                if son_idx >= self.leaf_idx_offset:
                    break

                son_idx = son_idx * 2 + 1

            batch_data_idx.append(son_idx)

        batch_seq_data = []
        for idx in batch_data_idx:
            seq_data = [self.dataset[idx]]
            while len(seq_data) < n_step:
                done = seq_data[-1][4]
                if done:
                    break

                next_leaf_idx = self.leaf_idx_offset + (idx + len(seq_data) - self.leaf_idx_offset) % self.capacity
                if next_leaf_idx not in self.dataset:
                    break
                seq_data.append(self.dataset[next_leaf_idx])
            batch_seq_data.append(seq_data)

        return batch_data_idx, batch_seq_data, [self.tree[idx] for idx in batch_data_idx]

# NoisyLinear (unchanged)
class NoisyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, std_init):
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

# ActionRepeatWrapper (unchanged)
class ActionRepeatWrapper(Wrapper):
    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat
        self.step_count = 0

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            self.step_count += 1
            if done or truncated:
                break
        return obs, total_reward, done, truncated, info

    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)

# ResetCompatibilityWrapper (unchanged)
class ResetCompatibilityWrapper(Wrapper):
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        obs = np.array(obs, dtype=np.uint8)
        if len(obs.shape) == 3 and obs.shape[-1] == 4:
            obs = obs[..., :3]
        elif len(obs.shape) == 2:
            obs = np.stack([obs] * 3, axis=-1)
        if isinstance(self.env, (FrameStack, CustomFrameStack)):
            return obs
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
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

# CustomFrameStack (unchanged)
class CustomFrameStack(Wrapper):
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = [None] * num_stack
        self.lz4_compress = False

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        obs = np.array(obs, dtype=np.uint8)
        obs = obs.squeeze(-1)
        self.frames = [obs] * self.num_stack
        return np.stack(self.frames, axis=0), info

    def step(self, action):
        result = self.env.step(action)
        obs, reward, terminated, truncated, info = result
        obs = np.array(obs, dtype=np.uint8)
        obs = obs.squeeze(-1)
        self.frames.pop(0)
        self.frames.append(obs)
        return np.stack(self.frames, axis=0), reward, terminated, truncated, info

# ResizeObservation (unchanged)
class ResizeObservation(Wrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = shape

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

# DuelingCategoricalDQN (modified to add zero_noise)
class DuelingCategoricalDQN(nn.Module):
    def __init__(self, input_shape, num_actions=12, num_atoms=51, V_min=-10, V_max=10, std_init=0.5):
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
            NoisyLinear(conv_out_size, 512, std_init=std_init),
            nn.ReLU(),
            NoisyLinear(512, num_atoms, std_init=std_init)
        )
        self.fc_advantage = nn.Sequential(
            NoisyLinear(conv_out_size, 512, std_init=std_init),
            nn.ReLU(),
            NoisyLinear(512, num_actions * num_atoms, std_init=std_init)
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

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def zero_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.w_epsilon.zero_()
                module.b_epsilon.zero_()

# Transition and stack_data (for compatibility with SumTree)
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'gamma'))

def stack_data(batch_data, device='cpu'):
    assert isinstance(batch_data, (list, tuple))
    if isinstance(batch_data[0], torch.Tensor):
        return torch.stack(batch_data).to(device)
    else:
        return torch.tensor(batch_data).to(device)

# get_beta (unchanged)
def get_beta(step, total_steps, beta_start=0.4, beta_end=1.0):
    fraction = min(1.0, step / total_steps)
    return beta_start + fraction * (beta_end - beta_start)

# train_rainbow_dqn (modified for gradient clipping and SumTree)
def train_rainbow_dqn(env, policy_net, target_net, optimizer, memory, args, start_steps=0, start_episode=0, warmup_done=False):
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
    steps_done = start_steps
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = args.total_steps // 2
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    def get_epsilon(step):
        fraction = min(1.0, step / epsilon_decay)
        return epsilon_start + fraction * (epsilon_end - epsilon_start)

    if not warmup_done:
        print("Starting warmup phase...")
        state, info = env.reset()
        state = torch.tensor(np.array(state), dtype=torch.float32, device=device) / 255.0
        with tqdm(total=warmup_steps, desc="Warmup Steps") as pbar:
            while steps_done < warmup_steps:
                action = env.action_space.sample()
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=device) / 255.0
                reward = reward
                gamma_val = 0.0 if done else gamma
                memory.add(data=(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done), priority=None)
                state = next_state
                steps_done += 1
                pbar.update(1)
                if done:
                    state, info = env.reset()
                    state = torch.tensor(np.array(state), dtype=torch.float32, device=device) / 255.0
        print("Warmup phase completed.")
    else:
        print("Warmup phase skipped (loaded from checkpoint).")

    with tqdm(range(start_episode, args.num_episodes), desc="Training Episodes", unit="episode") as pbar:
        for episode in pbar:
            policy_net.reset_noise()
            state, info = env.reset()
            state = torch.tensor(np.array(state), dtype=torch.float32, device=device) / 255.0
            episode_reward = 0
            done = False

            while not done:
                epsilon = get_epsilon(steps_done)
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        state_batched = state.unsqueeze(0)
                        with torch.cuda.amp.autocast() if device.type == "cuda" else torch.no_grad():
                            probs = policy_net(state_batched)
                            q_values = (probs * z).sum(dim=-1)
                        action = q_values.argmax(dim=1).item()
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=device) / 255.0
                reward = reward
                gamma_val = 0.0 if done else gamma
                episode_reward += reward
                memory.add(data=(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done), priority=None)
                state = next_state
                steps_done += 1

                if len(memory) >= batch_size and steps_done % 4 == 0:
                    # Sample using SumTree
                    batch_idx, batch_seq_data, sampling_weight = memory.sample(batch_size, n_step)
                    bias_max_weight = (len(memory) * memory.min_priority) ** (-get_beta(steps_done, args.total_steps))
                    bias_weight = (len(memory) * np.array(sampling_weight)) ** (-get_beta(steps_done, args.total_steps))
                    weights = torch.tensor(bias_weight / bias_max_weight, dtype=torch.float32, device=device)

                    # Process batch data for n-step learning
                    batch_data = []
                    for seq_data in batch_seq_data:
                        data = list(seq_data.pop(0))
                        data.append(len(seq_data) + 1)
                        if len(seq_data) > 0:
                            data[3] = seq_data[-1][3]
                        for i, _data in enumerate(seq_data):
                            data[2] += (gamma ** (i + 1)) * _data[2]
                        batch_data.append(data)

                    states = stack_data([data[0] for data in batch_data], device) / 255.0
                    actions = stack_data([data[1] for data in batch_data], device)
                    rewards = stack_data([data[2] for data in batch_data], device)
                    next_states = stack_data([data[3] for data in batch_data], device) / 255.0
                    dones = stack_data([data[4] for data in batch_data], device).to(rewards.dtype)
                    n_steps = [data[5] for data in batch_data]

                    with torch.no_grad():
                        with torch.cuda.amp.autocast() if device.type == "cuda" else torch.no_grad():
                            gammas = stack_data([gamma ** n for n in n_steps], device)
                            next_probs = target_net(next_states)
                            next_q = (next_probs * z).sum(dim=-1)
                            next_action = next_q.argmax(dim=1)
                            target_probs = next_probs[range(batch_size), next_action]
                            target_z = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * gammas.unsqueeze(1) * z.unsqueeze(0)
                            target_z = target_z.clamp(V_min, V_max)
                            b = (target_z - V_min) / delta_z
                            l = b.floor().long()
                            u = b.ceil().long()
                            m = torch.zeros(batch_size, num_atoms, device=device)

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

                    with torch.cuda.amp.autocast() if device.type == "cuda" else torch.no_grad():
                        probs = policy_net(states)[range(batch_size), actions]
                        loss = -(m * torch.log(probs + 1e-10)).sum(dim=-1)
                        weighted_loss = weights * loss
                        priorities = loss.detach().cpu().numpy() + 1e-5
                        loss = weighted_loss.mean()

                    optimizer.zero_grad()
                    if device.type == "cuda":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), args.grad_clip_norm)
                        optimizer.step()

                    for idx, priority in zip(batch_idx, priorities):
                        memory.update_priority(idx, (priority ** args.alpha).item())

                if steps_done % target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            history_rewards.append(episode_reward)
            pbar.set_postfix({"Reward": f"{episode_reward:.2f}"})
            if (episode + 1) % args.checkpoint_interval == 0:
                avg_reward = np.mean(history_rewards[-args.checkpoint_interval:]) if len(history_rewards) >= args.checkpoint_interval else np.mean(history_rewards)
                print(f"Episode {episode + 1}/{args.num_episodes}, Avg Reward (last {args.checkpoint_interval}): {avg_reward:.2f}")
                try:
                    os.makedirs("checkpoints", exist_ok=True)
                    checkpoint = {
                        'state_dict': policy_net.state_dict(),
                        'steps_done': steps_done,
                        'episode': episode,
                        'warmup_done': steps_done >= warmup_steps
                    }
                    torch.save(checkpoint, f"checkpoints/checkpoint_episode_{episode + 1}.pth")
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")
                evaluate_agent(env, policy_net, args, episode, steps_done)

    return policy_net

# evaluate_agent (modified for 10 episodes, checkpoint saving, and avg reward only)
def evaluate_agent(env, policy_net, args, episode, steps_done):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_atoms = args.num_atoms
    V_min = args.V_min
    V_max = args.V_max
    z = torch.linspace(V_min, V_max, num_atoms).to(device)

    policy_net.eval()
    total_reward = 0
    num_episodes = 10

    for ep in tqdm(range(num_episodes), desc="Evaluation Episodes"):
        policy_net.zero_noise()  # Use zero_noise during evaluation
        state, info = env.reset()
        state = torch.tensor(np.array(state), dtype=torch.float32, device=device) / 255.0
        episode_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                state_batched = state.unsqueeze(0)
                with torch.cuda.amp.autocast() if device.type == "cuda" else torch.no_grad():
                    probs = policy_net(state_batched)
                    q_values = (probs * z).sum(dim=-1)
                action = q_values.argmax(dim=1).item()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = torch.tensor(np.array(next_state), dtype=torch.float32, device=device) / 255.0
            episode_reward += reward
        total_reward += episode_reward

    avg_reward = total_reward / num_episodes
    print(f"Evaluation after episode {episode + 1}: Average reward = {avg_reward}")

    try:
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = f"checkpoints/eval_ckpt_{steps_done}.pth"
        torch.save({
            'state_dict': policy_net.state_dict(),
            'steps_done': steps_done,
            'episode': episode,
        }, ckpt_path)
        print(f"Saved evaluation checkpoint at {ckpt_path}")
    except Exception as e:
        print(f"Error saving evaluation checkpoint: {e}")

    policy_net.train()
    return avg_reward

# load_checkpoint (unchanged)
def load_checkpoint(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        steps_done = checkpoint.get('steps_done', 0)
        episode = checkpoint.get('episode', 0)
        warmup_done = checkpoint.get('warmup_done', False)
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from Episode {episode + 1}, Steps Done: {steps_done}, Warmup Done: {warmup_done}")
        return steps_done, episode, warmup_done
    except FileNotFoundError:
        print(f"Checkpoint file {checkpoint_path} not found.")
        raise
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

# main (modified to add grad_clip_norm)
def main():
    parser = argparse.ArgumentParser(description="Rainbow DQN for Super Mario Bros")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--warmup_steps", type=int, default=80000, help="Number of warmup steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--n_step", type=int, default=3, help="N-step return")
    parser.add_argument("--target_update", type=int, default=32000, help="Steps between target network updates")
    parser.add_argument("--num_atoms", type=int, default=51, help="Number of atoms in categorical DQN")
    parser.add_argument("--V_min", type=float, default=-10, help="Minimum value for categorical DQN")
    parser.add_argument("--V_max", type=float, default=10, help="Maximum value for categorical DQN")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate for Adam optimizer")
    parser.add_argument("--eps", type=float, default=1.5e-4, help="Epsilon for Adam optimizer")
    parser.add_argument("--memory_capacity", type=int, default=100000, help="Replay buffer capacity")
    parser.add_argument("--alpha", type=float, default=0.5, help="Prioritized replay alpha")
    parser.add_argument("--beta", type=float, default=0.4, help="Initial prioritized replay beta")
    parser.add_argument("--num_eval_episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--resize_shape", type=int, default=84, help="Size for resizing observations")
    parser.add_argument("--total_steps", type=int, default=10000000, help="Total steps for beta annealing")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint file to load (optional)")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="Interval (in episodes) to save checkpoints and evaluate")
    parser.add_argument("--std_init", type=float, default=0.5, help="Initial standard deviation for NoisyLinear layers")
    parser.add_argument("--num_stack", type=int, default=4, help="Number of frames to stack in CustomFrameStack")
    parser.add_argument("--repeat", type=int, default=4, help="Number of action repeats in ActionRepeatWrapper")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Gradient clipping norm")
    args = parser.parse_args()

    # Validate arguments
    if args.resize_shape <= 0:
        raise ValueError("resize_shape must be a positive integer")
    if args.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be a positive integer")
    if args.memory_capacity <= 0:
        raise ValueError("memory_capacity must be a positive integer")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if args.std_init <= 0:
        raise ValueError("std_init must be a positive float")
    if args.num_stack <= 0:
        raise ValueError("num_stack must be a positive integer")
    if args.repeat <= 0:
        raise ValueError("repeat must be a positive integer")
    if args.grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be a positive float")

    # Set up environment
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = ResetCompatibilityWrapper(env)
    env = ResizeObservation(env, shape=(args.resize_shape, args.resize_shape))
    env = CustomFrameStack(env, num_stack=args.num_stack)
    env = ActionRepeatWrapper(env, repeat=args.repeat)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Debug reset output
    result = env.reset()
    if isinstance(result, tuple):
        obs, info = result
        print(f"env.reset() output: tuple, length={len(result)}")
        print(f"Observation: type={type(obs)}, shape={obs.shape}, dtype={obs.dtype}")
    else:
        print(f"env.reset() output: {type(result)}, shape={result.shape}, dtype={result.dtype}")

    # Initialize networks
    policy_net = DuelingCategoricalDQN((args.num_stack, args.resize_shape, args.resize_shape), num_actions=env.action_space.n, num_atoms=args.num_atoms, V_min=args.V_min, V_max=args.V_max, std_init=args.std_init).to(device)
    target_net = DuelingCategoricalDQN((args.num_stack, args.resize_shape, args.resize_shape), num_actions=env.action_space.n, num_atoms=args.num_atoms, V_min=args.V_min, V_max=args.V_max, std_init=args.std_init).to(device)

    # Load checkpoint if provided
    start_steps = 0
    start_episode = 0
    warmup_done = False
    if args.checkpoint_path:
        start_steps, start_episode, warmup_done = load_checkpoint(policy_net, args.checkpoint_path, device)

    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr, eps=args.eps)
    memory = SumTree(capacity=args.memory_capacity)

    # Train the model
    print("Starting training...")
    policy_net = train_rainbow_dqn(env, policy_net, target_net, optimizer, memory, args, start_steps=start_steps, start_episode=start_episode, warmup_done=warmup_done)
    print("Training completed.")

    # Final evaluation
    print("Evaluating final model...")
    evaluate_agent(env, policy_net, args, args.num_episodes, steps_done)
    env.close()

if __name__ == "__main__":
    main()