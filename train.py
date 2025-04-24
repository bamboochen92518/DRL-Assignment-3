import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random
import os
from torchvision import transforms as T
from gym.wrappers import TimeLimit
import argparse
from tqdm import tqdm
import psutil
import lz4.frame
import gc

# Environment Wrappers
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip
    def step(self, action):
        total_reward, done = 0.0, False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done: break
        return obs, total_reward, done, info

class GrayScaleResize(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.transform = T.Compose([
            T.ToPILImage(), T.Grayscale(), T.Resize((84, 90)), T.ToTensor()
        ])
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1, 84, 90), dtype=np.float32)
    def observation(self, obs):
        return self.transform(obs)

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 1, shape=(shp[0]*k, shp[1], shp[2]), dtype=np.float32)
    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k): self.frames.append(obs)
        return np.concatenate(self.frames, axis=0)
    def step(self, action):
        obs, r, done, info = self.env.step(action)
        self.frames.append(obs)
        return np.concatenate(self.frames, axis=0), r, done, info

# ICM
class ICM(nn.Module):
    def __init__(self, feat_dim, n_actions, embed_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(feat_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU()
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(embed_dim*2, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.forward_model = nn.Sequential(
            nn.Linear(embed_dim + n_actions, 512), nn.ReLU(),
            nn.Linear(512, embed_dim)
        )
    def forward(self, feat, next_feat, action):
        phi = self.encoder(feat)
        phi_next = self.encoder(next_feat)
        inv_in = torch.cat([phi, phi_next], dim=1)
        logits = self.inverse_model(inv_in)
        a_onehot = F.one_hot(action, logits.size(-1)).float()
        fwd_in = torch.cat([phi, a_onehot], dim=1)
        pred_phi_next = self.forward_model(fwd_in)
        return logits, pred_phi_next, phi_next

# Noisy Linear Layer
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

# Dueling CNN
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

# Replay Buffer with Compression
class PrioritizedReplayBuffer:
    def __init__(self, cap, alpha, beta_start, beta_frames, n_step, gamma):
        self.cap, self.alpha = cap, alpha
        self.beta_start, self.beta_frames = beta_start, beta_frames
        self.beta_by_frame = lambda f: min(1.0, beta_start + f * (1.0 - beta_start) / beta_frames)
        self.n_step, self.gamma = n_step, gamma
        self.buffer = []  # Store compressed states
        self.prios = np.zeros((cap,), dtype=np.float32)  # Use float32 to save memory
        self.pos = 0
        self.n_buf = deque(maxlen=n_step)
        self.Exp = namedtuple('Exp', ['s', 'a', 'r', 's2', 'd'])  # Compressed s and s2
    def _get_n_step(self):
        r, s2, d = self.n_buf[-1].r, self.n_buf[-1].s2, self.n_buf[-1].d
        s2 = np.frombuffer(lz4.frame.decompress(s2), dtype=np.float32).reshape(4, 84, 90)  # Decompress s2
        for trans in reversed(list(self.n_buf)[:-1]):
            r = trans.r + self.gamma * r * (1 - trans.d)
            s2_decomp = np.frombuffer(lz4.frame.decompress(trans.s2), dtype=np.float32).reshape(4, 84, 90)
            s2, d = (s2_decomp, trans.d) if trans.d else (s2, d)
        return r, s2, d
    def add(self, s, a, r, s2, d):
        self.n_buf.append(self.Exp(
            lz4.frame.compress(s.tobytes()),  # Compress state
            a,
            np.float32(r),  # Use float32 for reward
            lz4.frame.compress(s2.tobytes()),  # Compress next state
            d
        ))
        if len(self.n_buf) < self.n_step: return
        r_n, s2_n, d_n = self._get_n_step()
        s0, a0 = self.n_buf[0].s, self.n_buf[0].a
        exp = self.Exp(s0, a0, r_n, lz4.frame.compress(s2_n.tobytes()), d_n)
        if len(self.buffer) < self.cap:
            self.buffer.append(exp)
            prio = 1.0 if len(self.buffer) == 1 else self.prios.max()
        else:
            self.buffer[self.pos] = exp
            prio = self.prios.max()
        self.prios[self.pos] = prio
        self.pos = (self.pos + 1) % self.cap
    def sample(self, bs, frame_idx):
        N = len(self.buffer)
        if N == 0: return [], [], [], [], [], [], []
        prios = self.prios[:N] ** self.alpha
        sum_p = prios.sum()
        probs = prios / sum_p if sum_p > 0 else np.ones_like(prios) / N
        idxs = np.random.choice(N, bs, p=probs)
        batch = self.Exp(*zip(*[self.buffer[i] for i in idxs]))
        # Decompress states
        s = [np.frombuffer(lz4.frame.decompress(s_i), dtype=np.float32).reshape(4, 84, 90) for s_i in batch.s]
        s2 = [np.frombuffer(lz4.frame.decompress(s2_i), dtype=np.float32).reshape(4, 84, 90) for s2_i in batch.s2]
        beta = self.beta_by_frame(frame_idx)
        weights = (N * probs[idxs]) ** (-beta)
        weights /= weights.max()
        return (np.array(s), batch.a, batch.r, np.array(s2), batch.d, weights.astype(np.float32), idxs)
    def update_priorities(self, idxs, errors):
        for i, e in zip(idxs, errors):
            self.prios[i] = abs(e) + 1e-6

# Agent
class Agent:
    def __init__(self, obs_shape, n_actions, device, args):
        self.device = device
        self.n_actions = n_actions
        self.online = DuelingCNN(obs_shape[0], n_actions, args.noisy_sigma_init).to(device)
        self.target = DuelingCNN(obs_shape[0], n_actions, args.noisy_sigma_init).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt = optim.Adam(self.online.parameters(), lr=args.lr, eps=args.adam_eps)
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape).to(device)
            feat_dim = self.online.features(dummy).shape[1]
        self.icm = ICM(feat_dim, n_actions, embed_dim=args.icm_embed_dim).to(device)
        self.icm_opt = optim.Adam(self.icm.parameters(), lr=args.icm_lr)
        self.buffer = PrioritizedReplayBuffer(
            args.buffer_capacity, args.per_alpha, args.per_beta, args.per_beta_frames, args.n_step, args.gamma
        )
        self.args = args
        self.frame_idx = 0
    def act(self, state):
        s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online(s_t)
        a = int(q.argmax(1).item())
        del s_t, q  # Free memory
        return a
    def push(self, s, a, r, s2, d):
        self.buffer.add(s, a, r, s2, d)
    def learn(self):
        if self.frame_idx < self.args.batch_size: return
        s, a, r_ext, s2, d, w, idxs = self.buffer.sample(self.args.batch_size, self.frame_idx)
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)
        w = torch.tensor(w, dtype=torch.float32, device=self.device)
        r_ext = torch.tensor(r_ext, dtype=torch.float32, device=self.device)
        feat = self.online.features(s / 255.0)
        nxt_feat = self.online.features(s2 / 255.0)
        feat_icm = feat.detach()
        nxt_feat_icm = nxt_feat.detach()
        logits, pred_phi_n, true_phi_n = self.icm(feat_icm, nxt_feat_icm, a)
        inv_loss = F.cross_entropy(logits, a)
        fwd_loss = F.mse_loss(pred_phi_n, true_phi_n.detach())
        icm_loss = (1 - self.args.icm_beta) * inv_loss + self.args.icm_beta * fwd_loss
        with torch.no_grad():
            int_r = self.args.icm_eta * 0.5 * (pred_phi_n - true_phi_n).pow(2).sum(dim=1)
        q_pred = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        a_n = self.online(s2).argmax(1)
        q_next = self.target(s2).gather(1, a_n.unsqueeze(1)).squeeze(1)
        total_r = r_ext + int_r
        q_tar = total_r + (self.args.gamma ** self.args.n_step) * q_next * (1 - d)
        td = q_pred - q_tar.detach()
        dqn_loss = (F.smooth_l1_loss(q_pred, q_tar.detach(), reduction='none') * w).mean()
        self.opt.zero_grad()
        dqn_loss.backward()
        self.opt.step()
        self.online.reset_noise()
        self.target.reset_noise()
        self.buffer.update_priorities(idxs, td.detach().cpu().numpy())
        self.icm_opt.zero_grad()
        icm_loss.backward()
        self.icm_opt.step()
        if self.frame_idx % self.args.copy_network_freq == 0:
            self.target.load_state_dict(self.online.state_dict())
        # Free memory
        del s, s2, a, d, w, r_ext, feat, nxt_feat, feat_icm, nxt_feat_icm, logits, pred_phi_n, true_phi_n
        del int_r, q_pred, a_n, q_next, total_r, q_tar, td, dqn_loss, icm_loss

# Utility Function for Memory Usage
def get_memory_usage(device):
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024 / 1024  # Convert to MB
    gpu_mem = "N/A"
    if device.type == "cuda":
        gpu_mem = torch.cuda.memory_allocated(device) / 1024 / 1024  # Convert to MB
    return cpu_mem, gpu_mem

# Training Loop
def train(args, checkpoint_path='checkpoints/rainbow_icm.pth'):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, args.skip_frames)
    env = GrayScaleResize(env)
    env = FrameStack(env, 4)
    env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = Agent(env.observation_space.shape, env.action_space.n, device, args)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    start_ep, fi = 1, 0
    if os.path.isfile(checkpoint_path):
        ck = torch.load(checkpoint_path, map_location=device)
        agent.online.load_state_dict(ck['model'])
        agent.target.load_state_dict(ck['model'])
        agent.opt.load_state_dict(ck['optimizer'])
        agent.icm_opt.load_state_dict(ck['icm_opt'])
        fi = ck.get('frame_idx', 0)
        start_ep = ck.get('episode', 0) + 1
    agent.frame_idx = fi
    raw = env.reset()
    state = raw
    last_cpu_mem, last_gpu_mem = None, None
    while len(agent.buffer.buffer) < args.batch_size:
        a = agent.act(state)
        nxt, r, d, _ = env.step(a)
        agent.push(state, a, r, nxt, d)
        state = nxt
        if d: state = env.reset()
    hist = {'reward': [], 'env_reward': [], 'stage': [], 'Trun': []}
    with tqdm(range(start_ep, args.num_episodes + 1), desc="Training Episodes", unit="episode") as pbar:
        for ep in pbar:
            obs = env.reset()
            state = obs
            ep_r, ep_er, prev_x, prev_life = 0, 0, None, None
            done = False
            steps_in_episode = 0
            while not done:
                agent.frame_idx += 1
                steps_in_episode += 1
                a = agent.act(state)
                nxt, r_env, done, info = env.step(a)
                truncated = info.get('TimeLimit.truncated', False)
                done_flag = done and not truncated
                cr = r_env
                x_pos, life = info.get('x_pos'), info.get('life')
                if x_pos is not None:
                    if prev_x is None: prev_x = x_pos
                    dx = x_pos - prev_x
                    cr += args.backward_penalty if dx < 0 else args.stay_penalty if dx == 0 else 0
                    prev_x = x_pos
                if prev_life is None: prev_life = life
                elif life < prev_life:
                    cr += args.death_penalty
                    prev_life = life
                agent.push(state, a, cr, nxt, done_flag)
                agent.learn()
                state = nxt
                ep_r += cr
                ep_er += r_env
                if steps_in_episode % 10 == 0:
                    last_cpu_mem, last_gpu_mem = get_memory_usage(device)
                postfix = {
                    "Steps": steps_in_episode,
                    "CustR": f"{ep_r:.2f}",
                    "EnvR": f"{ep_er:.2f}",
                    "CPU Mem (MB)": f"{last_cpu_mem:.2f}" if last_cpu_mem is not None else "N/A"
                }
                if last_gpu_mem != "N/A":
                    postfix["GPU Mem (MB)"] = f"{last_gpu_mem:.2f}" if last_gpu_mem is not None else "N/A"
                pbar.set_postfix(**postfix)
                if steps_in_episode % 50 == 0:
                    gc.collect()  # Periodic garbage collection
            hist['reward'].append(ep_r)
            hist['env_reward'].append(ep_er)
            hist['stage'].append(env.unwrapped._stage)
            hist['Trun'].append("TERMINATED" if done else "TRUNCATED")
            last_cpu_mem, last_gpu_mem = get_memory_usage(device)
            postfix = {
                "Steps": steps_in_episode,
                "CustR": f"{ep_r:.2f}",
                "EnvR": f"{ep_er:.2f}",
                "CPU Mem (MB)": f"{last_cpu_mem:.2f}" if last_cpu_mem is not None else "N/A"
            }
            if last_gpu_mem != "N/A":
                postfix["GPU Mem (MB)"] = f"{last_gpu_mem:.2f}" if last_gpu_mem is not None else "N/A"
            pbar.set_postfix(**postfix)
            if ep % 100 == 0:
                eval_env = gym_super_mario_bros.make('SuperMarioBros-v0')
                eval_env = JoypadSpace(eval_env, COMPLEX_MOVEMENT)
                eval_env = SkipFrame(eval_env, args.skip_frames)
                eval_env = GrayScaleResize(eval_env)
                eval_env = FrameStack(eval_env, 4)
                eval_env = TimeLimit(eval_env, max_episode_steps=args.max_episode_steps)
                eval_rewards = []
                with tqdm(range(10), desc=f"Evaluation at Episode {ep}", unit="eval ep") as eval_bar:
                    for _ in eval_bar:
                        e_obs = eval_env.reset()
                        done = False
                        total = 0.0
                        step = 0
                        while not done and step < 2000:
                            a = agent.act(e_obs)
                            e_obs, r, done, _ = eval_env.step(a)
                            total += r
                            step += 1
                            if step % 10 == 0:
                                last_cpu_mem, last_gpu_mem = get_memory_usage(device)
                            postfix = {
                                "Steps": step,
                                "Reward": f"{total:.2f}",
                                "CPU Mem (MB)": f"{last_cpu_mem:.2f}" if last_cpu_mem is not None else "N/A"
                            }
                            if last_gpu_mem != "N/A":
                                postfix["GPU Mem (MB)"] = f"{last_gpu_mem:.2f}" if last_gpu_mem is not None else "N/A"
                            eval_bar.set_postfix(**postfix)
                            if step % 50 == 0:
                                gc.collect()  # Periodic garbage collection
                        eval_rewards.append(total)
                eval_env.close()
                print(f"Evaluation at Episode {ep}: Avg Reward over 10 eps: {np.mean(eval_rewards):.2f}")
                torch.save({
                    'model': agent.online.state_dict(),
                    'optimizer': agent.opt.state_dict(),
                    'icm_opt': agent.icm_opt.state_dict(),
                    'frame_idx': agent.frame_idx,
                    'episode': ep
                }, checkpoint_path)
                gc.collect()  # Garbage collection after evaluation
    print("Training complete.")
    return hist

def main():
    parser = argparse.ArgumentParser(description="DQN with ICM for Super Mario Bros")
    parser.add_argument("--num_episodes", type=int, default=100000, help="Number of training episodes")
    parser.add_argument("--copy_network_freq", type=int, default=10000, help="Frequency to copy online to target network")
    parser.add_argument("--buffer_capacity", type=int, default=10000, help="Replay buffer capacity")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--eps_start", type=float, default=1.0, help="Starting epsilon for exploration")
    parser.add_argument("--eps_end", type=float, default=0.01, help="Ending epsilon for exploration")
    parser.add_argument("--eps_decay", type=float, default=0.9999, help="Epsilon decay rate")
    parser.add_argument("--lr", type=float, default=0.00025, help="Learning rate for DQN optimizer")
    parser.add_argument("--adam_eps", type=float, default=0.00015, help="Epsilon for Adam optimizer")
    parser.add_argument("--per_alpha", type=float, default=0.6, help="Prioritized replay alpha")
    parser.add_argument("--per_beta", type=float, default=0.4, help="Initial prioritized replay beta")
    parser.add_argument("--per_beta_frames", type=int, default=2000000, help="Frames for beta annealing")
    parser.add_argument("--per_epsilon", type=float, default=0.1, help="Epsilon for prioritized replay")
    parser.add_argument("--n_step", type=int, default=5, help="N-step return")
    parser.add_argument("--noisy_sigma_init", type=float, default=2.5, help="Initial sigma for noisy layers")
    parser.add_argument("--backward_penalty", type=float, default=0, help="Penalty for moving backward")
    parser.add_argument("--stay_penalty", type=float, default=0, help="Penalty for staying still")
    parser.add_argument("--death_penalty", type=float, default=-100, help="Penalty for dying")
    parser.add_argument("--skip_frames", type=int, default=4, help="Number of frames to skip")
    parser.add_argument("--max_episode_steps", type=int, default=3000, help="Max steps per episode")
    parser.add_argument("--max_frames", type=int, default=44800000, help="Total training frames")
    parser.add_argument("--icm_embed_dim", type=int, default=256, help="Embedding dimension for ICM")
    parser.add_argument("--icm_beta", type=float, default=0.2, help="ICM inverse vs forward loss trade-off")
    parser.add_argument("--icm_eta", type=float, default=0.01, help="ICM intrinsic reward scale")
    parser.add_argument("--icm_lr", type=float, default=1e-4, help="Learning rate for ICM optimizer")
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()