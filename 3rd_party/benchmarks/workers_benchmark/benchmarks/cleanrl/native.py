"""CleanRL native benchmark -- runs PPO directly, no MOSAIC wrapping.

This is the baseline: identical PPO code to CleanRL's ppo.py, inlined
here so we avoid subprocess overhead and measure pure training time
(like rl-tools benchmarks its own training binary directly).
"""

import os
import random
import time
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from workers_benchmark.utils import BenchmarkResult, BenchmarkTimer, print_run_header, print_run_result

os.environ["WANDB_MODE"] = "disabled"


# ---------------------------------------------------------------------------
# Inline PPO agent (from CleanRL's ppo.py, unmodified logic)
# ---------------------------------------------------------------------------

def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class _Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_size = int(np.array(envs.single_observation_space.shape).prod())
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(obs_size, 64)), nn.Tanh(),
            _layer_init(nn.Linear(64, 64)), nn.Tanh(),
            _layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            _layer_init(nn.Linear(obs_size, 64)), nn.Tanh(),
            _layer_init(nn.Linear(64, 64)), nn.Tanh(),
            _layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _run_ppo(
    env_id: str,
    total_timesteps: int,
    num_envs: int,
    seed: int,
    learning_rate: float = 2.5e-4,
    num_steps: int = 128,
) -> Tuple[float, float]:
    """Train PPO and return (final_return, final_length)."""
    device = torch.device("cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def make_env(idx):
        def thunk():
            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(seed + idx)
            env.observation_space.seed(seed + idx)
            return env
        return thunk

    envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(num_envs)])
    batch_size = num_envs * num_steps
    num_iterations = total_timesteps // batch_size

    agent = _Agent(envs).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)

    ep_returns: List[float] = []

    gamma, gae_lambda = 0.99, 0.95
    update_epochs, num_minibatches = 4, 4
    clip_coef, ent_coef, vf_coef, max_grad_norm = 0.2, 0.01, 0.5, 0.5
    minibatch_size = batch_size // num_minibatches

    for iteration in range(1, num_iterations + 1):
        frac = 1.0 - (iteration - 1.0) / num_iterations
        optimizer.param_groups[0]["lr"] = frac * learning_rate

        for step in range(num_steps):
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done_np).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ep_returns.append(float(info["episode"]["r"]))

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(batch_size)
        for _ in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb = b_inds[start:start + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb], b_actions.long()[mb])
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()

                mb_adv = b_advantages[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)).mean()
                newvalue = newvalue.view(-1)
                v_clipped = b_values[mb] + torch.clamp(newvalue - b_values[mb], -clip_coef, clip_coef)
                v_loss = 0.5 * torch.max((newvalue - b_returns[mb]) ** 2, (v_clipped - b_returns[mb]) ** 2).mean()
                loss = pg_loss - ent_coef * entropy.mean() + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

    envs.close()
    final_return = float(np.mean(ep_returns[-10:])) if ep_returns else 0.0
    final_length = 0.0
    return final_return, final_length


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_native_benchmark(config) -> BenchmarkResult:
    """Run CleanRL PPO in-process (no subprocess, no MOSAIC wrapping)."""
    print_run_header(config.worker_name, "native", config.env_id,
                     config.total_timesteps, config.num_envs, config.seed,
                     getattr(config, "_current_iteration", 1),
                     config.iterations)

    with BenchmarkTimer() as timer:
        final_return, final_length = _run_ppo(
            env_id=config.env_id,
            total_timesteps=config.total_timesteps,
            num_envs=config.num_envs,
            seed=config.seed,
            learning_rate=config.learning_rate,
            num_steps=config.num_steps,
        )

    sps = config.total_timesteps / timer.elapsed_seconds if timer.elapsed_seconds > 0 else 0.0

    result = BenchmarkResult(
        worker_name="cleanrl",
        scenario="native",
        env_id=config.env_id,
        total_timesteps=config.total_timesteps,
        wall_time_seconds=timer.elapsed_seconds,
        steps_per_second=sps,
        peak_memory_mb=timer.peak_memory_mb,
        final_episode_return=final_return,
        final_episode_length=final_length,
        seed=config.seed,
        num_envs=config.num_envs,
        iteration=getattr(config, "_current_iteration", 1),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    print_run_result(result)
    return result
