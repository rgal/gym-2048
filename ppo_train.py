"""PPO training script for the 2048 game environment using Stable Baselines 3.

Observation: (16, 4, 4) one-hot board (channels-first).
Actions:     Discrete(4) — 0=Up, 1=Right, 2=Down, 3=Left
Reward:      Score gained per move; 0 and terminated on illegal move.

Usage:
    python ppo_train.py
    python ppo_train.py --total-timesteps 10_000_000 --filters 128 --anneal-lr
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import env  # noqa: F401 — registers 2048-v0
from model import ResidualBlock


# ---------------------------------------------------------------------------
# Custom feature extractor
# ---------------------------------------------------------------------------

class ResNetExtractor(BaseFeaturesExtractor):
    """Residual CNN trunk as an SB3 feature extractor.

    Takes a (16, 4, 4) observation and returns a flat feature vector of size
    filters * 4 * 4. SB3 then attaches linear policy and value heads.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        filters: int = 64,
        residual_blocks: int = 4,
    ) -> None:
        board_layers, board_h, board_w = observation_space.shape
        features_dim = filters * board_h * board_w
        super().__init__(observation_space, features_dim=features_dim)

        self.trunk = nn.Sequential(
            nn.Conv2d(board_layers, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            *[ResidualBlock(filters) for _ in range(residual_blocks)],
            nn.Flatten(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs.float())


# ---------------------------------------------------------------------------
# Callback: track highest tile
# ---------------------------------------------------------------------------

class HighestTileCallback(BaseCallback):
    """Logs mean highest tile over the last 100 completed episodes."""

    def __init__(self) -> None:
        super().__init__()
        self._highest: deque[int] = deque(maxlen=100)

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "highest" in info:
                self._highest.append(int(info["highest"]))
        if self._highest:
            self.logger.record("rollout/highest_tile", np.mean(self._highest))
        return True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    env_instance = gym.make("2048-v0")

    policy_kwargs = dict(
        features_extractor_class=ResNetExtractor,
        features_extractor_kwargs=dict(
            filters=args.filters,
            residual_blocks=args.residual_blocks,
        ),
        # No extra MLP layers — extractor output goes straight to the heads
        net_arch=[],
    )

    # SB3 lr schedule receives progress_remaining: 1.0 → 0.0
    lr = (lambda p: args.lr * p) if args.anneal_lr else args.lr

    model = PPO(
        "CnnPolicy",
        env_instance,
        learning_rate=lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_coef,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        verbose=1,
        device="auto",
        tensorboard_log="./tensorboard_logs/",
    )

    callbacks: list[BaseCallback] = [HighestTileCallback()]
    if args.save_interval > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=args.save_interval * args.n_steps,
                save_path=".",
                name_prefix="ppo_model",
            )
        )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        log_interval=args.log_interval,
    )

    final_path = f"ppo_model_final_{int(time.time())}.zip"
    model.save(final_path)
    print(f"\nTraining complete. Model saved to {final_path}")
    env_instance.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO training for 2048-v0 via Stable Baselines 3")

    p.add_argument("--total-timesteps", type=int, default=5_000_000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--n-steps", type=int, default=2048,
                   help="Steps collected per rollout")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=4)

    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--anneal-lr", action="store_true",
                   help="Linearly decay LR to 0 over training")

    p.add_argument("--filters", type=int, default=64)
    p.add_argument("--residual-blocks", type=int, default=4)

    p.add_argument("--log-interval", type=int, default=10,
                   help="Log every N rollouts")
    p.add_argument("--save-interval", type=int, default=100,
                   help="Save checkpoint every N rollouts (0 = disable)")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
