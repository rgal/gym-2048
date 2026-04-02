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
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
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
# Callback: record video
# ---------------------------------------------------------------------------

class VideoRecorderCallback(BaseCallback):
    """Records one episode of the current policy every `record_freq` timesteps."""

    def __init__(self, record_freq: int, video_folder: str = "./videos") -> None:
        super().__init__()
        self.record_freq = record_freq
        self.video_folder = video_folder
        self._last_recorded = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_recorded >= self.record_freq:
            self._last_recorded = self.num_timesteps
            eval_env = RecordVideo(
                gym.make("2048-v0", render_mode="rgb_array"),
                video_folder=self.video_folder,
                episode_trigger=lambda _: True,
                name_prefix=f"ppo_{self.num_timesteps}",
                disable_logger=True,
            )
            obs, _ = eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = eval_env.step(int(action))
                done = terminated or truncated
            eval_env.close()
        return True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    env_instance = make_vec_env("2048-v0", n_envs=args.n_envs)

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
        device="mps" if torch.backends.mps.is_available() else "auto",
        tensorboard_log="./tensorboard_logs/",
    )

    if args.pretrained:
        print(f"Loading pre-trained policy weights from {args.pretrained}")
        pretrained = PPO.load(args.pretrained)
        model.policy.load_state_dict(pretrained.policy.state_dict())
        print("  Pre-trained weights loaded.")

    print("Compiling policy (first rollout will be slower)...")
    model.policy = torch.compile(model.policy)

    callbacks: list[BaseCallback] = [HighestTileCallback()]
    if args.video_freq > 0:
        callbacks.append(VideoRecorderCallback(args.video_freq))
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
    p.add_argument("--n-envs", type=int, default=8,
                   help="Number of parallel environments")
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

    p.add_argument("--pretrained", default=None,
                   help="Path to BC pre-trained model from pretrain_bc.py (no .zip extension)")

    p.add_argument("--video-freq", type=int, default=1_000_000,
                   help="Record a video every N timesteps (0 = disable)")

    p.add_argument("--log-interval", type=int, default=10,
                   help="Log every N rollouts")
    p.add_argument("--save-interval", type=int, default=100,
                   help="Save checkpoint every N rollouts (0 = disable)")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
