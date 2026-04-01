"""Behavioural cloning pre-training for the 2048 PPO agent.

Loads human gameplay CSV data (from gather_training_data.py), trains the SB3
policy with cross-entropy loss to imitate those actions, then saves the model
ready for PPO fine-tuning with ppo_train.py --pretrained.

Usage:
    python pretrain_bc.py data/test_data.csv
    python pretrain_bc.py data1.csv data2.csv --epochs 20 --output bc_pretrained
    python pretrain_bc.py data/test_data.csv --no-augment --epochs 10 --batch-size 512
"""

from __future__ import annotations

import argparse
import time

import env  # noqa: F401 — registers 2048-v0
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env.envs.game2048_env import stack as env_stack
from ppo_train import ResNetExtractor
from training_data import training_data


def load_csvs(paths: list[str]) -> training_data:
    """Load and merge one or more CSV files using training_data.import_csv."""
    combined = training_data()
    for path in paths:
        td = training_data()
        td.import_csv(path)
        combined.merge(td)
    return combined


def boards_to_obs(boards: np.ndarray) -> np.ndarray:
    """Convert (N, 4, 4) flat boards to (N, 16, 4, 4) one-hot observations."""
    return np.stack([env_stack(b) for b in boards]).astype(np.float32)


def train_bc(
    policy,
    obs: np.ndarray,
    actions: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> None:
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    obs_t = torch.tensor(obs, device=device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)
    n = len(obs_t)

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        correct = 0
        batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            obs_batch = obs_t[idx]
            act_batch = actions_t[idx]

            features = policy.extract_features(obs_batch, policy.features_extractor)
            latent_pi, _ = policy.mlp_extractor(features)
            logits = policy.action_net(latent_pi)

            loss = F.cross_entropy(logits, act_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == act_batch).sum().item()
            batches += 1

        avg_loss = total_loss / batches
        accuracy = correct / n
        print(f"  epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  accuracy={accuracy:.3f}")


def pretrain(args: argparse.Namespace) -> None:
    print(f"Loading data from: {args.data}")
    td = load_csvs(args.data)
    print(f"  {td.size()} samples loaded")

    if not args.no_augment:
        td.augment()
        print(f"  {td.size()} samples after augmentation (8x flip/rotate)")

    obs = boards_to_obs(td.get_x())
    actions = td.get_y_digit().flatten()

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build PPO with the same architecture used in ppo_train.py
    dummy_env = make_vec_env("2048-v0", n_envs=1)
    policy_kwargs = dict(
        features_extractor_class=ResNetExtractor,
        features_extractor_kwargs=dict(
            filters=args.filters,
            residual_blocks=args.residual_blocks,
        ),
        net_arch=[],
    )
    model = PPO(
        "CnnPolicy",
        dummy_env,
        policy_kwargs=policy_kwargs,
        device=device,
    )
    dummy_env.close()

    policy = model.policy
    policy.train()

    action_counts = np.bincount(actions, minlength=4)
    print(f"\nAction distribution: up={action_counts[0]}  right={action_counts[1]}"
          f"  down={action_counts[2]}  left={action_counts[3]}")
    print(f"\nTraining BC: {len(obs)} samples, {args.epochs} epochs, batch={args.batch_size}\n")

    train_bc(
        policy,
        obs,
        actions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    model.save(args.output)
    print(f"\nPre-trained model saved to {args.output}.zip")
    print(f"Use with: python ppo_train.py --pretrained {args.output}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Behavioural cloning pre-training for 2048 PPO")
    p.add_argument("data", nargs="+", help="CSV file(s) from gather_training_data.py")
    p.add_argument("--output", default=f"bc_pretrained_{int(time.time())}",
                   help="Output model path (no .zip extension)")
    p.add_argument("--no-augment", action="store_true",
                   help="Disable 8x board augmentation (flip + 3 rotations)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--filters", type=int, default=64)
    p.add_argument("--residual-blocks", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    pretrain(parse_args())
