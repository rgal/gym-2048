#!/usr/bin/env python

"""Model training pipeline for the 2048 agent (PyTorch).

Previously used Keras/TensorFlow; migrated to PyTorch.
The public API (build_model, evaluate_model, predict, train) is preserved
so that gather_training_data.py requires minimal changes.
"""

from __future__ import print_function

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import gymnasium as gym

import gym_2048
import training_data as td_module
from model import Game2048Model, build_model, observation_to_tensor, stacked_to_tensor


class BoardDataset(Dataset):
    """PyTorch Dataset wrapping stacked board observations and action labels.

    Args:
        x_stacked: Board observations of shape (N, 4, 4, 16).
        y_digit: Action labels of shape (N, 1) or (N,).
    """

    def __init__(self, x_stacked: np.ndarray, y_digit: np.ndarray) -> None:
        self.x = stacked_to_tensor(x_stacked)
        self.y = torch.from_numpy(np.array(y_digit, dtype=np.int64).reshape(-1))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def train(
    model: Game2048Model,
    x_stacked: np.ndarray,
    y_digit: np.ndarray,
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 0.001,
) -> None:
    """Train the model on stacked board observations.

    Args:
        model: The model to train (modified in-place).
        x_stacked: Board observations of shape (N, 4, 4, 16).
        y_digit: Action labels of shape (N, 1) or (N,).
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate for the Adam optimiser.
    """
    dataset = BoardDataset(x_stacked, y_digit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x_batch)
            correct += (output.argmax(1) == y_batch).sum().item()
        avg_loss = total_loss / len(dataset)
        accuracy = correct / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f} — accuracy: {accuracy:.4f}")


def predict(model: Game2048Model, observation: np.ndarray) -> np.ndarray:
    """Get action probabilities for a single board observation.

    Args:
        model: Trained model.
        observation: Stacked board array of shape (4, 4, 16).

    Returns:
        Action probabilities as a numpy array of shape (4,).
    """
    model.eval()
    with torch.no_grad():
        x = observation_to_tensor(observation)
        return model(x).squeeze(0).numpy()


def choose_action(
    model: Game2048Model,
    observation: np.ndarray,
    epsilon: float = 0.0,
) -> int:
    """Choose an action using an epsilon-greedy policy.

    Args:
        model: Trained model.
        observation: Stacked board array of shape (4, 4, 16).
        epsilon: Probability of choosing a random action.

    Returns:
        Chosen action index (0–3).
    """
    predictions = predict(model, observation)
    if random.uniform(0, 1) > epsilon:
        return int(np.argmax(predictions))
    return random.randint(0, 3)


def evaluate_episode(
    model: Game2048Model,
    env: gym.Env,
    epsilon: float,
    seed: Optional[int] = None,
    agent_seed: Optional[int] = None,
) -> Tuple[float, int, int, int]:
    """Evaluate the model for one episode.

    Args:
        model: Trained model.
        env: Unwrapped gymnasium environment.
        epsilon: Probability of choosing a random action.
        seed: Optional seed for the environment reset.
        agent_seed: Optional seed for the agent's random choices.

    Returns:
        Tuple of (total_reward, moves_taken, total_illegals, highest_tile).
    """
    if agent_seed is not None:
        random.seed(agent_seed)
    else:
        random.seed()

    total_reward = 0.0
    total_illegals = 0
    moves_taken = 0

    state, _ = env.reset(seed=seed)
    while True:
        action = choose_action(model, state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        if info['illegal_move']:
            total_illegals += 1
        moves_taken += 1
        if moves_taken > 2000:
            break
        state = next_state
        if done:
            break

    return total_reward, moves_taken, total_illegals, int(info['highest'])


def evaluate_model(
    model: Game2048Model,
    episodes: int,
    epsilon: float,
) -> dict:
    """Evaluate the model over multiple episodes.

    Args:
        model: Trained model.
        episodes: Number of episodes to run.
        epsilon: Epsilon for epsilon-greedy action selection.

    Returns:
        Dict with keys 'Average score', 'Max score', 'Highest tile', 'Episodes'.
    """
    env = gym.make('2048-v0').unwrapped
    env.set_illegal_move_reward(-1.)

    scores = []
    for i in range(episodes):
        total_reward, moves_taken, total_illegals, highest = evaluate_episode(
            model, env, epsilon, seed=456 + i, agent_seed=123 + i
        )
        print(
            f"Episode {i}, epsilon {epsilon}, highest {highest}, "
            f"reward {total_reward:.1f}, moves {moves_taken}, illegals {total_illegals}"
        )
        scores.append({
            'total_reward': total_reward,
            'highest': highest,
            'moves': moves_taken,
            'illegal_moves': total_illegals,
        })

    env.close()

    average_score = sum(s['total_reward'] for s in scores) / episodes
    max_score = max(s['total_reward'] for s in scores)
    highest_tile = max(s['highest'] for s in scores)
    print(f"Highest tile: {highest_tile}, Average score: {average_score:.1f}, Max score: {max_score:.1f}")

    return {
        'Average score': average_score,
        'Max score': max_score,
        'Highest tile': highest_tile,
        'Episodes': scores,
    }


def report_evaluation_results(results: dict, label: str = 'eval') -> None:
    """Write evaluation results to a CSV file.

    Args:
        results: Results dict as returned by evaluate_model().
        label: Label used in the output filename.
    """
    with open(f'scores_{label}.csv', 'w') as f:
        fieldnames = ['total_reward', 'highest', 'moves', 'illegal_moves']
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        for s in results['Episodes']:
            writer.writerow(s)


if __name__ == '__main__':
    print(f"PyTorch version: {torch.__version__}")

    board_size = 4
    board_layers = 16
    outputs = 4

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help="Training data CSV file")
    parser.add_argument('--output-model', default='model.pt', help="Output model path")
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--filters', type=int, default=64)
    parser.add_argument('--residual-blocks', type=int, default=8)
    args = parser.parse_args()

    model = build_model(board_size, board_layers, outputs, args.filters, args.residual_blocks)
    print(model)

    data = td_module.training_data()
    data.import_csv(args.input)
    data.shuffle()
    training, validation = data.split(0.8)
    training.augment()
    training.make_boards_unique()

    epsilon = 0.1
    evaluation_episodes = 10

    results = evaluate_model(model, evaluation_episodes, epsilon)
    report_evaluation_results(results, 'pretraining')

    train(
        model,
        training.get_x_stacked(),
        training.get_y_digit(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Evaluate on validation set
    val_dataset = BoardDataset(validation.get_x_stacked(), validation.get_y_digit())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            output = model(x_batch)
            val_loss += criterion(output, y_batch).item() * len(x_batch)
            correct += (output.argmax(1) == y_batch).sum().item()
    n = len(val_dataset)
    print(f"Validation — loss: {val_loss / n:.4f} — accuracy: {correct / n:.4f}")

    torch.save(model, args.output_model)
    print(f"Model saved to {args.output_model}")

    results = evaluate_model(model, evaluation_episodes, epsilon)
    report_evaluation_results(results, 'trained')
