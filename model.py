"""PyTorch model definition for the 2048 game agent."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A residual block with two convolutional layers and batch normalisation."""

    def __init__(self, filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class Game2048Model(nn.Module):
    """Residual CNN policy network for the 2048 game.

    Accepts a one-hot stacked board in channels-first format and outputs
    a probability distribution over the 4 actions (up, right, down, left).

    Args:
        board_size: Side length of the board (default 4).
        board_layers: Number of one-hot layers per cell (default 16).
        outputs: Number of output actions (default 4).
        filters: Number of convolutional filters (default 64).
        residual_blocks: Number of residual blocks (default 4).
    """

    def __init__(
        self,
        board_size: int = 4,
        board_layers: int = 16,
        outputs: int = 4,
        filters: int = 64,
        residual_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.board_layers = board_layers

        self.initial_conv = nn.Conv2d(board_layers, filters, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(filters) for _ in range(residual_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, board_layers, board_size, board_size).

        Returns:
            Action probabilities of shape (batch, 4).
        """
        x = self.relu(self.initial_bn(self.initial_conv(x)))
        for block in self.res_blocks:
            x = block(x)
        x = self.relu(self.policy_bn(self.policy_conv(x)))
        x = x.flatten(1)
        return torch.softmax(self.policy_fc(x), dim=1)


def build_model(
    board_size: int = 4,
    board_layers: int = 16,
    outputs: int = 4,
    filters: int = 64,
    residual_blocks: int = 4,
) -> Game2048Model:
    """Build and return a new untrained Game2048Model.

    Args:
        board_size: Side length of the board.
        board_layers: Number of one-hot layers per board cell.
        outputs: Number of output actions.
        filters: Number of convolutional filters in each layer.
        residual_blocks: Number of residual blocks.

    Returns:
        A new Game2048Model instance.
    """
    return Game2048Model(board_size, board_layers, outputs, filters, residual_blocks)


def observation_to_tensor(observation: np.ndarray) -> torch.Tensor:
    """Convert a single stacked board observation to a model input tensor.

    Args:
        observation: Numpy array of shape (4, 4, 16) from the gym environment.

    Returns:
        Float tensor of shape (1, 16, 4, 4) ready for model input.
    """
    # (4, 4, 16) -> (16, 4, 4) -> (1, 16, 4, 4)
    arr = observation.astype(np.float32).transpose(2, 0, 1)
    return torch.from_numpy(arr).unsqueeze(0)


def stacked_to_tensor(stacked: np.ndarray) -> torch.Tensor:
    """Convert a batch of stacked board observations to a model input tensor.

    Args:
        stacked: Numpy array of shape (N, 4, 4, 16).

    Returns:
        Float tensor of shape (N, 16, 4, 4).
    """
    # (N, 4, 4, 16) -> (N, 16, 4, 4)
    arr = stacked.astype(np.float32).transpose(0, 3, 1, 2)
    return torch.from_numpy(arr)
