# Agents Guide

This file provides context for AI coding assistants working on this project.

## Project Overview

This project implements an AI agent to play the game 2048. It includes:
- A custom **Gymnasium environment** for 2048
- Scripts for **gathering and managing supervised learning (SL) training data**
- A **model training pipeline** using supervised learning (PyTorch)
- Planned extension to **reinforcement learning (RL)**

---

## Tech Stack

| Area | Stack |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch |
| Data | NumPy |
| Environment | Gymnasium |

---

## PyTorch Conventions

The project uses PyTorch throughout. When writing or editing code:

- Use `nn.Module` subclasses, `DataLoader`, and explicit training loops (`optimizer.zero_grad()` / `loss.backward()` / `optimizer.step()`)
- Prefer `torch.utils.data.Dataset` for data pipelines
- Do **not** introduce TensorFlow or Keras dependencies

---

## Project Structure

Current layout:

```
/
├── gym_2048/                    # Gymnasium environment for 2048
│   └── envs/game2048_env.py     # Game logic and env implementation
├── data/                        # Training data (CSV files)
├── model.py                     # PyTorch model definition (Game2048Model)
├── train.py                     # Training pipeline and evaluation
├── gather_training_data.py      # Interactive training data collection
├── evaluate.py                  # Model evaluation script
├── training_data.py             # Training data management and augmentation
└── agents.md                    # This file
```

Target layout:

```
/
├── env/                  # Gymnasium environment for 2048
├── models/               # Model definitions
├── data/                 # Training data (CSV files)
├── train.py              # Main training script
├── evaluate.py           # Model evaluation
└── agents.md             # This file
```

---

## Domain Knowledge

### The 2048 Game
- The board is a **4×4 grid** of integer tile values (powers of 2)
- At each step the agent chooses one of **4 actions**: up, down, left, right
- Tiles slide and merge; the goal is to reach the 2048 tile (or beyond)
- A common state representation is the **log₂ of tile values** (so 2→1, 4→2, 1024→10, etc.), which normalises the input scale

### Supervised Learning Setup
- Training data consists of **(state, action)** pairs, typically sourced from heuristic or human play
- The model learns to imitate a reference policy

### Reinforcement Learning (Planned)
- The natural next step is to use RL (e.g. DQN, PPO, or A2C) with the Gym environment
- The reward signal is typically the **score gained per step** (sum of merged tile values)
- When adding RL, prefer **Stable-Baselines3** or a clean custom PyTorch RL loop — avoid mixing in Keras-based RL libraries

---

## Coding Conventions

- Python 3.10+
- Type hints on all function signatures
- Docstrings on all public functions and classes (Google style preferred)
- Keep environment logic (`gym_2048/`) decoupled from model logic (`model.py`, `train.py`)
- Use `numpy` arrays for board state internally; convert to `torch.Tensor` at the model boundary
- Reproducibility: always accept and pass through a `seed` parameter where randomness is involved
- Avoid hardcoding paths — use `pathlib.Path` and config arguments

---

## What to Avoid

- Do not add TensorFlow or Keras imports to any file
- Do not modify the Gymnasium environment's public API (`reset`, `step`, `render`) without flagging it clearly — downstream scripts depend on it
- Do not store large data files in the repo; use references or config paths instead

---

## Current Priorities

1. Add a reinforcement learning training loop using the existing Gymnasium environment
