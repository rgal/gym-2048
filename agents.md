# Agents Guide

This file provides context for AI coding assistants working on this project.

## Project Overview

This project implements an AI agent to play the game 2048. It includes:
- A custom **OpenAI Gym environment** for 2048
- Scripts for **gathering and managing supervised learning (SL) training data**
- A **model training pipeline** using supervised learning
- Planned extension to **reinforcement learning (RL)**

---

## Tech Stack

| Area | Current | Target |
|---|---|---|
| Language | Python | Python |
| Deep Learning | TensorFlow / Keras | **PyTorch** |
| Data | NumPy, Pandas | NumPy |
| Environment | Gymnasium |  Gymnasium |

---

## Migration: Keras → PyTorch

The project is actively migrating from TensorFlow/Keras to PyTorch. When writing or editing code:

- **Prefer PyTorch** for all new model code (`torch`, `torch.nn`, `torch.optim`)
- Do **not** introduce new Keras or TensorFlow dependencies
- When touching existing Keras code, flag it as a candidate for migration if it's in scope
- Use PyTorch idioms: `nn.Module` subclasses, `DataLoader`, training loops with explicit `optimizer.zero_grad()` / `loss.backward()` / `optimizer.step()`
- Prefer `torch.utils.data.Dataset` over custom data pipelines where possible

---

## Project Structure

```
/
├── env/                  # OpenAI Gym environment for 2048
├── data/                 # Training data collection and management scripts
├── models/               # Model definitions (migrating to PyTorch)
├── train.py              # Main training script
├── evaluate.py           # Model evaluation
└── agents.md             # This file
```

> Note: Update this structure if it doesn't match the actual layout.

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
- Keep environment logic (`env/`) decoupled from model logic (`models/`)
- Use `numpy` arrays for board state internally; convert to `torch.Tensor` at the model boundary
- Reproducibility: always accept and pass through a `seed` parameter where randomness is involved
- Avoid hardcoding paths — use `pathlib.Path` and config arguments

---

## What to Avoid

- Do not add new TensorFlow or Keras imports to any file
- Do not modify the Gym environment's public API (`reset`, `step`, `render`) without flagging it clearly — downstream scripts depend on it
- Do not store large data files in the repo; use references or config paths instead

---

## Current Priorities

1. Migrate existing model definitions from Keras to PyTorch
2. Ensure the training pipeline works end-to-end with PyTorch
3. Add a reinforcement learning training loop using the existing Gym environment
