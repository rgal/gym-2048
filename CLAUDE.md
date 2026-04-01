## File Operations
- Use `git mv <old> <new>` for all file moves and renames
- Use `git rm <file>` for deletions
- Run `git status` after file operations to verify tracking

## Environment
- Env is registered as `"2048-v0"` (not `"Game2048-v0"`)
- Observation space: `(16, 4, 4)` one-hot channels-first; actions: 0=Up 1=Right 2=Down 3=Left
- Do not change the env's public API (`step`, `reset`, `move`, `shift`, `set_board`, `get_board`, `highest`, `isend`) without updating `test_game2048_env.py`
- Run tests with: `pytest env/envs/test_game2048_env.py`

## Python environment
- Key packages: PyTorch, Stable-Baselines3, Gymnasium, NumPy, Pillow

## Training pipeline
- PPO training: `ppo_train.py` (Stable Baselines 3, `ResNetExtractor` custom feature extractor)
- BC pre-training: `pretrain_bc.py` — run before PPO when human gameplay CSVs are available
- Human gameplay data: collected via `gather_training_data.py`, stored as CSV files
- CSV format (current): 16 board cols + action + reward + 16 next_board cols + done = 35 cols
- Use `training_data.py` for all CSV import/export — do not reimplement
- Always augment BC training data (`augment()` balances action distribution and grows dataset 8×)
- TensorBoard logs: `./tensorboard_logs/`; model checkpoints saved as `ppo_model_*.zip`

## Testing
- Run all tests with: `pytest .`
- `env/envs/test_game2048_env.py` — env logic: shift, move, isend, step behaviour, observation shape
- `test_training_data.py` — training_data.py: load/save, augmentation, split, shuffle, normalisation
- Tests must pass before committing changes to `env/envs/game2048_env.py` or `training_data.py`

## Data files
- Do not commit CSV data files or `.zip` model files to the repo
- Reference data file for testing CSV import: `data/test_data.csv`
