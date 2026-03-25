from gymnasium.envs.registration import register

register(
    id='2048-v0',
    entry_point='env.envs:Game2048Env',
)
