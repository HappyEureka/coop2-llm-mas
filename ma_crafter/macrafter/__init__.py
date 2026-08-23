import gymnasium as gym
from .env import Env, Env_Single
from .coop_env import CooperativeEnv

gym.register(
    id='MCrafter-single-agent',
    entry_point='mcrafter:Env_Single')
