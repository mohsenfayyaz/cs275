import os
import argparse
from tqdm import tqdm
import gymnasium as gym
from stable_baselines3 import PPO
import envs  # Register new envs


def test(
        environment_id: str,
        checkpoint_dir: str = None,
        steps=500,
):
    print("Testing")
    model = PPO.load(checkpoint_dir)
    env = gym.make(environment_id, render_mode="human")
    observation, info = env.reset()
    # print("demoing with direction:", env.target_direction)
    for _ in range(steps):
        # action = env.action_space.sample()
        action, _states = model.predict(observation, deterministic=False)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
            # print("demoing with direction:", env.target_direction)
    env.close()
