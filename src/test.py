import os
import argparse

import numpy as np
from tqdm import tqdm
import gymnasium as gym
from stable_baselines3 import PPO, SAC
import src.envs  # Register new envs
from pprint import pprint


def test(
        environment_id: str,
        checkpoint_dir: str = None,
        steps=500,
        step_callback=None,
        never_reset=False,
        algorithm="ppo",  # ppo, sac
):
    print(f"Testing {algorithm}")
    algo_map = {
        "ppo": PPO,
        "sac": SAC
    }
    algo = algo_map[algorithm]
    model = algo.load(checkpoint_dir)
    env = gym.make(environment_id, render_mode="human")
    observation, info = env.reset()
    # print("demoing with speed:", env.target_speed)
    for _ in range(steps):
        # action = env.action_space.sample()
        if step_callback is not None:
            env, model = step_callback(env, model)
        action, _states = model.predict(observation, deterministic=False)
        # action = np.zeros((17,))
        observation, reward, terminated, truncated, info = env.step(action)
        # print(observation[-2:], env.target_direction)
        # print(info["x_velocity"])
        # print(observation[5:8])
        # print(env.unwrapped.get_body_com("pelvis")[2])
        # print(np.linalg.norm(env.unwrapped.get_body_com("left_foot")[:2] - env.unwrapped.get_body_com("right_foot")[:2]))
        if (terminated or truncated) and not never_reset:
            observation, info = env.reset()
            # print("demoing with direction:", env.target_direction)
    pprint(env.info)
    env.close()
