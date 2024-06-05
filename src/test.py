import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import gymnasium as gym
from stable_baselines3 import PPO, SAC
import src.envs  # Register new envs


def test(
        environment_id: str,
        checkpoint_dir: str = None,
        steps=500,
        step_callback=None,
        never_reset=False,
        algorithm="sac",  # ppo, sac
        save_predictions=False,
):
    print(f"Testing {algorithm}")
    saved_preds = []
    algo_map = {
        "ppo": PPO,
        "sac": SAC
    }
    algo = algo_map[algorithm]
    model = algo.load(checkpoint_dir)
    env = gym.make(environment_id, render_mode="human")
    observation, info = env.reset()
    # print("demoing with direction:", env.target_direction)
    for _ in range(steps):
        # action = env.action_space.sample()
        if step_callback is not None:
            env, model = step_callback(env, model)
        action, _states = model.predict(observation, deterministic=False)
        observation, reward, terminated, truncated, info = env.step(action)
        # print(observation[-2:], env.target_direction)
        if (terminated or truncated) and not never_reset:
            observation, info = env.reset()
            # print("demoing with direction:", env.target_direction)
        if save_predictions:
            saved_preds.append(action)
            pd.DataFrame(saved_preds).to_json("predictions.json")
    env.close()
