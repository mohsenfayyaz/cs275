import os
import argparse
from tqdm import tqdm
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import PPO
import envs  # Register new envs
from test import test


def train(
        environment_id: str,
        checkpoint_dir: str = None,
        show_demo: bool = False,
        epochs=1000,
        saving_interval=100,
        time_steps=1000,
        output_dir="checkpoints/",
) -> None:
    """
    Train PPO on environment_id
    :param environment_id: refer to registered ids in src/envs/__init__.py
    :param checkpoint_dir: Continue training from checkpoint e.x. "checkpoints/10". Start a new training if None.
    :param show_demo: Show visual demo on saving
    :param epochs: Training epochs
    :param saving_interval: Save model after every saving_interval epochs
    :param time_steps: each epoch's time_steps of learning
    :param output_dir: Directory for saving checkpoints
    :return:
    """
    print("Training")
    env = gym.make(environment_id)
    # env = make_vec_env(environment, n_envs=4)

    if checkpoint_dir is None:
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="./tensorboard/")
        start_epoch = 0
    else:
        model = PPO.load(checkpoint_dir)
        model.set_env(env)
        start_epoch = int(checkpoint_dir.split("/")[-1].replace(".zip", ""))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for e in tqdm(range(start_epoch, start_epoch + epochs + 1)):
        model.learn(total_timesteps=time_steps, reset_num_timesteps=False)
        if e % saving_interval == 0:
            output_path = f"{output_dir}{e}"
            model.save(output_path)
            print("Saved Model:", output_path)
            if show_demo:
                test(environment_id, output_path)
