import os
import argparse
from tqdm import tqdm
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import PPO, SAC
import envs  # Register new envs
from test import test


def train(
        environment_id: str,
        checkpoint_dir: str = None,
        show_demo: bool = False,
        epochs=1000,
        saving_interval=100,
        time_steps=1000,
        learning_rate=0.0003,
        batch_size=64,
        output_dir="checkpoints/",
        algorithm="ppo",  # ppo, sac
) -> None:
    """
    Train PPO/SAC on environment_id
    :param environment_id: refer to registered ids in src/envs/__init__.py
    :param checkpoint_dir: Continue training from checkpoint e.x. "checkpoints/10". Start a new training if None.
    :param show_demo: Show visual demo on saving
    :param epochs: Training epochs
    :param saving_interval: Save model after every saving_interval epochs
    :param time_steps: each epoch's time_steps of learning
    :param output_dir: Directory for saving checkpoints
    :param learning_rate:
    :param batch_size:
    :param algorithm:
    :return:
    """
    print(f"Training {algorithm}")
    env = gym.make(environment_id)
    # env = make_vec_env(environment, n_envs=4)

    algo_map = {
        "ppo": PPO,
        "sac": SAC
    }
    algo = algo_map[algorithm]

    if checkpoint_dir is None:
        model = algo("MlpPolicy", env, verbose=0, learning_rate=learning_rate, batch_size=batch_size,
                    tensorboard_log="./tensorboard/")
        start_epoch = 0
    else:
        print("Continuing Training from:", checkpoint_dir)
        model = algo.load(checkpoint_dir, learning_rate=learning_rate, batch_size=batch_size,
                         tensorboard_log="./tensorboard/")
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
                test(environment_id, output_path, algorithm=algorithm)
