import numpy as np
import keyboard
from stable_baselines3 import SAC
from src.train import train
from src.test import test


def test_human():
    global direction, model, MODEL_MAP, KEYBOARD_MAP
    MODEL_MAP = {
        "1": SAC.load("final_checkpoints/2700_walk.zip"),
        "2": SAC.load("final_checkpoints/6400_crouch.zip"),
        # "3": SAC.load("checkpoints_humanoid_standup/2650.zip"),
    }
    KEYBOARD_MAP = {
        "8": [0, 1],
        "5": [0, -1],
        "4": [-1, 0],
        "6": [1, 0]
    }
    direction = np.array([1, 0])
    model = MODEL_MAP["1"]

    def set_direction(key: keyboard.KeyboardEvent):
        global direction
        direction = np.array(KEYBOARD_MAP[key.name])
        print(direction)

    def set_model(key: keyboard.KeyboardEvent):
        global model
        model = MODEL_MAP[key.name]
        print(model)

    for k in KEYBOARD_MAP.keys():
        keyboard.on_press_key(k, set_direction)
    for k in MODEL_MAP.keys():
        keyboard.on_press_key(k, set_model)

    def callback(env, model_):
        global direction, model
        env.unwrapped.set_target(direction)
        return env, model

    test(environment_id="CustomHumanoid-v4", checkpoint_dir="checkpoints_humanoid/2700.zip", steps=20000,
         step_callback=callback, never_reset=True, algorithm="sac")


test_human()
