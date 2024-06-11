import numpy as np
import keyboard
from stable_baselines3 import SAC
from src.train import train
from src.test import test


def test_human():
    global direction, model, reset, MODEL_MAP, KEYBOARD_MAP
    MODEL_MAP = {
        "1": SAC.load("final_checkpoints/4000_stop.zip"),
        "2": SAC.load("final_checkpoints/2700_walk.zip"),
        "3": SAC.load("final_checkpoints/3600_walk_fast.zip"),
        # "3": SAC.load("final_checkpoints/4000_walk_fast2.zip"),
        "7": SAC.load("final_checkpoints/4200_walk_fast3.zip"),
        # "7": SAC.load("final_checkpoints/5900_walk_fast4.zip"),
        "9": SAC.load("final_checkpoints/6400_walk_crazy.zip"),
        # "1": SAC.load("checkpoints_humanoid/0.zip"),
        # "2": SAC.load("checkpoints_humanoid/500.zip"),
        # "3": SAC.load("checkpoints_humanoid/1000.zip"),
        # "9": SAC.load("checkpoints_humanoid/2700.zip"),
    }
    KEYBOARD_MAP = {
        "8": [0, 1],
        "5": [0, -1],
        "4": [-1, 0],
        "6": [1, 0]
    }
    direction = np.array([1, 0])
    model = MODEL_MAP["1"]
    reset = False

    def set_direction(key: keyboard.KeyboardEvent):
        global direction
        direction = np.array(KEYBOARD_MAP[key.name])
        print(direction)

    def set_model(key: keyboard.KeyboardEvent):
        global model
        model = MODEL_MAP[key.name]
        print(model)

    def set_reset(key: keyboard.KeyboardEvent):
        global reset
        reset = True

    for k in KEYBOARD_MAP.keys():
        keyboard.on_press_key(k, set_direction)
    for k in MODEL_MAP.keys():
        keyboard.on_press_key(k, set_model)
    keyboard.on_press_key("0", set_reset)

    def callback(env, model_):
        global direction, model, reset
        env.unwrapped.set_target(direction)
        if reset:
            env.reset_model()
            reset = False
        return env, model

    test(environment_id="CustomHumanoid-v4", checkpoint_dir="checkpoints_humanoid/2700.zip", steps=1000000,
         step_callback=callback, never_reset=True, algorithm="sac")


test_human()
