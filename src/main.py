import numpy as np
from src.train import train
from src.test import test
import keyboard

# train(
#       environment_id="CustomHumanoid-v4", epochs=20000, saving_interval=100, show_demo=True,
#       output_dir="checkpoints_humanoid_crouch_v5/", learning_rate=3e-4, batch_size=256, algorithm="sac",
#       checkpoint_dir="checkpoints_humanoid_crouch_v5/8200.zip", tb_log_name="v5"
# )
# 8200 -- before gait length penalty
# test(environment_id="CustomHumanoid-v4", checkpoint_dir="checkpoints_humanoid_crouch_v5/12500.zip", algorithm="sac")

# train(
#     environment_id="ControlAnt-v4-Mohsen", epochs=20000, saving_interval=100, show_demo=True,
#     output_dir="checkpoints_ant/", learning_rate=0.0003, batch_size=64, time_steps=2000,
#     checkpoint_dir="checkpoints_ant/2600.zip"
# )


def test_ant():
    global direction
    direction = np.array([1, 0])

    def set_direction(key: keyboard.KeyboardEvent):
        KEYBOARD_MAP = {
            "8": [0, 1],
            "5": [0, -1],
            "4": [-1, 0],
            "6": [1, 0]
        }
        global direction
        direction = np.array(KEYBOARD_MAP[key.name])
        print(direction)

    keyboard.on_press_key("8", set_direction)
    keyboard.on_press_key("5", set_direction)
    keyboard.on_press_key("4", set_direction)
    keyboard.on_press_key("6", set_direction)

    def callback(env, model):
        global direction
        env.unwrapped.set_target(direction)
        return env, model

    test(environment_id="ControlAnt-v4-Mohsen", checkpoint_dir="checkpoints_ant/2700.zip", steps=20000,
         step_callback=callback, never_reset=True)

# test_ant()
def test_human():
    global direction, reset
    direction = np.array([1, 0])
    reset = False

    def set_direction(key: keyboard.KeyboardEvent):
        KEYBOARD_MAP = {
            "1": [0, 1],
            "2": [0, -1],
            "3": [-1, 0],
            "4": [1, 0]
        }
        global direction
        direction = np.array(KEYBOARD_MAP[key.name])
        print(direction)
    
    def set_reset(key: keyboard.KeyboardEvent):
        global reset
        reset = True

    keyboard.on_press_key("1", set_direction)
    keyboard.on_press_key("2", set_direction)
    keyboard.on_press_key("3", set_direction)
    keyboard.on_press_key("4", set_direction)
    keyboard.on_press_key("0", set_reset)

    def callback(env, model):
        global direction, reset
        env.unwrapped.set_target(direction)
        if reset:
            env.unwrapped.reset_model()
            reset = False
        return env, model

    test(environment_id="CustomHumanoid-v4", checkpoint_dir="checkpoints_humanoid_crouch_v5/12500.zip", steps=20000,
         step_callback=callback, never_reset=True, algorithm="sac")


test_human()
