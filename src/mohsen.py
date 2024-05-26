from train import train
from test import test

train(environment_id="CustomHumanoid-v4", epochs=20000, saving_interval=100, show_demo=True, output_dir="checkpoints_humanoid/")
# test(environment_id="CustomHumanoid-v4", checkpoint_dir="checkpoints_humanoid/1000.zip")

# train(environment_id="ControlAnt-v4", saving_interval=100, show_demo=True, output_dir="checkpoints_ant/")
# test(environment_id="CustomHumanoid-v4", checkpoint_dir="src/checkpoints_ant/0.zip")
