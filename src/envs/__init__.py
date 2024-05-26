"""
Register new environments here
"""

from gymnasium.envs.registration import register

register(
    id='ControlAnt-v4',
    entry_point='src.envs.ant:ControllableAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0
)

register(
    id="CustomHumanoid-v4",
    entry_point="src.envs.humanoid:CustomHumanoidEnv",
    max_episode_steps=1000,
)

print("Registered Environments")
