import numpy as np
from gymnasium import utils
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv
# from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv, mass_center, DEFAULT_CAMERA_CONFIG
from gymnasium.envs.mujoco.humanoidstandup_v4 import HumanoidStandupEnv, DEFAULT_CAMERA_CONFIG
from sklearn.metrics.pairwise import cosine_similarity


class CustomHumanoidStandupEnv(HumanoidStandupEnv):
    def __init__(self, **kwargs):
        # ADDED {
        self.target_direction = np.array([0.0, 0.0])
        # }
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(376 + 2,), dtype=np.float64
        )
        MujocoEnv.__init__(
            self,
            "humanoidstandup.xml",
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        utils.EzPickle.__init__(self, **kwargs)

    def _get_obs(self):
        data = self.data
        return np.concatenate(
            [
                data.qpos.flat[2:],
                data.qvel.flat,
                data.cinert.flat,
                data.cvel.flat,
                data.qfrc_actuator.flat,
                data.cfrc_ext.flat,
                # ADDED
                self.target_direction,
            ]
        )

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        pos_after = self.data.qpos[2]
        data = self.data
        uph_cost = (pos_after - 0) / self.model.opt.timestep

        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1

        if self.render_mode == "human":
            self.render()
        return (
            self._get_obs(),
            reward,
            False,
            False,
            dict(
                reward_linup=uph_cost,
                reward_quadctrl=-quad_ctrl_cost,
                reward_impact=-quad_impact_cost,
            ),
        )

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(
                low=-c,
                high=c,
                size=self.model.nv,
            ),
        )
        # ADDED {
        DIR = {
            0: [1, 0],
            1: [-1, 0],
            2: [0, 1],
            3: [0, -1],
            # 4: [0, 0],
        }
        # self.target_direction = np.array(DIR[np.random.choice(np.arange(4))])
        self.target_direction = [0, 0]
        # print("Reset Direction:", self.target_direction)
        # print("target_velocity:", self.target_velocity)
        # }
        return self._get_obs()
