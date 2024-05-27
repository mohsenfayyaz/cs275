from gymnasium.envs.mujoco.ant_v4 import AntEnv
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 50.0,
    "lookat": np.array((0.0, 1.0, 1.0)),
    "elevation": -50.0,
}


class ControllableAntEnvMohsen(AntEnv):

    # def __init__(self, *args, **kwargs):
    #    super().__init__(*args, **kwargs)
    def __init__(
            self,
            xml_file="ant.xml",
            ctrl_cost_weight=0.5,
            use_contact_forces=False,
            contact_cost_weight=5e-4,
            healthy_reward=1.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(0.2, 1.0),
            contact_force_range=(-1.0, 1.0),
            reset_noise_scale=0.3,
            exclude_current_positions_from_observation=True,
            **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            ctrl_cost_weight,
            use_contact_forces,
            contact_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._use_contact_forces = use_contact_forces

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self.target_direction = np.array([-1.0, 1.0])

        obs_shape = 27
        obs_shape += 2  # Add 2 for the target direction
        if not exclude_current_positions_from_observation:
            obs_shape += 2
        if use_contact_forces:
            obs_shape += 84

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # forward_reward = x_velocity

        forward_reward = 2 * cosine_similarity(xy_velocity.reshape(1, -1), self.target_direction.reshape(1, -1))[0][0]
        # if self.target_direction == [0, 0]:
        #     forward_reward = -np.linalg.norm(xy_velocity)
        # print(xy_velocity, self.target_direction, forward_reward)
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        costs = ctrl_cost = self.control_cost(action)

        terminated = self.terminated
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            info["reward_ctrl"] = -contact_cost

        reward = rewards - costs

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self._use_contact_forces:
            contact_force = self.contact_forces.flat.copy()
            return np.concatenate((position, velocity, contact_force))
        else:
            return np.concatenate((position, velocity, self.target_direction))

    def set_target(self, t):
        self.target_direction = t
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
                self.init_qvel
                + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        angle = self.np_random.uniform(0, 2 * np.pi)
        self.target_direction = np.array([np.cos(angle), np.sin(angle)])
        DIR = {
            0: [1, 0],
            1: [-1, 0],
            2: [0, 1],
            3: [0, -1],
            # 4: [0, 0],
        }
        self.target_direction = np.array(DIR[np.random.choice(np.arange(4))])
        print("Reset Direction:", self.target_direction)
        return observation
