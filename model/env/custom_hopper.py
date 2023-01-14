"""Implementation of the Hopper environment supporting
domain randomization optimization."""
import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from torch.distributions.uniform import Uniform
from scipy.stats import truncnorm


BODY_PARTS = {'torso': 1, 'thigh': 2, 'leg': 3, 'foot': 4}


class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None, randomize=False, uniform_ratio=0.5):
        self.randomize = randomize
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses

        if domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
            self.sim.model.body_mass[1] -= 1.0

        self.n_episodes = 0
        np.random.seed(42)

        thigh_mass = self.sim.model.body_mass[BODY_PARTS['thigh']]
        leg_mass = self.sim.model.body_mass[BODY_PARTS['leg']]
        foot_mass = self.sim.model.body_mass[BODY_PARTS['foot']]

        self.distributions = {'thigh': Uniform((1 - uniform_ratio) * thigh_mass, (1 + uniform_ratio) * thigh_mass),
                              'leg': Uniform((1 - uniform_ratio) * leg_mass, (1 + uniform_ratio) * leg_mass),
                              'foot': Uniform((1 - uniform_ratio) * foot_mass, (1 + uniform_ratio) * foot_mass)}

    def set_random_parameters(self):
        """Set random masses
        TODO
        """
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution
        """
        torso_mass = self.sim.model.body_mass[BODY_PARTS['torso']]
        thigh_mass = self.distributions['thigh'].sample()
        leg_mass = self.distributions['leg'].sample()
        foot_mass = self.distributions['foot'].sample()

        return np.array([torso_mass, thigh_mass, leg_mass, foot_mass])

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array(self.sim.model.body_mass[1:])
        return masses

    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        print(task)
        self.sim.model.body_mass[1:] = task

    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        if self.randomize and done:
            self.n_episodes += 1
            print(self.n_episodes)
            self.set_random_parameters()

        return ob, reward, done, {}

    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-UDR-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source",
                "randomize": True,
                "uniform_ratio": 0.5}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

