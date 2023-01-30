import gym
import numpy as np
from enum import Enum
from gym.spaces import Box
import torch.nn.functional as F


class CustomWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            0, 255, shape=(128, 128, 3), dtype=np.uint8)

    def observation(self, obs):
        return obs["pixels"]


class RewardWrapperMode(Enum):
    MINIMIZE = 0
    MAXIMIZE = 1


class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, mode=RewardWrapperMode.MAXIMIZE, target=None):
        super().__init__(env)
        if mode != RewardWrapperMode.MINIMIZE and mode != RewardWrapperMode.MAXIMIZE:
            raise ValueError("mode parameter must be one of RewardWrapperMode")
        if mode == RewardWrapperMode.MINIMIZE and target is None:
            raise ValueError("Impossible to minimize if target is None")
        self.env = env
        self.mode = mode
        self.target = env if target is None else target

    def reward(self, reward):
        coeff = 1

        # get the value of the masses of the target environment (always not modified, even in UDR scenario)
        target_masses = self.original_masses
        target_masses = target_masses/np.linalg.norm(target_masses)

        # get the valeu of the masses of the training environment (UDR masses)
        source_masses = self.env.sim.model.body_mass[1:]
        source_masses = source_masses/np.linalg.norm(source_masses)

        distance = source_masses.T @ target_masses  # cosine similarity
        if self.mode == RewardWrapperMode.MINIMIZE:
            coeff = 1 + (1-distance)
        else:
            coeff = 1 + distance

        return coeff*reward


[1, 2, 3]
[2, 2, 3]
