"""
This file holds some custom gym Wrappers
"""

from enum import Enum

import gym
import numpy as np
from gym.spaces import Box


class ExtractionWrapper(gym.ObservationWrapper):
    """
    This class extracts the output from the PixelObservationWrapper and adapts it for the ResizeObservationWrapper
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            0, 255, shape=(128, 128, 3), dtype=np.uint8)

    def observation(self, obs):
        """
        This method just returns the pixels of the input observation.

        :param obs: the environment visual observation
        :return: The pixels of the image
        """
        return obs["pixels"]


class RewardWrapperMode(Enum):
    """
    This class is an enum type used to determine which variant of the RewardWrapper to run

    Attributes:
        MINIMIZE = Give a bonus to the reward if the sampled parameters are close to the target ones
        MAXIMIZE = Give a bonus to the reward if the sampled parameters are distant from the target ones
    """
    MINIMIZE = 0
    MAXIMIZE = 1


class RewardWrapper(gym.RewardWrapper):
    """
    This class is a wrapper used to increase or decrease the rewards based on the values of :param mode
    """

    def __init__(self, env, mode=RewardWrapperMode.MAXIMIZE, target=None):
        super().__init__(env)
        if mode not in list(RewardWrapperMode):
            raise ValueError("mode parameter must be one of RewardWrapperMode")
        if mode == RewardWrapperMode.MINIMIZE and target is None:
            raise ValueError("Impossible to minimize if target is None")
        self.env = env
        self.mode = mode
        self.target = target

    def reward(self, reward):
        """
        Override the default reward method: the default value is scaled based on the value of self.mode.
        """
        coeff = 1

        # get the value of the masses of the target environment (always not modified, even in UDR scenario)
        target_masses = self.target.original_masses

        # get the value of the masses of the training environment (UDR masses)
        source_masses = self.env.sim.model.body_mass[1:]

        # cosine similarity
        similarity = (source_masses.T @ target_masses) / \
            (np.linalg.norm(source_masses) * np.linalg.norm(target_masses))
        if self.mode == RewardWrapperMode.MAXIMIZE:
            # bonus if masses very different
            coeff += (1 - similarity)
        else:
            # bonus if masses very similar
            coeff += similarity

        new_reward = coeff * reward
        return new_reward
