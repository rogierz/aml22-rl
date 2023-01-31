import gym
import numpy as np
from enum import Enum
from gym.spaces import Box


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
        if mode not in list(RewardWrapperMode):
            raise ValueError("mode parameter must be one of RewardWrapperMode")
        if mode == RewardWrapperMode.MINIMIZE and target is None:
            raise ValueError("Impossible to minimize if target is None")
        self.env = env
        self.mode = mode
        self.target = env if target is None else target

    def reward(self, reward):
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
            coeff += (1-similarity)
        else:
            # bonus if masses very similar
            coeff += similarity

        new_reward = coeff*reward
        if coeff < 0 or reward < 0 or new_reward < 0:
            print(
                f"DEBUG: found coeff={float(coeff):.2f} and reward={float(reward):.2f}, new reward is {float(new_reward):.2f}")
            print(f"DEBUG: target_masses={target_masses}")
            print(f"DEBUG: source_masses={source_masses}")
            print(f"DEBUG: mode={self.mode.name}")
        return new_reward
