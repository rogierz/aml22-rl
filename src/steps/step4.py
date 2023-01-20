"""
VISION-BASED REINFORCEMENT LEARNING 
Implement an RL training pipeline that uses raw images as state observations to the agent.
In this final step, you're given more flexibility on how you can implement the pipeline,
leaving room for variants and group's own implementations.
"""

import gym
import cv2
from gym.spaces import Box
from stable_baselines3 import SAC
from model.env.custom_hopper import *
from gym.wrappers.pixel_observation import PixelObservationWrapper
from gym.wrappers.resize_observation import ResizeObservation


class CustomWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            0, 255, shape=(512, 512, 3), dtype=np.uint8)

    def observation(self, obs):
        # obs = obs["pixels"]
        # obs = cv2.resize(obs, (256, 256), interpolation=cv2)
        return obs["pixels"]


def main(base_prefix="."):
    sac_params = {
        "learning_rate": 2e-3,
        "gamma": 0.99,
        "batch_size": 128
    }

    total_timesteps = int(1e5)

    env = gym.make(f"CustomHopper-source-v0")
    env = ResizeObservation(CustomWrapper(
        PixelObservationWrapper(env)), shape=(128, 128))
    obs = env.reset()

    model = SAC('CnnPolicy', env, **sac_params, seed=42, buffer_size=100000)

    model.learn(total_timesteps=total_timesteps, progress_bar=True)


if __name__ == '__main__':
    main()
