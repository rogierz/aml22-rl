"""
VISION-BASED REINFORCEMENT LEARNING 
Implement an RL training pipeline that uses raw images as state observations to the agent.
In this final step, you're given more flexibility on how you can implement the pipeline,
leaving room for variants and group's own implementations.
"""
import os
import shutil

from gym.spaces import Box
from gym.wrappers.pixel_observation import PixelObservationWrapper
from gym.wrappers.resize_observation import ResizeObservation
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from model.env.custom_hopper import *
from datetime import datetime


class CustomWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)

    def observation(self, obs):
        return obs["pixels"]


def main(base_prefix=".", force=False):

    logdir = f"{base_prefix}/sac_tb_step4_log"

    if os.path.isdir("logdir"):
        if force:
            try:
                shutil.rmtree(logdir)
            except Exception as e:
                print(e)
        else:
            print(f"Directory {logdir} already exists. Shutting down...")
            return

    logger = configure(logdir, ["stdout", "tensorboard"])

    sac_params = {
        "learning_rate": 2e-3,
        "gamma": 0.99,
        "batch_size": 128
    }

    total_timesteps = int(10)

    env = gym.make(f"CustomHopper-UDR-source-v0")
    env_source = ResizeObservation(CustomWrapper(
                PixelObservationWrapper(gym.make(f"CustomHopper-source-v0"))), shape=(128, 128))
    env_target =  ResizeObservation(CustomWrapper(
                PixelObservationWrapper(gym.make(f"CustomHopper-target-v0"))), shape=(128, 128))

    env = ResizeObservation(CustomWrapper(
                PixelObservationWrapper(env)), shape=(128, 128))
    obs = env.reset()

    model = SAC('CnnPolicy', env, **sac_params, seed=42, buffer_size=100000)
    model.set_logger(logger)

    model.learn(total_timesteps=total_timesteps, progress_bar=True, tb_log_name="SAC_training_CNN")

    if os.path.isfile(os.path.join("trained_models", "step4.zip")):
        fname = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(fname)
        model.save(os.path.join("trained_models", f"step4_{fname}"))
    else:
        model.save(os.path.join("trained_models", "step4"))

    n_episodes = 50

    for env_name, test_env in [("source", env_source), ("target", env_target)]:
        run_avg_return = 0
        for ep in range(n_episodes):
            done = False
            n_steps = 0
            obs = test_env.reset()
            episode_return = 0

            while not done:  # Until the episode is over
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)
                n_steps += 1
                episode_return += reward

            logger.record(f'episode_return/{env_name}', episode_return)
            logger.dump(ep)
            run_avg_return += episode_return
        run_avg_return /= n_episodes
        logger.record(f'run_avg_return/{env_name}', run_avg_return)
        logger.dump()


if __name__ == '__main__':
    main()
