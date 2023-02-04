"""
VISION-BASED REINFORCEMENT LEARNING 
Implement an RL training pipeline that uses raw images as state observations to the agent.
In this final step, you're given more flexibility on how you can implement the pipeline,
leaving room for variants and group's own implementations.

Variants: at this stage, you may feel free to implement any idea to attempt at further improving
the sim-to-real transfer in our simple scenario, for the vision-based RL task.
For example, variants may try to investigate domain randomization on the appearance
of the images to improve the transfer from source to target domain. Other possible directions
may include investigating better representation learning techniques (such as transfer learning
from pre-trained convolutional neural networks) or domain augmentation.
Rather than requiring you to obtain actual improvements, this step is for you to go beyond
the guidelines and get a feeling of a research-like approach.

The following variant 
"""
import os
import shutil

from gym.wrappers.pixel_observation import PixelObservationWrapper
from gym.wrappers.resize_observation import ResizeObservation
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from model.env.custom_hopper import *
from ..utils.wrapper import ExtractionWrapper, RewardWrapper, RewardWrapperMode


def main(base_prefix=".", force=False):
    for variant in [RewardWrapperMode.MINIMIZE, RewardWrapperMode.MAXIMIZE]:
        print(f"Executing variant {variant.name}:\n")
        logdir = f"{base_prefix}/sac_tb_step4_1_{variant.value}_log"

        if os.path.isdir(logdir):
            if force:
                try:
                    shutil.rmtree(logdir)
                except Exception as e:
                    print(e)
            else:
                print(f"Directory {logdir} already exists. Shutting down...")
                return

        logger = configure(logdir, ["tensorboard"])

        sac_params = {
            "learning_rate": 2e-3,
            "gamma": 0.99,
            "batch_size": 128
        }

        total_timesteps = int(250_000)

        env = gym.make(f"CustomHopper-UDR-source-v1")
        env_source = ResizeObservation(ExtractionWrapper(
            PixelObservationWrapper(gym.make(f"CustomHopper-source-v0"))), shape=(128, 128))
        env_target = ResizeObservation(ExtractionWrapper(
            PixelObservationWrapper(gym.make(f"CustomHopper-target-v0"))), shape=(128, 128))

        env = RewardWrapper(
            ResizeObservation(
                ExtractionWrapper(
                    PixelObservationWrapper(env)),
                shape=(128, 128)), variant,
            target=(None if variant == RewardWrapperMode.MAXIMIZE else env_target))

        model = SAC('CnnPolicy', env, **sac_params,
                    seed=42, buffer_size=100000)
        model.set_logger(logger)

        model.learn(total_timesteps=total_timesteps,
                    progress_bar=True, tb_log_name="SAC_training_CNN")

        model.save(os.path.join("trained_models", f"step4_1_{variant.value}"))

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
