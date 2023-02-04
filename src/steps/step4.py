"""
VISION-BASED REINFORCEMENT LEARNING 
Implement an RL training pipeline that uses raw images as state observations to the agent.
In this final step, you're given more flexibility on how you can implement the pipeline,
leaving room for variants and group's own implementations.
"""
import os
import shutil
from datetime import datetime
from enum import Enum
from functools import partial

from gym.wrappers.pixel_observation import PixelObservationWrapper
from gym.wrappers.resize_observation import ResizeObservation
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from model.env.custom_hopper import *
from ..utils.wrapper import ExtractionWrapper


class VariantStep4(Enum):
    UDR = 1
    NO_UDR = 2


def main(base_prefix=".", force=False, variant=None):
    variant_to_do = [variant] if variant is not None else list(VariantStep4)
    for variant in variant_to_do:
        print(f"Running variant {variant.name}...")
        logdir = f"{base_prefix}/sac_tb_step4_{variant.value}_log"

        if os.path.isdir("logdir"):
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

        total_timesteps = 250_000

        training_env_name = "CustomHopper-UDR-source-v1" if variant == VariantStep4.UDR else "CustomHopper-source-v0"
        env = gym.make(training_env_name)
        env_source = ResizeObservation(ExtractionWrapper(
            PixelObservationWrapper(gym.make(f"CustomHopper-source-v0"))), shape=(128, 128))
        env_target = ResizeObservation(ExtractionWrapper(
            PixelObservationWrapper(gym.make(f"CustomHopper-target-v0"))), shape=(128, 128))

        env = ResizeObservation(ExtractionWrapper(
            PixelObservationWrapper(env)), shape=(128, 128))

        model = SAC('CnnPolicy', env, **sac_params,
                    seed=42, buffer_size=100000)
        model.set_logger(logger)

        model.learn(total_timesteps=total_timesteps,
                    progress_bar=True, tb_log_name="SAC_training_CNN")

        if os.path.isfile(os.path.join("trained_models", f"step4_{variant.name}.zip")):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # print(timestamp)
            model.save(os.path.join("trained_models",
                       f"step4_{variant.name}_{timestamp}"))
        else:
            model.save(os.path.join("trained_models", f"step4_{variant.name}"))

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

main_udr = partial(main, variant=VariantStep4.UDR)
main_no_udr = partial(main, variant=VariantStep4.NO_UDR)
