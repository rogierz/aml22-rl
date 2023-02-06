"""

"""
import os
from enum import Enum
from itertools import product

from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from gym.wrappers.pixel_observation import PixelObservationWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from tqdm import tqdm

from model.env.custom_hopper import *
from src.utils.wrapper import ExtractionWrapper


class TestVariant(Enum):
    MLP = 1
    CNN = 2


class RewardWrapperMode(Enum):
    MINIMIZE = 0
    MAXIMIZE = 1


def main(base_prefix=".", force=False, models_dir="trained_models"):
    reward_wrapper_variants = list(RewardWrapperMode)
    arch_variants = list(TestVariant)
    for arch_variant, reward_variant in product(arch_variants, reward_wrapper_variants):
        if arch_variant == TestVariant.MLP:
            test_env = gym.make('CustomHopper-target-v0')
        else:
            test_env = FrameStack(GrayScaleObservation(ResizeObservation(ExtractionWrapper(
                PixelObservationWrapper(gym.make('CustomHopper-target-v0'))), shape=(64, 64))), 3)

        model = SAC.load(os.path.join(models_dir, f"step4_1_{reward_variant.value}_{arch_variant.name}.zip"),
                         env=test_env)

        logdir = f"{base_prefix}/test_{reward_variant.name}_{arch_variant.name}_log"
        logger = configure(logdir, ["tensorboard"])
        model.set_logger(logger)

        n_episodes = 1000
        run_avg_return = 0
        run_avg_length = 0
        for ep in tqdm(range(n_episodes)):
            done = False
            state = test_env.reset()
            episode_return = 0
            episode_length = 0

            while not done:  # Until the episode is over
                if arch_variant == TestVariant.CNN:
                    state = np.array(state)
                action, _ = model.predict(state)

                state, reward, done, info = test_env.step(action)  # Step the simulator to the next timestep
                episode_return += reward
                episode_length += 1
            logger.record(f'episode_return', episode_return)
            logger.record(f'episode_length', episode_length)
            logger.dump(ep)
            run_avg_return += episode_return
            run_avg_length += episode_length
        run_avg_return /= n_episodes
        run_avg_length /= n_episodes
        logger.record(f'run_avg_return', run_avg_return)
        logger.record(f'run_avg_length', run_avg_length)
        logger.dump()


if __name__ == '__main__':
    main(base_prefix="logs")
