from functools import partial
import os
import shutil
import gym
import numpy as np

from gym.wrappers.pixel_observation import PixelObservationWrapper
from gym.wrappers.resize_observation import ResizeObservation
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from gym.wrappers.frame_stack import FrameStack
from datetime import datetime
from enum import Enum
from ..networks.resnet import ResNet, MobileNet, ShuffleNet
from ..utils.wrapper import CustomWrapper


class Variant(Enum):
    NATURECNN = 0
    RESNET = 1
    RESNET_PRETRAIN = 2


def main(base_prefix=".", force=False, variant=None):
    variant_to_do = [variant] if variant is not None else list(Variant)
    for variant in variant_to_do:
        print(f"Running variant {variant.name}...")
        logdir = f"{base_prefix}/sac_tb_step4_2_{variant.value}_log"

        sac_params = {
            "learning_rate": 2e-3,
            "gamma": 0.99,
            "batch_size": 128
        }

        if os.path.isdir(logdir):
            if force:
                try:
                    shutil.rmtree(logdir)
                except Exception as e:
                    print(e)
            else:
                print(f"Directory {logdir} already exists. Shutting down...")
                return

        env = FrameStack(GrayScaleObservation(ResizeObservation(CustomWrapper(
            PixelObservationWrapper(gym.make(f"CustomHopper-UDR-source-v0"))), shape=(64, 64))), 3)

        env_source = FrameStack(GrayScaleObservation(ResizeObservation(CustomWrapper(
            PixelObservationWrapper(gym.make(f"CustomHopper-source-v0"))), shape=(64, 64))), 3)

        env_target = FrameStack(GrayScaleObservation(ResizeObservation(CustomWrapper(
            PixelObservationWrapper(gym.make(f"CustomHopper-target-v0"))), shape=(64, 64))), 3)

        logger = configure(logdir, ["tensorboard"])

        if variant == Variant.NATURECNN:
            model = SAC("CnnPolicy", env, **sac_params,
                        seed=42, buffer_size=10000)
        elif variant == Variant.RESNET:
            policy_kwargs = dict(features_extractor_class=ShuffleNet)
            # for resNet: add policy_kwargs=policy_kwargs as parameter
            model = SAC("CnnPolicy", env, **sac_params,
                        policy_kwargs=policy_kwargs, seed=42, buffer_size=10000)
        else:
            policy_kwargs = dict(
                features_extractor_class=ResNet, features_extractor_kwargs={"pre_train": True})
            model = SAC("CnnPolicy", env, **sac_params,
                        policy_kwargs=policy_kwargs, seed=42, buffer_size=10000)

        model.set_logger(logger)

        model.learn(total_timesteps=250_000, progress_bar=True,
                    tb_log_name=f"SAC_training_frameStack_{variant.name}")

        if os.path.isfile(os.path.join("trained_models", f"step4_2_{variant.name}.zip")):
            fname = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(fname)
            model.save(os.path.join("trained_models",
                       f"step4_2_{variant.name}_{fname}"))
        else:
            model.save(os.path.join("trained_models",
                       f"step4_2_{variant.name}"))

        n_episodes = 50

        for env_name, test_env in [("source", env_source), ("target", env_target)]:
            print(f"Testing on {env_name}")
            run_avg_return = 0
            for ep in range(n_episodes):
                done = False
                n_steps = 0
                obs = test_env.reset()
                obs = np.array(obs)
                episode_return = 0

                while not done:  # Until the episode is over

                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = test_env.step(action)
                    obs = np.array(obs)
                    n_steps += 1
                    episode_return += reward

                logger.record(f'episode_return/{env_name}', episode_return)
                logger.dump(ep)
                run_avg_return += episode_return
            run_avg_return /= n_episodes
            logger.record(f'run_avg_return/{env_name}', run_avg_return)
            logger.dump()


main_naturecnn = partial(main, variant=Variant.NATURECNN)
main_resnet = partial(main, variant=Variant.RESNET)
main_rsenet_pretrained = partial(main, variant=Variant.RESNET_PRETRAIN)
