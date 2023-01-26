"""
VISION-BASED REINFORCEMENT LEARNING 
Implement an RL training pipeline that uses raw images as state observations to the agent.
In this final step, you're given more flexibility on how you can implement the pipeline,
leaving room for variants and group's own implementations.
"""
import os
import shutil
import gym

from gym.spaces import Box
from gym.wrappers.pixel_observation import PixelObservationWrapper
from gym.wrappers.resize_observation import ResizeObservation
from stable_baselines3 import SAC
from torch.utils.tensorboard import SummaryWriter

from model.env.custom_hopper import *


class CustomWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)

    def observation(self, obs):
        return obs["pixels"]


def main(base_prefix=".", force=False):
    logdir = f"{base_prefix}/sac_tb_step4_log"

    if os.path.isdir(logdir):
        if force:
            try:
                shutil.rmtree(logdir)
            except Exception as e:
                print(e)
        else:
            print(f"Directory {logdir} already exists. Shutting down...")
            return

    sac_params = {
        "learning_rate": 2e-3,
        "gamma": 0.99,
        "batch_size": 128
    }

    total_timesteps = int(1e5)

    # env = gym.make(f"CustomHopper-UDR-source-v0")
    # env_source = gym.make(f"CustomHopper-source-v0")
    # env_target = gym.make(f"CustomHopper-target-v0")
    # env = ResizeObservation(CustomWrapper(
    #     PixelObservationWrapper(env)), shape=(128, 128))
    # obs = env.reset()
    # print(obs.shape)

    # model = SAC('CnnPolicy', env, **sac_params, seed=42, buffer_size=100000)

    # model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # model.save("cnn_model")

    # n_episodes = 50

    # for env_name, test_env in [("source", env_source), ("target", env_target)]:
    #     run_avg_return = 0
    #     for ep in range(n_episodes):
    #         done = False
    #         n_steps = 0
    #         obs = test_env.reset()
    #         episode_return = 0

    #         while not done:  # Until the episode is over
    #             action, _ = model.predict(obs, deterministic=True)
    #             obs, reward, done, info = test_env.step(action)
    #             n_steps += 1
    #             episode_return += reward
    #             test_env.render()

    #         writer.add_scalar(f'episode_return', episode_return, ep)
    #         print(f"episode {ep} : {episode_return:.2f}")
    #         run_avg_return += episode_return
    #     run_avg_return /= n_episodes
    #     print(f"source - {env_name} : {run_avg_return:.2f} ")

    env = gym.make(f"CustomHopper-UDR-source-v0")
    env_source = gym.make(f"CustomHopper-source-v0")
    env_target = gym.make(f"CustomHopper-target-v0")

    env = ResizeObservation(CustomWrapper(
                PixelObservationWrapper(env)), shape=(128, 128))
    obs = env.reset()
    print(obs.shape)

    model = SAC('CnnPolicy', env, **sac_params, seed=42, buffer_size=100000, tensorboard_log=logdir)

    model.learn(total_timesteps=total_timesteps, progress_bar=True, tb_log_name="SAC_training_CNN")

    model.save(os.path.join("trained_models", "step4"))

    n_episodes = 50
    run_avg_return = 0

    with SummaryWriter(log_dir=logdir) as writer:
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
                    test_env.render()

                writer.add_scalar(f'episode_return', episode_return, ep)
                print(f"episode {ep} : {episode_return:.2f}")
                run_avg_return += episode_return
            run_avg_return /= n_episodes
            print(f"source - {env_name} : {run_avg_return:.2f} ")


if __name__ == '__main__':
    main()
