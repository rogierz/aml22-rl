"""Sample script for training a control policy on the Hopper environment

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between TRPO, PPO, and SAC.
"""
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnMaxEpisodes

from model.env.custom_hopper import *


# class OurCallback(BaseCallback):
#     def _on_step(self) -> bool:
#         print("step")
#
#         return True


def main():
    # env = gym.make('CustomHopper-source-v0')

    # print('State space:', env.observation_space)  # state-space
    # print('Action space:', env.action_space)  # action-space
    # print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper
    #
    """
        TODO:

            - train a policy with stable-baselines3 on source env
            - test the policy with stable-baselines3 on <source,target> envs
    """

    source = 'source'
    target = 'target'

    sac_params = {
        "learning_rate": 2e-3,
        "gamma": 0.99,
        "batch_size": 512
    }

    total_timesteps = int(1e9)

    env_source_train = gym.make(f"CustomHopper-UDR-{source}-v0")
    env_source_test = gym.make(f"CustomHopper-{source}-v0")
    env_target = gym.make(f"CustomHopper-{target}-v0")

    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=100, verbose=1)

    model = SAC('MlpPolicy', env_source_train, **sac_params, seed=42)

    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback_max_episodes)

    n_episodes = 50
    run_avg_return = 0
    for ep in range(n_episodes):
        done = False
        n_steps = 0
        obs = env_target.reset()
        episode_return = 0

        while not done:  # Until the episode is over
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env_target.step(action)
            n_steps += 1
            episode_return += reward

        run_avg_return += episode_return
    run_avg_return /= n_episodes

    print(f"AVG RETURN: {run_avg_return}")


if __name__ == '__main__':
    main()
