"""
Script using the third-party library stable-baselines3 (sb3) and
train the Hopper agent with one algorithm of choice between
TRPO [8], PPO [9] and SAC [7].
"""

from stable_baselines3 import SAC
from env.custom_hopper import *
import gym


def main():
    env = gym.make('CustomHopper-source-v0')

    model = SAC('MlpPolicy', env, verbose=1,
                tensorboard_log="./sac_tb_step2a_log")
    model.learn(total_timesteps=10_000, progress_bar=True, tb_log_name="run")

    vec_env = model.get_env()
    n_episodes = 50

    for ep in range(n_episodes):
        done = False
        obs = vec_env.reset()

        while not done:  # Until the episode is over
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            vec_env.render()


if __name__ == "__main__":
    main()
