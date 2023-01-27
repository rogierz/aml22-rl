"""
Script using the third-party library stable-baselines3 (sb3) and
train the Hopper agent with one algorithm of choice between
TRPO [8], PPO [9] and SAC [7].
"""
import os
import shutil
from stable_baselines3 import SAC
from model.env.custom_hopper import *
from stable_baselines3.common.logger import configure


def main(base_prefix='.', force=False):
    logdir = f"{base_prefix}/sac_tb_step2_2_log"

    if os.path.isdir(logdir):
        if force:
            try:
                shutil.rmtree(logdir)
            except Exception as e:
                print(e)
        else:
            print(f"Directory {logdir} already exists. Shutting down...")
            return

    logger = configure(logdir, ["stdout", "tensorboard"])

    env = gym.make('CustomHopper-source-v0')
    model = SAC('MlpPolicy', env, verbose=1)
    model.set_logger(logger)

    model.learn(total_timesteps=1000, progress_bar=True)
    model.save(os.path.join("trained_models", "step2"))

    vec_env = model.get_env()
    n_episodes = 50

    for ep in range(n_episodes):
        done = False
        obs = vec_env.reset()

        while not done:  # Until the episode is over
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)


if __name__ == "__main__":
    main()
