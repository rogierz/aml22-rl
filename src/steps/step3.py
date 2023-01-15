import shutil

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnMaxEpisodes

from model.env.custom_hopper import *


def main(params={}, base_prefix='.'):
    logdir = f"{base_prefix}/sac_tb_step3_log"

    try:
        shutil.rmtree(logdir)
    except Exception as e:
        print(e)

    sac_params = {
        "learning_rate": 2e-3,
        "gamma": 0.99,
        "batch_size": 512
    }

    total_timesteps = int(1e6)

    env_source_UDR = gym.make(f"CustomHopper-UDR-source-v0")
    env_source = gym.make(f"CustomHopper-source-v0")
    env_target = gym.make(f"CustomHopper-target-v0")

    # callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=100_000, verbose=1)

    model = SAC('MlpPolicy', env_source_UDR, **sac_params, verbose=1, seed=42, tensorboard_log=logdir)

    model.learn(total_timesteps=total_timesteps, progress_bar=True, tb_log_name="run")  # , callback=callback_max_episodes)

    for i, env in enumerate([env_source, env_target]):
        n_episodes = 50
        run_avg_return = 0
        for ep in range(n_episodes):
            done = False
            n_steps = 0
            obs = env.reset()
            episode_return = 0

            while not done:  # Until the episode is over
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                n_steps += 1
                episode_return += reward

            run_avg_return += episode_return
        run_avg_return /= n_episodes

        print(f"AVG RETURN ON config_{i}: {run_avg_return}")


if __name__ == '__main__':
    main()
