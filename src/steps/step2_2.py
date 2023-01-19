"""
Script using the third-party library stable-baselines3 (sb3) and
train the Hopper agent with one algorithm of choice between
TRPO [8], PPO [9] and SAC [7].
"""
from gym.wrappers.pixel_observation import PixelObservationWrapper
from stable_baselines3 import SAC
from model.env.custom_hopper import *
import gym
import shutil
import code


def observation(self, obs):
    return self._add_pixel_observation(obs)['pixels']


def main(base_prefix='.'):
    logdir = f"{base_prefix}/sac_tb_step2_2_log"

    try:
        shutil.rmtree(logdir)
    except Exception as e:
        print(e)

    env = gym.make('CustomHopper-source-v0')

    wrapped_env = PixelObservationWrapper(env)

    wrapped_env.observation = observation

    model = SAC('CnnPolicy', wrapped_env, verbose=1,
                tensorboard_log=logdir)

    code.interact(local=locals())

    model.learn(total_timesteps=1000, progress_bar=True, tb_log_name="run")

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
