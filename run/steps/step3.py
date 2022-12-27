"""
Train two agents with your algorithm of choice, on the source and target domain Hoppers respectively.
Then, test each model and report its average reward over 50 test episodes.
In particular, report results for the following “training→test” configurations: source→source, source→target (lower bound), target→target (upper bound).
"""

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

from model.env.custom_hopper import *
import gym
import hydra


def learn_and_test(source='source', target='target', config={}):
    if source != 'source' and source != 'target':
        raise ValueError(
            f"Expected values: [source|target]. Received values: {source}, {target}")
    if target != 'source' and target != 'target':
        raise ValueError(
            f"Expected values: [source|target]. Received values: {source}, {target}")

    env_source = gym.make(f"CustomHopper-{source}-v0")
    env_target = gym.make(f"CustomHopper-{target}-v0")

    model = SAC('MlpPolicy', env_source, verbose=1,
                tensorboard_log=f"sac_tb_step3_log")
    model.learn(total_timesteps=1000, progress_bar=True,
                tb_log_name=f"run_{source}_{target}")
    n_episodes = 50

    run_avg_return = 0

    writer = SummaryWriter(log_dir=f"sac_tb_step3_log/run_{source}_{target}")
    # logger = model.logger

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
        
        writer.add_scalar(f'episode_return', episode_return, ep)

        run_avg_return += episode_return
    
    writer.close()
    run_avg_return /= n_episodes
    print(f"--- S: {source} | T: {target} ---")
    print(f"run_avg_return: {run_avg_return}")
    print("---------------------------------")


def main(cfg):
    for source, target in [('source', 'source'), ('source', 'target'), ('target', 'target')]:
        learn_and_test(source, target)


if __name__ == "__main__":
    main({})
