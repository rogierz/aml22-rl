"""
Train two agents with your algorithm of choice, on the source and target domain Hoppers respectively.
Then, test each model and report its average reward over 50 test episodes.
In particular, report results for the following “training→test” configurations: source→source, source→target (lower bound), target→target (upper bound).
"""
from functools import partial
import os
from typing import Callable

import optuna
import gym
from model.env.custom_hopper import *
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import SAC
import shutil


def constant_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Wrapper for no LR schedule
    """
    def func(progress_remaining: float) -> float:
        return initial_value

    return func

def step_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Wrapper for step LR schedule
    """
    def func(progress_remaining: float) -> float:

        if progress_remaining >= 0.7:
            return initial_value / 2
        elif progress_remaining >= 0.4:
            return initial_value / 4
        elif progress_remaining >= 0.1:
            return initial_value / 8
        else:
            return initial_value

    return func


LR_SCHEDULES = {"constant": constant_schedule, "step": step_schedule}


def sample_sac_params(trial):
    gamma = trial.suggest_float("gamma", 0.9, 0.99)
    lr = trial.suggest_float("learning_rate", 1e-3, 5e-3)
    batch_size = trial.suggest_int("batch_size", 128, 512)
    lr_schedule = trial.suggest_categorical("lr_schedule", ["constant", "step"])

    return {
        "learning_rate": lr,
        "gamma": gamma,
        "batch_size": batch_size,
        "lr_schedule": lr_schedule
    }


def objective_fn(trial, logdir='.'):
    params = sample_sac_params(trial)
    lr = params['learning_rate']
    gamma = params['gamma']
    batch_size = params['batch_size']
    lr_schedule = LR_SCHEDULES[params['lr_schedule']]

    src_trg_avg_return = 0
    params_metric = {}
    with SummaryWriter(log_dir=f"{logdir}/trial_{trial.number}") as writer:
        last_trained_env = None

        for source, target in [('source', 'source'), ('source', 'target'), ('target', 'target')]:
            env_source = gym.make(f"CustomHopper-{source}-v0")
            env_target = gym.make(f"CustomHopper-{target}-v0")

            # We only want to train in source and target once for each env
            if last_trained_env != source:
                # if we aren't in ('source', 'target') we retrain on target env
                model = SAC('MlpPolicy', env_source, learning_rate=lr_schedule(lr), batch_size=batch_size, gamma=gamma,
                            tensorboard_log=f"{logdir}/trial_{trial.number}")

                model.learn(total_timesteps=int(1e6), progress_bar=True,
                            tb_log_name=f"SAC_training_{source}")

                last_trained_env = source

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

                writer.add_scalar(f'episode_return', episode_return, ep)

                run_avg_return += episode_return
            run_avg_return /= n_episodes

            params_metric[f"{source}_{target}/avg_return"] = run_avg_return
            if source == 'source' and target == 'target':
                src_trg_avg_return = run_avg_return

        writer.add_hparams(params, params_metric)

    return src_trg_avg_return


def main(base_prefix='.'):
    logdir = f"{base_prefix}/sac_tb_step2_3_log"

    try:
        shutil.rmtree(logdir)
    except Exception as e:
        print(e)

    search_space = {
        "gamma": [0.9, 0.99],
        "learning_rate": [1e-3, 2e-3, 5e-3],
        "batch_size": [128, 256, 512],
        "lr_schedule": ["constant", "step"]
    }

    study = optuna.create_study(
        sampler=optuna.samplers.GridSampler(search_space), direction="maximize", study_name="Our awesome study")

    objective = partial(objective_fn, logdir=logdir)
    study.optimize(objective)


if __name__ == "__main__":
    main()
