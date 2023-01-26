"""
Train two agents with your algorithm of choice, on the source and target domain Hoppers respectively.
Then, test each model and report its average reward over 50 test episodes.
In particular, report results for the following “training→test” configurations: source→source, source→target (lower bound), target→target (upper bound).
"""
import os
import shutil
from functools import partial
from typing import Callable

import optuna
from stable_baselines3 import SAC
from torch.utils.tensorboard import SummaryWriter

from model.env.custom_hopper import *


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
    gamma = 0.99  # trial.suggest_float("gamma", 0.9, 0.99)
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

    metric = 0
    params_metric = {}
    with SummaryWriter(log_dir=f"{logdir}/trial_{trial.number}") as writer:
        last_trained_env = None

        env_source_UDR = gym.make(f"CustomHopper-UDR-source-v0")
        env_source = gym.make(f"CustomHopper-source-v0")
        env_target = gym.make(f"CustomHopper-target-v0")

        model = SAC('MlpPolicy', env_source_UDR, learning_rate=lr_schedule(lr), batch_size=batch_size, gamma=gamma,
                    tensorboard_log=f"{logdir}/trial_{trial.number}")

        model.learn(total_timesteps=50_000, progress_bar=True,
                    tb_log_name=f"SAC_training_UDR")

        model.save(os.path.join("trained_models", f"step3_trial_{trial.number}"))

        n_episodes = 50

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

                writer.add_scalar(f'episode_return/{env_name}', episode_return, ep)

                run_avg_return += episode_return
            run_avg_return /= n_episodes

            params_metric[f"test_{env_name}/avg_return"] = run_avg_return

            if env_name == 'target':
                metric = run_avg_return

        writer.add_hparams(params, params_metric)

    return metric


def main(base_prefix='.', force=False):
    logdir = f"{base_prefix}/sac_tb_step3_log"

    if os.path.isdir(logdir):
        if force:
            try:
                shutil.rmtree(logdir)
            except Exception as e:
                print(e)
        else:
            print(f"Directory {logdir} already exists. Shutting down...")
            return

    search_space = {
        "gamma": [0.99],
        "learning_rate": [1e-3, 2e-3],
        "batch_size": [128, 256],
        "lr_schedule": ["constant", "step"]
    }

    study = optuna.create_study(
        sampler=optuna.samplers.GridSampler(search_space), direction="maximize", study_name="Our awesome study")

    objective = partial(objective_fn, logdir=logdir)
    study.optimize(objective)


if __name__ == "__main__":
    main()
