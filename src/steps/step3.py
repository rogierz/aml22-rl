"""
Train two agents with your algorithm of choice, on the source and target domain Hoppers respectively.
Then, test each model and report its average reward over 50 test episodes.
In particular, report results for the following “training→test” configurations: source→source, source→target (lower bound), target→target (upper bound).
"""
import os
import shutil
import optuna

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure, HParam
from model.env.custom_hopper import *
from src.utils.lr_schedules import LR_SCHEDULES
from functools import partial


def sample_sac_params(trial):
    gamma = 0.99  # trial.suggest_float("gamma", 0.9, 0.99)
    lr = trial.suggest_float("learning_rate", 1e-3, 5e-3)
    batch_size = trial.suggest_int("batch_size", 128, 512)
    lr_schedule = trial.suggest_categorical(
        "lr_schedule", ["constant", "step"])

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

    logger = configure(f"{logdir}/trial_{trial.number}", ["tensorboard"])
    metric = 0
    params_metric = {}
    env_source_UDR = gym.make(f"CustomHopper-UDR-source-v0")
    env_source = gym.make(f"CustomHopper-source-v0")
    env_target = gym.make(f"CustomHopper-target-v0")

    model = SAC('MlpPolicy', env_source_UDR, learning_rate=lr_schedule(
        lr), batch_size=batch_size, gamma=gamma, verbose=1)
    model.set_logger(logger)
    model.learn(total_timesteps=int(1e6), progress_bar=True,
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

            logger.record(
                f"episode_return/{env_name}", episode_return, exclude="stdout")
            logger.dump(ep)
            run_avg_return += episode_return
        run_avg_return /= n_episodes

        params_metric[f"avg_return/test_{env_name}"] = run_avg_return

        if env_name == 'target':
            metric = run_avg_return

    logger.record("hparams", HParam(params, params_metric), exclude="stdout")
    logger.dump()

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
