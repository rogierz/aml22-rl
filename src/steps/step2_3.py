"""
Train two agents with SAC algorithm, on the source and target domain Hoppers respectively.
Then, test each model in the configurations: source→source, source→target, target→target.

The ablation study is ran thanks to the optuna module.
"""
import os
import shutil
from functools import partial

import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure, HParam

from model.env.custom_hopper import *
from src.utils.lr_schedules import LR_SCHEDULES


def sample_sac_params(trial):
    """
    A utility function that samples the hyperparameters value from chosen search space based on the current trial
    The ablated hyperparameters are the discount factor (gamma), the learning rate (lr), the batch size for the
    off-policy training (batch_size) and the learning rate schedule (lr_schedule)

    :param trial: The current optuna trial
    :return: A dictionary of hyperparameters for the sac algorithm
    """
    gamma = trial.suggest_float("gamma", 0.9, 0.99)
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
    """
    The objective_fn function is the objective function that optuna optimizes.
    It takes a trial object as an argument and returns the value of the objective function calculated with the sampled
    hparams. The objective function is the average return for the source→target configuration, which is the lower bound
    for the following steps.

    :param trial: The current optuna trial
    :param logdir: The directory to which store the logs
    :return: The average return of the source→target configuration
    """
    params = sample_sac_params(trial)
    lr = params['learning_rate']
    gamma = params['gamma']
    batch_size = params['batch_size']
    lr_schedule = LR_SCHEDULES[params['lr_schedule']]
    src_trg_avg_return = 0
    params_metric = {}

    last_trained_env = None
    model = None

    for source, target in [('source', 'source'), ('source', 'target'), ('target', 'target')]:
        logger = configure(f"{logdir}/trial_{trial.number}/{source}_{target}",
                           ["tensorboard"])
        env_source = gym.make(f"CustomHopper-{source}-v0")
        env_target = gym.make(f"CustomHopper-{target}-v0")

        # We only want to train in source and target once for each env
        if last_trained_env != source:
            # if we aren't in ('source', 'target') we retrain on target env
            model = SAC('MlpPolicy', env_source, learning_rate=lr_schedule(
                lr), batch_size=batch_size, gamma=gamma)
            model.set_logger(logger)
            model.learn(total_timesteps=100_000, progress_bar=True)

            model.save(os.path.join("trained_models",
                                    f"step2_3_trial_{trial.number}_env_{source}"))

            last_trained_env = source

        n_episodes = 50
        run_avg_return = 0
        for ep in range(n_episodes):
            done = False
            n_steps = 0
            obs = env_target.reset()
            episode_return = 0

            while not done:  # Until the episode is over
                action, _ = model.predict(obs)
                obs, reward, done, info = env_target.step(action)
                n_steps += 1
                episode_return += reward

            logger.record(
                f'episode_return', episode_return)
            logger.dump(ep)
            run_avg_return += episode_return
        run_avg_return /= n_episodes

        params_metric[f"avg_return/{source}_{target}"] = run_avg_return
        if source == 'source' and target == 'target':
            src_trg_avg_return = run_avg_return

    logger.record("hparams", HParam(
        params, params_metric))
    for k, v in params_metric.items():
        logger.record(k, v)
    logger.dump()

    return src_trg_avg_return


def main(base_prefix='.', force=False):
    """
    This function runs the ablation study through the optuna APIs.

    :param base_prefix: Specify the path to the directory where to save the results
    :param force: If it is true (from command line argument), overwrite previous existing logs
    """
    logdir = f"{base_prefix}/sac_tb_step2_3_log"

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
