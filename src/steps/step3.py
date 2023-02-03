"""
Train the agent with the SAC algorithm, on the source domain with the Uniform Domain Randomization.
Then, test it on both the source and target environments (not randomized) and report the average reward over 50 test
episodes.
"""
import os
import shutil
from enum import Enum
from functools import partial

import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure, HParam

from model.env.custom_hopper import *
from src.utils.lr_schedules import LR_SCHEDULES


class VariantStep3(Enum):
    INTERVAL = 1
    RELATIVE = 2
    ABSOLUTE = 3


def sample_sac_params(trial):
    """
    A utility function that samples the hyperparameters value from chosen search space based on the current trial
    The ablated hyperparameters are the learning rate (lr), the batch size for the off-policy training (batch_size) and
    the learning rate schedule (lr_schedule). The hyperparameter gamma is hardcoded to 0.99, after noticing that
    the model with this value outperforms any model with lower gamma, regardless the other hparams values.

    :param trial: The current optuna trial
    :return: A dictionary of hyperparameters for the sac algorithm
    """
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


def objective_fn(trial, logdir='.', variant=None):
    """
    The objective_fn function is the objective function that optuna optimizes.
    It takes a trial object as an argument and returns the value of the objective function calculated with the sampled
    hparams. The objective function is the average return for the source→target configuration, which is the lower bound
    for the following steps.

    :param trial: The current optuna trial
    :param logdir: The directory to which store the logs
    :param variant: The variant of the domain randomization to run
    :return: The average return of the sourceUDR→target configuration
    """
    params = sample_sac_params(trial)
    lr = params['learning_rate']
    gamma = params['gamma']
    batch_size = params['batch_size']
    lr_schedule = LR_SCHEDULES[params['lr_schedule']]

    logger = configure(f"{logdir}/trial_{trial.number}", ["tensorboard"])
    metric = 0
    params_metric = {}
    env_source_UDR = gym.make(f"CustomHopper-UDR-source-v{variant.value}")
    env_source = gym.make(f"CustomHopper-source-v0")
    env_target = gym.make(f"CustomHopper-target-v0")

    model = SAC('MlpPolicy', env_source_UDR, learning_rate=lr_schedule(
        lr), batch_size=batch_size, gamma=gamma, verbose=1)
    model.set_logger(logger)
    model.learn(total_timesteps=int(1e5), progress_bar=True,
                tb_log_name=f"SAC_training_UDR")

    model.save(os.path.join("trained_models", f"step3_{variant.value}_trial_{trial.number}"))

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
    for k, v in params_metric.items():
        logger.record(k, v)
    logger.dump()

    return metric


def main(base_prefix='.', force=False, variant=None):
    """
    This function runs the ablation study through the optuna APIs.

    :param base_prefix: Specify the path to the directory where to save the results
    :param force: If it is true (from command line argument), overwrite previous existing logs
    :param variant: Determine which variant of domain randomization to run
    """
    variant_to_do = [variant] if variant is not None else list(VariantStep3)
    for variant in variant_to_do:
        print(f"Running variant {variant.name}...")
        logdir = f"{base_prefix}/sac_tb_step3_{variant.value}_log"

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

        objective = partial(objective_fn, logdir=logdir, variant=variant)
        study.optimize(objective)


main_interval = partial(main, variant=VariantStep3.INTERVAL)
main_relative = partial(main, variant=VariantStep3.RELATIVE)
main_absolute = partial(main, variant=VariantStep3.ABSOLUTE)
