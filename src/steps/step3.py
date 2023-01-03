"""
Train two agents with your algorithm of choice, on the source and target domain Hoppers respectively.
Then, test each model and report its average reward over 50 test episodes.
In particular, report results for the following “training→test” configurations: source→source, source→target (lower bound), target→target (upper bound).
"""
from functools import partial
import os
import optuna
import gym
from model.env.custom_hopper import *
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import SAC
import shutil


def sample_sac_params(trial, n=None):
    GAMMA_MIN = 0.01
    GAMMA_MAX = 0.99
    gamma_l = GAMMA_MIN
    gamma_r = GAMMA_MAX

    if n > 0:
        gamma_l = GAMMA_MIN + (GAMMA_MAX - GAMMA_MIN) * (n-1)/3
        gamma_r = GAMMA_MIN + (GAMMA_MAX - GAMMA_MIN) * (n)/3

    gamma = trial.suggest_float("gamma", gamma_l, gamma_r)
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size_exp = trial.suggest_int("batch_size_exp", 7, 10)
    batch_size = 2**batch_size_exp
    total_timesteps = trial.suggest_int("total_timesteps", 1e4, 1e5, log=True)

    return {
        "learning_rate": lr,
        "gamma": gamma,
        "batch_size": batch_size,
        "total_timesteps": total_timesteps
    }


def objective_fn(trial, logdir='.', n=None):
    params = sample_sac_params(trial, n=n)
    sac_params = {k: v for k, v in params if k != "total_timesteps"}

    src_trg_avg_return = 0
    params_metric = {}
    with SummaryWriter(log_dir=f"{logdir}/run_trial_{trial.number}") as writer:

        for source, target in [('source', 'source'), ('source', 'target'), ('target', 'target')]:
            env_source = gym.make(f"CustomHopper-{source}-v0")
            env_target = gym.make(f"CustomHopper-{target}-v0")
            model = SAC('MlpPolicy', env_source, **sac_params,
                        tensorboard_log=f"{logdir}/run_trial_{trial.number}")

            model.learn(total_timesteps=params["total_timesteps"], progress_bar=True,
                        tb_log_name=f"SAC_{source}_{target}")

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


def main(params, base_prefix='.'):
    logdir = f"{base_prefix}/sac_tb_step3_log"

    try:
        shutil.rmtree(logdir)
    except Exception as e:
        print(e)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(), direction="maximize", study_name="Our awesome study")

    objective = partial(objective_fn, logdir=logdir, n=params.n)
    study.optimize(objective, n_trials=10)


if __name__ == "__main__":
    main({})
