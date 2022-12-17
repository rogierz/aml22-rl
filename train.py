"""Sample script for training a control policy on the Hopper environment

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between TRPO, PPO, and SAC.
"""
from env_utils import *

def main():
    render_mode = None  # you do not want to render at training time

    env = make_env(domain="source", render_mode=render_mode)

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper

    """
        TODO:

            - train a policy
            - test the policy
    """

if __name__ == '__main__':
    main()