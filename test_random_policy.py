"""Test a random policy on the Gymnasium Hopper environment

Play around with this code to get familiar with the
Hopper environment.

For example, what happens if you don't reset the environment
even after the episode is over?
When exactly is the episode over?
What is an action here?

Useful resources:
- https://gymnasium.farama.org/
- https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/hopper_v4.py
"""
from env_utils import *


def main():
    # set to 'human' for rendering (does not work on Google Colab)
    render_mode = 'human'

    env = make_env(domain="source", render_mode=render_mode)

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    # masses of each link of the Hopper
    print('Dynamics parameters:', env.get_parameters())

    n_episodes = 5

    for ep in range(n_episodes):
        print(f"Starting ep: {ep}")
        done = False
        state = env.reset()  # Reset environment to initial state

        while not done:  # Until the episode is over
            action = env.action_space.sample()  # Sample random action

            # Step the simulator to the next timestep
            state, reward, done, info = env.step(action)


if __name__ == '__main__':
    main()
