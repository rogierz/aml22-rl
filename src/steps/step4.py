"""
VISION-BASED REINFORCEMENT LEARNING 
Implement an RL training pipeline that uses raw images as state observations to the agent.
In this final step, you're given more flexibility on how you can implement the pipeline,
leaving room for variants and group's own implementations.
"""

from gym.spaces import Box
from stable_baselines3 import SAC
from model.env.custom_hopper import *
from gym.wrappers.pixel_observation import PixelObservationWrapper
from gym.wrappers.resize_observation import ResizeObservation


class CustomWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)

    def observation(self, obs):
        # obs = obs["pixels"]
        # obs = cv2.resize(obs, (256, 256), interpolation=cv2)
        return obs["pixels"]


def main(base_prefix="."):
    sac_params = {
        "learning_rate": 2e-3,
        "gamma": 0.99,
        "batch_size": 128
    }

    total_timesteps = int(1e5)

    env = gym.make(f"CustomHopper-UDR-source-v0")
    env_source = gym.make(f"CustomHopper-source-v0")
    env_target = gym.make(f"CustomHopper-target-v0")
    env = ResizeObservation(CustomWrapper(
        PixelObservationWrapper(env)), shape=(128, 128))
    obs = env.reset()
    print(obs.shape)

    model = SAC('CnnPolicy', env, **sac_params, seed=42, buffer_size=100000)

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save("cnn_model")

    n_episodes = 5_000

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
                test_env.render()

            run_avg_return += episode_return
        run_avg_return /= n_episodes


if __name__ == '__main__':
    main()
