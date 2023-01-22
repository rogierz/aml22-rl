import csv
import gym
import os
from PIL import Image
from model.env.custom_hopper import *
from stable_baselines3 import SAC
from tqdm import trange, tqdm


def main():
    target_env = gym.make('CustomHopper-source-v0')
    base_prefix = os.path.join("dataset")
    if not os.path.exists(base_prefix):
        os.mkdir(base_prefix)

    for split in ["train", "val", "test"]:
        if not os.path.exists(os.path.join(base_prefix, split)):
            os.mkdir(os.path.join(base_prefix, split))

    if not os.path.exists(os.path.join("model", "data_model.zip")):
        train_env = gym.make('CustomHopper-UDR-source-v0')
        model = SAC("MlpPolicy", train_env, learning_rate=1e-3,
                    batch_size=128, gamma=0.99)
        model.learn(total_timesteps=100_000, progress_bar=True)
        model.save(os.path.join("model", "data_model"))
    else:
        model = SAC.load(os.path.join("model", "data_model.zip"))

    with open(os.path.join(base_prefix, "data.csv"), 'w') as csvfile:

        dataset_writer = csv.writer(
            csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        dataset_writer.writerow(["image", "split", "height", "angle-top", "angle-thigh", "angle-leg", "angle-foot", "velocity-top-x",
                                "velocity-top-z", "ang-velocity-top", "ang-velocity-thigh", "ang-velocity-leg", "ang-velocity-foot"])
        n_images = 0
        state = target_env.reset()

        for length, split, p in tqdm([(12_000, "train", 1), (3_000, "val", 0.9), (4_000, "test", 0.8)], desc=f"{split} - Split generation"):
            for _ in trange(length, desc="Sample generation"):

                action, _ = model.predict(state)
                if np.random.rand() > p:
                    action += np.random.normal(0, 0.1, action.shape)
                # Step the simulator to the next timestep
                state, _, done, _ = target_env.step(action)

                img_state = target_env.render(
                    mode="rgb_array", width=224, height=224)
                name_img = f"hopper-{n_images}.jpeg"
                Image.fromarray(img_state).save(
                    os.path.join(base_prefix, split, name_img))
                dataset_writer.writerow([name_img, split, *state])
                n_images += 1

                if done:
                    state = target_env.reset()


if __name__ == "__main__":
    main()
