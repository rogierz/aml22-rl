import csv
import gym
import os
from PIL import Image
from model.env.custom_hopper import *
from stable_baselines3 import SAC

def main():
    env = gym.make('CustomHopper-source-v0')
    base_prefix = os.path.join("dataset")
    
    if not os.path.exists(os.path.join("model", "data_model.zip")):
        model = SAC("MlpPolicy", env)
        model.learn(total_timesteps = 10_000)
        model.save(os.path.join("model", "data_model"))
    else:
        model = SAC.load(os.path.join("model", "data_model.zip"))
    
    
    with open(os.path.join(base_prefix, "data.csv"), 'w') as csvfile:
        
        dataset_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        dataset_writer.writerow(["image", "height", "angle-top", "angle-thigh", "angle-leg", "angle-foot", "velocity-top-x", "velocity-top-z", "ang-velocity-top", "ang-velocity-thigh", "ang-velocity-leg", "ang-velocity-foot"])
        n_images = 0
        state = env.reset()
        for i in range(10_000):  
            
            action, _ = model.predict(state, deterministic=True)  # Sample random action
            state, _, done, _ = env.step(action)  # Step the simulator to the next timestep

            if ((i+1) % 10 == 0): #save on csv
                
                img_state = env.render(mode="rgb_array", width=224, height=224)
                name_img = os.path.join(base_prefix, f"hopper-{n_images}.jpeg")
                Image.fromarray(img_state).save(name_img)
                dataset_writer.writerow([name_img, *state])
                n_images += 1
            
            if done:
                state = env.reset()

if __name__ == "__main__":
    main()    
