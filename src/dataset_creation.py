import csv
import gym
import os
from PIL import Image
from model.env.custom_hopper import *

def main():
    env = gym.make('CustomHopper-source-v0')
    base_prefix = os.path.join("..","dataset")

    with open(os.path.join(base_prefix, "data.csv"), 'w') as csvfile:
        dataset_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        n_images = 0
        for i in range(10_000):  
            
            action = env.action_space.sample()  # Sample random action

            state, _, done, _ = env.step(action)  # Step the simulator to the next timestep

            if (i+1 % 10 == 0): #save on csv
                
                img_state = env.render(mode="rgb_array", width=224, height=224)
                name_img = os.path.join(base_prefix, f"hopper-{n_images}.jpeg")
                Image.fromarray(img_state).save(name_img)
                
                dataset_writer.writerow([name_img, state])
                n_images += 1
            
            if done:
                state = env.reset()

if __name__ == "__main__":
    main()    

