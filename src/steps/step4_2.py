import torch as th
from torch import nn
import gym
import numpy as np
from gym import spaces
from gym.spaces import Box
from gym.wrappers.pixel_observation import PixelObservationWrapper
from gym.wrappers.resize_observation import ResizeObservation
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import SAC
from torchvision.models import resnet18
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from datetime import datetime
import os
import shutil

class CustomWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(0, 255, shape=(224, 224, 3), dtype=np.uint8)

    def observation(self, obs):
        return obs["pixels"]

class LSTM(BaseFeaturesExtractor):
    
    def __init__(self,  observation_space: spaces.Box, features_dim: int = 128, embed_dim = 128, hidden_size = 128, num_layers = 1):
        super().__init__(observation_space, features_dim)
        self.DEVICE = "cuda"

        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ------ Backbone
        self.backbone = resnet18(weights='DEFAULT') #models.ssdlite320_mobilenet_v3_large(weights="DEFAULT") 
        # stem adjustment
        self.backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        # only feature maps
        self.backbone.fc = nn.Identity()
        
        self.proj_embedding = nn.Sequential( # Projection head
        nn.Linear(512, 128),
        nn.ReLU()
        )
        # ------ LSTM
        self.lstm_cell = nn.LSTM(128, 128, num_layers=num_layers, batch_first=True)
        self.backbone.train(False)
        self.lstm_cell.train(True)

    def forward(self, x):      
        # print("\n INPUT TO THE NET: ", x.shape)
        x = x.permute(0, 1, 4, 2, 3)
        # print("\n AFTER RESHAPING: ", x.shape)
        batch_size, img_size = x.shape[0], x.shape[2:]
        x = x.reshape(-1, *img_size) # i merge the batch_size and num_seq in order to feed everything to the cnn

        # print("\n INPUT TO THE BACKBONE SHAPE: ", x.shape)
        x = self.backbone(x)
        # print("\n OUTPUT OF BACKBONE: ", x.shape)

        x = self.proj_embedding(x)
        # print("OUTPUT OF PROJECTION LAYER: ", x.shape)

        x = x.reshape(batch_size, -1, self.embed_dim) # then i comeback the original shape
        # print("RESHAPING.. INPUT TO LSTM: ", x.shape)  
        # lstm part
        h_0 = th.autograd.Variable(th.randn(self.num_layers, x.size(0), self.hidden_size)).to(self.DEVICE)
        c_0 = th.autograd.Variable(th.randn(self.num_layers, x.size(0), self.hidden_size)).to(self.DEVICE)
        y, (hn, cn) = self.lstm_cell(x, (h_0, c_0))
        # print("\n LSTM OUTPUT SHAPE: ", y.shape)          
        y = y[:, -1, :]

        # print("\n FINAL OUTPUT SHAPE: ", y.shape)
        return y

#ResNet18 with some adjustments 
class ResNet(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512, n_frames = 4):
        super().__init__(observation_space, features_dim)

        self.backbone = resnet18()#weights='IMAGENET1K_V1')
        # stem adjustment 
        self.backbone.conv1 = nn.Conv2d(n_frames, 64, 3, 1, 1, bias=False)
        self.backbone.maxpool = nn.Identity()
        # only feature maps
        self.backbone.fc = nn.Identity()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.backbone(observations)

def main(base_prefix=".", force=False):

    logdir = f"{base_prefix}/sac_tb_step4_2_log"

    # policy_kwargs = dict(features_extractor_class = ResNet)

    sac_params = {
            "learning_rate": 2e-3,
            "gamma": 0.99,
            "batch_size": 128
    }

    if os.path.isdir("logdir"):
        if force:
            try:
                shutil.rmtree(logdir)
            except Exception as e:
                print(e)
        else:
            print(f"Directory {logdir} already exists. Shutting down...")
            return
    
    # env = gym.make(f"CustomHopper-UDR-source-v0")
    # env = ResizeObservation(CustomWrapper(
    #           PixelObservationWrapper(gym.make(f"CustomHopper-source-v0"))), shape=(128, 128))

    # env = FrameStack(env, 4)

    # env_source = ResizeObservation(CustomWrapper(
    #             PixelObservationWrapper(gym.make(f"CustomHopper-source-v0"))), shape=(128, 128))
    # env_target =  ResizeObservation(CustomWrapper(
    #             PixelObservationWrapper(gym.make(f"CustomHopper-target-v0"))), shape=(128, 128))

    # env_source = FrameStack(env_source, 4)
    # env_target = FrameStack(env_target, 4)
    #env = GrayScaleObservation(env, keep_dim=True)
    #print("\n OBSERVATION SPACE GRAYSCALE:", env.observation_space.shape)
 
    env = VecFrameStack(DummyVecEnv([lambda: GrayScaleObservation(ResizeObservation(CustomWrapper(
            PixelObservationWrapper(gym.make(f"CustomHopper-UDR-source-v0"))), shape=(128, 128)), keep_dim=True)]), 4, "last")
    
    env_source = VecFrameStack(DummyVecEnv([lambda: GrayScaleObservation(ResizeObservation(CustomWrapper(
            PixelObservationWrapper(gym.make(f"CustomHopper-source-v0"))), shape=(128, 128)), keep_dim=True)]), 4, "last")
    
    env_target = VecFrameStack(DummyVecEnv([lambda: GrayScaleObservation(ResizeObservation(CustomWrapper(
            PixelObservationWrapper(gym.make(f"CustomHopper-target-v0"))), shape=(128, 128)), keep_dim=True)]), 4, "last")
    
    logger = configure(logdir, ["stdout", "tensorboard"])

    model = SAC("CnnPolicy", env, **sac_params, seed=42, buffer_size=10000) #for resNet: add policy_kwargs=policy_kwargs as parameter
    
    model.learn(total_timesteps=250000, progress_bar=True)

    if os.path.isfile(os.path.join("trained_models", "step4_2.zip")):
        fname = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(fname)
        model.save(os.path.join("trained_models", f"step4_2_{fname}"))
    else:
        model.save(os.path.join("trained_models", "step4_2"))

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

            logger.record(f'episode_return/{env_name}', episode_return)
            logger.dump(ep)
            run_avg_return += episode_return
        run_avg_return /= n_episodes
        logger.record(f'run_avg_return/{env_name}', run_avg_return)
        logger.dump()
