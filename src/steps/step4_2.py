import torch as th
from torch import nn
import gym
import numpy as np
from gym import spaces
from gym.spaces import Box
from gym.wrappers.pixel_observation import PixelObservationWrapper
from gym.wrappers.resize_observation import ResizeObservation
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import SAC
from torchvision.models import resnet18

class CustomWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(0, 255, shape=(224, 224, 3), dtype=np.uint8)

    def observation(self, obs):
        return obs["pixels"]
        
#ResNet18 with some adjustments 
class ResNet(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        self.backbone = resnet18()#weights='IMAGENET1K_V1')
        # stem adjustment
        self.backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.backbone.maxpool = nn.Identity()
        # only feature maps
        self.backbone.fc = nn.Identity()

        self.proj_head = nn.Sequential( # Projection head
        nn.Linear(512, 256),
        nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        print("INPUT SHAPE: ", observations.shape)
        x = self.backbone(observations)
        x = self.proj_head(x)
        print("OUTPUT SHAPE: ", x.shape)
        return x

policy_kwargs = dict(features_extractor_class=ResNet)

sac_params = {
        "learning_rate": 2e-3,
        "gamma": 0.99,
        "batch_size": 128
    }


def main(base_prefix=".", force=False):
    env = gym.make(f"CustomHopper-UDR-source-v0")
    env = ResizeObservation(CustomWrapper(
                PixelObservationWrapper(gym.make(f"CustomHopper-source-v0"))), shape=(224, 224))

    model = SAC("CnnPolicy", env, **sac_params, policy_kwargs=policy_kwargs, seed=42, buffer_size=10000)
    
    model.learn(total_timesteps=1000, progress_bar=True)
