import torch as th
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torchvision.models import resnet18


class ResNet(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 32, n_frames=4, pre_train=False):
        super().__init__(observation_space, features_dim)

        # weights='IMAGENET1K_V1')
        self.backbone = resnet18(
            weights='IMAGENET1K_V1') if pre_train else resnet18()
        # stem adjustment
        self.backbone.conv1 = nn.Conv2d(n_frames, 64, 3, 1, 1, bias=False)
        # only feature maps
        self.backbone.fc = nn.Identity()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.backbone(observations)
