import timm
import torch.nn as nn
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# transform = timm.data.create_transform(( 1 , 200 , 200 ))
class MobileNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 8):
        super().__init__(observation_space, features_dim)
        self.model = timm.create_model('tf_mobilenetv3_small_minimal_100', pretrained=False,num_classes=8, in_chans=2)
        self.model = nn.DataParallel(self.model)

    def forward(self,x):
        return self.model.forward(x)
    