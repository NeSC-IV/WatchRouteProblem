import timm
import torch.nn as nn
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# transform = timm.data.create_transform(( 1 , 200 , 200 ))
class EfficientnetB0(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        self.model = timm.create_model('efficientnet_b0', pretrained=False,num_classes=features_dim,in_chans=1)
        self.model = nn.DataParallel(self.model)

    def forward(self,x):
        return self.model.forward(x)
    
class EfficientnetB4(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        self.model = timm.create_model('tf_efficientnet_b4', pretrained=False,num_classes=features_dim,in_chans=1)
        self.model = nn.DataParallel(self.model)

    def forward(self,x):
        return self.model.forward(x)
class FbNetv3(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        self.model = timm.create_model('fbnetv3_b', pretrained=False,num_classes=features_dim,in_chans=1)
        self.model = nn.DataParallel(self.model)

    def forward(self,x):
        return self.model.forward(x)
    
class MobileNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 1024):
        super().__init__(observation_space, features_dim)
        self.model = timm.create_model('tf_mobilenetv3_small_100', pretrained=False,num_classes=0, in_chans=1)
        self.model = nn.DataParallel(self.model)

    def forward(self,x):
        return self.model.forward(x)
    
class MobileVit(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        self.model = timm.create_model('mobilevitv2_050', pretrained=False,num_classes=features_dim,in_chans=1)
        self.model = nn.DataParallel(self.model)

    def forward(self,x):
        return self.model.forward(x)
        
class XCIT(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        self.model = timm.create_model('xcit_tiny_24_p8_384_dist', pretrained=False,num_classes=features_dim,in_chans=1)
        self.model = nn.DataParallel(self.model)

    def forward(self,x):
        return self.model.forward(x)

class Tinynet(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        self.model = timm.create_model('tinynet_c', pretrained=False,num_classes=features_dim,in_chans=1)
        self.model = nn.DataParallel(self.model)

    def forward(self,x):
        return self.model.forward(x)
    
class NatureCNNBlock(nn.Module):

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
    ) -> None:
        super(NatureCNNBlock, self).__init__()
        

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
    
class NatureCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        self.model = NatureCNNBlock(observation_space, features_dim)
        self.model = nn.DataParallel(self.model)

    def forward(self,x):
        return self.model.forward(x)