import math
import torch as th
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from collections import OrderedDict
from typing import Dict, List, Tuple, Type, Union
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out + x

class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class ImpalaModel(nn.Module):
    def __init__(self,in_channels, out_channels=256):
        super(ImpalaModel, self).__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16)
        self.block2 = ImpalaBlock(in_channels=16, out_channels=32)
        # self.block3 = ImpalaBlock(in_channels=32, out_channels=64)
        # self.block4 = ImpalaBlock(in_channels=64, out_channels=128)
        self.avgpool = nn.Sequential(nn.AvgPool2d(2))
        
        self.fc = nn.Linear(in_features=8192, out_features=out_channels)#32*8*8
        self.flatten = nn.Flatten()

        self.output_dim = 256

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        # x = self.block3(x)
        # x = self.block4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    block = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    return block
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        m = OrderedDict()
        m['conv1'] = conv3x3(inplanes, planes, stride)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(planes, planes)
        m['bn2'] = nn.BatchNorm2d(planes)
        self.group1 = nn.Sequential(m)
        self.relu= nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual
        out = self.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()

        m = OrderedDict()
        m['conv1'] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m['bn1'] = nn.BatchNorm2d(64)
        m['relu1'] = nn.ReLU(inplace=True)
        m['maxpool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group1= nn.Sequential(m)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.Sequential(nn.AvgPool2d(3))

        self.group2 = nn.Sequential(
            OrderedDict([
                ('fc', nn.Linear(512 * block.expansion, num_classes))
            ])
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.group1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.group2(x)

        return x
    
class IMPALA(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512,normalized_image: bool = False):
        super().__init__(observation_space, features_dim)
        self.model = ImpalaModel(1,features_dim)

    def forward(self,x):
        return self.model.forward(x)
    
class ResNet18(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512,normalized_image: bool = False):
        super().__init__(observation_space, features_dim)
        self.model = ResNet(BasicBlock, [1, 1, 1, 1],features_dim)

    def forward(self,x):
        return self.model.forward(x)

class Flatten1d(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 10):
        super().__init__(observation_space, features_dim)
        self.flatten = nn.Flatten()
        self.avgpool = nn.AdaptiveAvgPool1d(features_dim)
    def forward(self,x):
        x = self.flatten(x)
        x = self.avgpool(x)
        return  x

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        # cnn_output_dim: int = 45,#114
        cnn_output_dim: int = 95,#114
        normalized_image: bool = False,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = ResNet18(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)
    
def is_image_space(
    observation_space: spaces.Space,
    check_channels: bool = False,
    normalized_image: bool = False,
) -> bool:
    check_dtype = check_bounds = not normalized_image
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        # Check the type
        if check_dtype and observation_space.dtype != np.uint8:
            return False

        # Check the value range
        incorrect_bounds = np.any(observation_space.low != 0) or np.any(observation_space.high != 255)
        if check_bounds and incorrect_bounds:
            return False

        # Skip channels check
        if not check_channels:
            return True
        # Check the number of channels
        if is_image_space_channels_first(observation_space):
            n_channels = observation_space.shape[0]
        else:
            n_channels = observation_space.shape[-1]
        # GrayScale, RGB, RGBD
        return n_channels in [1, 3, 4]
    return False

def is_image_space_channels_first(observation_space: spaces.Box) -> bool:
    smallest_dimension = np.argmin(observation_space.shape).item()
    if smallest_dimension == 1:
        print("Treating image space as channels-last, while second dimension was smallest of the three.")
    return smallest_dimension == 0

def get_flattened_obs_dim(observation_space: spaces.Space) -> int:
    if isinstance(observation_space, spaces.MultiDiscrete):
        return sum(observation_space.nvec)
    else:
        return spaces.utils.flatdim(observation_space)