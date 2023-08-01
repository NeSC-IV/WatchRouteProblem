import torch
import timm
from torchsummary import summary
from eca_res import eca_resnet18
from wrpsolver.bc.cunstomCnn import ResNet,ResNet18,BasicBlock
features_dim = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('ghostnet_050', pretrained=False).to(device)
# model = eca_resnet18().to(device)
summary(model,input_size=(3,224,224))