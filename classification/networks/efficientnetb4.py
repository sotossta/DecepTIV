"""
Modified from github repo: https://github.com/lukemelas/EfficientNet-PyTorch
"""

from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.nn.functional as F

class Efficientnetb4(nn.Module):
    def __init__(self, efficientnetb4_config):
        super(Efficientnetb4, self).__init__()
        self.num_classes = efficientnetb4_config['num_classes']

        self.net = EfficientNet.from_pretrained("efficientnet-b4", advprop=True)
        self.feature_extractor = self.net.extract_features
        self.classifier = nn.Linear(self.net._fc.in_features, self.num_classes)  # Two-class classification

    def features(self,input):

        x = self.feature_extractor(input)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        return x


    def forward(self, input):
        
        x = self.features(input)
        out = self.classifier(x)
        return out, x
