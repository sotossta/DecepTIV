"""
Modified from hugginface repo: https://huggingface.co/timm/swinv2_tiny_window8_256.ms_in1k
"""
import timm
import torch.nn as nn
import torch.nn.functional as F

class SwinT(nn.Module):

    def __init__(self, swint_config):
        super(SwinT,self).__init__()
        self.num_classes = swint_config['num_classes']
        self.net = timm.create_model('swinv2_tiny_window8_256.ms_in1k',pretrained=True,num_classes=0)  # remove classifier nn.Linear
        self.classifier = nn.Linear(self.net.num_features, self.num_classes)  # Two-class classification

    def features(self,input):

        x = self.net.forward_features(input)  # [B, H, W, C]
        x = x.mean(dim=(1, 2))  # Global average pool over H and W → [B, C]
        return x

    def forward(self, input):
        
        x = self.features(input)
        out = self.classifier(x)
        return out, x
