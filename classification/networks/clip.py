"""
Modified from github repo:https://github.com/openai/CLIP/tree/main/clip
"""
import torch
from torch import nn
from clip_util import clip

class Clip(nn.Module):
    def __init__(self, clip_config):
        super(Clip, self).__init__()
        self.clip_model, preprocess = clip.load(clip_config['pretrained'])
        self.clip_model = self.clip_model.float()

    def features(self,input):

        b, t, _, h, w = input.shape
        images = input.view(b * t, 3, h, w)
        sequence_output = self.clip_model.encode_image(images)
        _, _, c = sequence_output.shape
        sequence_output = sequence_output.view(b, t, 14, 14, c)
        sequence_output = sequence_output.permute(0, 4, 1, 2, 3)
        return sequence_output


    def forward(self, input):
        
        return self.features(input)
