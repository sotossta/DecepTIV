"""
Modified from github repo:https://github.com/openai/CLIP/tree/main/clip
"""
import torch
from torch import nn
from clip_util import clip

class Clip(nn.Module):
    def __init__(self, clip_config):
        super(Clip, self).__init__()
        self.clip_model, preprocess = clip.load(clip_config['pretrained'], clip_size = clip_config['clip_size'])
        self.clip_model = self.clip_model.float()
        self.clip_size = clip_config['clip_size']

    def features(self,input):

        if self.clip_size ==1:
            #For CLIP_AD
            sequence_output = self.clip_model.encode_image(input)
            return sequence_output
        else:
            b, t, _, h, w = input.shape
            images = input.view(b * t, 3, h, w)
            sequence_output = self.clip_model.encode_image(images)
            return sequence_output


    def forward(self, input):
        
        return self.features(input)
