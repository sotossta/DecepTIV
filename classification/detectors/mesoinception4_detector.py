"""
Reference:
@inproceedings{afchar2018mesonet,
  title={Mesonet: a compact facial video forgery detection network},
  author={Afchar, Darius and Nozick, Vincent and Yamagishi, Junichi and Echizen, Isao},
  booktitle={2018 IEEE international workshop on information forensics and security (WIFS)},
  pages={1--7},
  year={2018},
  organization={IEEE}
}

Modified from github repo: https://github.com/HongguLiu/MesoNet-Pytorch
"""


import importlib
import torch.nn as nn
import torch

#MesoInception4Detector class definition
class MesoInception4_Detector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        #Initialize backbone
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        
    def build_backbone(self, config):
        # prepare the backbone
        backbone_name = config["backbone"]["name"]
        backbone_module = importlib.import_module(f".{backbone_name.lower()}", package="networks")
        backbone_class = getattr(backbone_module, f"{backbone_name}")
        backbone = backbone_class(config["backbone"])
        return backbone
    
    def build_loss(self, config):
        """
        Dynamically import and return the loss function based on the config.
        """
        loss_func_name = config['loss_func']
        module_name = f"losses.{loss_func_name.lower()}_loss"
        class_name = loss_func_name.capitalize() + "_Loss"
        module = importlib.import_module(module_name)
        loss_class = getattr(module, class_name)
        return loss_class()
    
    def features(self, x):
        return self.backbone.features(x)

    def classifier(self, features):
        return self.backbone.classifier(features)
    
    def get_losses(self, label, pred):
        loss = self.loss_func(pred, label)
        return loss
    

    def forward(self, x):
        # get the features by backbone
        features = self.features(x)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict