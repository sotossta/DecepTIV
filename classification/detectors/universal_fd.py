"""
Reference:
@inproceedings{ojha2023towards,
  title={Towards universal fake image detectors that generalize across generative models},
  author={Ojha, Utkarsh and Li, Yuheng and Lee, Yong Jae},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={24480--24489},
  year={2023}
}
"""

import torch
import torch.nn as nn
import importlib

class Universal_FD_Detector(nn.Module):
     
    def __init__(
        self, config, channel_size=512, class_num=2
    ):
        super(Universal_FD_Detector, self).__init__()
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.num_classes = config["backbone"]["num_classes"]
        # Freeze backbone weights
        for param in self.backbone.parameters():
            param.requires_grad = False 
        self.fc = nn.Linear(channel_size, self.num_classes)

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
    
    def get_losses(self, label, pred):
        loss = self.loss_func(pred, label)
        return loss

    def build_backbone(self,config):
        # prepare the backbone
        backbone_name = config["backbone"]["name"]
        backbone_module = importlib.import_module(f".{backbone_name.lower()}", package="networks")
        backbone_class = getattr(backbone_module, f"{backbone_name}")
        backbone = backbone_class(config["backbone"])
        return backbone

    def forward(self, x):

        features = self.backbone(x)  
        cls = self.fc(features)  # raw logits
        # Probability score for binary classification
        prob = torch.softmax(cls, dim=1)[:, 1]
        # Build prediction dictionary
        pred_dict = {
                'cls': cls,
                'prob': prob,
                'feat': features}
        return pred_dict
      