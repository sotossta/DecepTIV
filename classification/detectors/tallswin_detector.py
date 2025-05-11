"""
Reference:
@inproceedings{xu2023tall,
  title={Tall: Thumbnail layout for deepfake video detection},
  author={Xu, Yuting and Liang, Jian and Jia, Gengyun and Yang, Ziming and Zhang, Yanhao and He, Ran},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={22658--22668},
  year={2023}
}

Modified from github repo: https://github.com/rainy-xu/TALL4Deepfake/tree/main
"""

import torch.nn as nn
import importlib
import torch
# XceptionDetector class definition
class TallSwin_Detector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        # prepare the backbone
        backbone_name = config["backbone"]["name"]
        backbone_module = importlib.import_module(f".{backbone_name.lower()}", package="networks")
        backbone_class = getattr(backbone_module, f"{backbone_name}")
        backbone = backbone_class(config["backbone"])
        #Load pre-trained weights
        if config['pretrained']==False:
            return backbone
        else:
            state_dict = torch.load(config['pretrained'])
            for name, weights in state_dict.items():
                if 'pointwise' in name:
                    state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
            state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
            backbone.load_state_dict(state_dict, False)
            
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

    def features(self,x):
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