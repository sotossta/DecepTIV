"""
Reference:
@inproceedings{liu2021swin,
  title={Swin transformer: Hierarchical vision transformer using shifted windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={10012--10022},
  year={2021}
}

Modified from hugginface repo: https://huggingface.co/timm/swinv2_tiny_window8_256.ms_in1k
"""


import torch.nn as nn
import importlib
import torch
# SwinTDetector class definition
class SwinT_Detector(nn.Module):

    def __init__(self, config):
        super(SwinT_Detector, self).__init__()
        # Initialize EfficientNet model
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
        


    

