"""
Reference:
@article{chen2024demamba,
  title={Demamba: Ai-generated video detection on million-scale genvideo benchmark},
  author={Chen, Haoxing and Hong, Yan and Huang, Zizheng and Xu, Zhuoer and Gu, Zhangxuan and Li, Yaohui and Lan, Jun and Zhu, Huijia and Zhang, Jianfu and Wang, Weiqiang and others},
  journal={arXiv preprint arXiv:2405.19707},
  year={2024}
}

Modified from github repo: https://github.com/chenhaoxing/DeMamba/blob/main/models/DeMamba.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
#from mamba_base import MambaConfig, ResidualBlock
import torch.nn.init as init
#from clip import clip
import math
import importlib



class DeMamba_Detector(nn.Module):
    def __init__(
        self, config, channel_size=512, class_num=1
    ):
        super(DeMamba_Detector, self).__init__()
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        blocks = []
        self.fusing_ratios = 2
        self.patch_nums = (14//self.fusing_ratios)**2
        self.mamba = self.build_mamba(config)
        channel = config['base']['channel']
        self.num_classes = config["backbone"]["num_classes"]
        self.fc1 = nn.Linear(channel*(self.patch_nums+1), self.num_classes)
        self.bn1 = nn.BatchNorm1d(channel)
        self.initialize_weights(self.fc1)

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

    
    def build_mamba(self, config):
        # prepare mamba base
        name = config["base"]["name"]
        module = importlib.import_module(f".{name.lower()}", package="networks")
        mamba_config_class = getattr(module, f"{config['base']['config']}")
        mamba_config = mamba_config_class(d_model = config['base']['channel'])
        block = getattr(module, f"{config['base']['block']}")
        mamba_block = block(mamba_config)        
        return mamba_block
    
    def build_backbone(self,config):
        # prepare the backbone
        backbone_name = config["backbone"]["name"]
        backbone_module = importlib.import_module(f".{backbone_name.lower()}", package="networks")
        backbone_class = getattr(backbone_module, f"{backbone_name}")
        backbone = backbone_class(config["backbone"])
        return backbone
       
    def initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        b, t, _, h, w = x.shape
        sequence_output = self.backbone(x)
        _, _, c = sequence_output.shape
        sequence_output = sequence_output.view(b, t, -1, c)
        global_feat = sequence_output.reshape(b, -1, c)
        global_feat = global_feat.mean(1)

        _, _, f_w, _ = sequence_output.shape
        f_h, f_w = int(math.sqrt(f_w)), int(math.sqrt(f_w))

        s = f_h//self.fusing_ratios
        sequence_output = sequence_output.view(b, t, self.fusing_ratios, s, self.fusing_ratios, s, c)
        x = sequence_output.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(b*s*s, t, -1, c)
        b_l = b*s*s
        
        x = reorder_data(x, self.fusing_ratios)
        x = x.permute(0, 2, 1, 3).contiguous().view(b_l, -1, c)
        res = self.mamba(x)
        video_level_features = res.mean(1)
        video_level_features = video_level_features.view(b, -1)

        video_level_features = torch.cat((global_feat, video_level_features), dim=1)
        cls = self.fc1(video_level_features)  # raw logits

        # Probability score for binary classification
        prob = torch.softmax(cls, dim=1)[:, 1]

        # Build prediction dictionary
        pred_dict = {
            'cls': cls,
            'prob': prob,
            'feat': video_level_features
        }

        return pred_dict

def create_reorder_index(N, device):
    new_order = []
    for col in range(N):
        if col % 2 == 0:
            new_order.extend(range(col, N*N, N))
        else:
            new_order.extend(range(col + N*(N-1), col-1, -N))
    return torch.tensor(new_order, device=device)

def reorder_data(data, N):
    assert isinstance(data, torch.Tensor), "data should be a torch.Tensor"
    device = data.device
    new_order = create_reorder_index(N, device)
    B, t, _, _ = data.shape
    index = new_order.repeat(B, t, 1).unsqueeze(-1)
    reordered_data = torch.gather(data, 2, index.expand_as(data))
    return reordered_data