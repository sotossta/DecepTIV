'''
Reference:
@inproceedings{liu2021spatial,
  title={Spatial-phase shallow learning: rethinking face forgery detection in frequency domain},
  author={Liu, Honggu and Li, Xiaodan and Zhou, Wenbo and Chen, Yuefeng and He, Yuan and Xue, Hui and Zhang, Weiming and Yu, Nenghai},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={772--781},
  year={2021}
}

Modified from github repo: https://github.com/SCLBD/DeepfakeBench/blob/main/training/detectors/spsl_detector.py

Notes:
To ensure consistency in the comparison with other detectors, we have opted not to utilize the shallow Xception architecture.
Instead, we are employing the original Xception model.

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib


class Spsl_Detector(nn.Module):
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
        state_dict = torch.load(config['pretrained'])
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
        conv1_data = state_dict['conv1.weight'].data
        backbone.load_state_dict(state_dict, False)
        
        # copy on conv1
        # let new conv1 use old param to balance the network
        backbone.conv1 = nn.Conv2d(4, 32, 3, 2, 0, bias=False)
        avg_conv1_data = conv1_data.mean(dim=1, keepdim=True)  # average across the RGB channels
        backbone.conv1.weight.data = avg_conv1_data.repeat(1, 4, 1, 1)  # repeat the averaged weights across the 4 new channels
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

    
    
    def features(self, x, phase_fea):
        features = torch.cat((x, phase_fea), dim=1)
        return self.backbone.features(features)

    def classifier(self, features):
        return self.backbone.classifier(features)
    
    def get_losses(self, label, pred):
        loss = self.loss_func(pred, label)
        return loss


    def forward(self, x):
        # get the phase features
        phase_fea = self.phase_without_amplitude(x)
        # bp
        features = self.features(x, phase_fea)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}

        return pred_dict

    def phase_without_amplitude(self, x):
        # Convert to grayscale
        gray_img = torch.mean(x, dim=1, keepdim=True) # shape: (batch_size, 1, 256, 256)
        # Compute the DFT of the input signal
        X = torch.fft.fftn(gray_img,dim=(-1,-2))
        #X = torch.fft.fftn(img)
        # Extract the phase information from the DFT
        phase_spectrum = torch.angle(X)
        # Create a new complex spectrum with the phase information and zero magnitude
        reconstructed_X = torch.exp(1j * phase_spectrum)
        # Use the IDFT to obtain the reconstructed signal
        reconstructed_x = torch.real(torch.fft.ifftn(reconstructed_X,dim=(-1,-2)))
        # reconstructed_x = torch.real(torch.fft.ifftn(reconstructed_X))
        return reconstructed_x