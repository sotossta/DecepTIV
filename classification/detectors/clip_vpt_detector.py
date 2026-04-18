"""
Reference:
@inproceedings{jia2022visual,
  title={Visual prompt tuning},
  author={Jia, Menglin and Tang, Luming and Chen, Bor-Chun and Cardie, Claire and Belongie, Serge and Hariharan, Bharath and Lim, Ser-Nam},
  booktitle={European conference on computer vision},
  pages={709--727},
  year={2022},
  organization={Springer}
}
"""

import torch
import torch.nn as nn
import importlib
from clip_util import clip

class Clip_VPT_Detector(nn.Module):
     
    def __init__(
        self, config, channel_size=512, class_num=2
    ):
        super(Clip_VPT_Detector, self).__init__()
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.config = config
        self.visual = self.backbone.clip_model.visual  # extract CLIP visual encode
        self.m = config["backbone"]["prompt_length"]
        self.d = self.visual.conv1.out_channels
        self.L = len(self.visual.transformer.resblocks) 
        self.num_classes = config["backbone"]["num_classes"]
        # Freeze backbone weights
        for param in self.backbone.parameters():
            param.requires_grad = False 
        
        self.vpt_type = config["model"]["type"]
        #Define learnable visual prompts
        if self.vpt_type == "shallow":
            self.prompts = nn.Parameter(torch.zeros(self.m,  self.d)) # (m, dim)
            nn.init.trunc_normal_(self.prompts, std=0.02)
        elif self.vpt_type == "deep":
            self.prompts = nn.Parameter(torch.zeros(self.L, self.m, self.d)) #(L, m, dim)
            nn.init.trunc_normal_(self.prompts, std=0.02)
        else:
            raise ValueError("VPT type must be 'shallow' or 'deep'")

        self.fc = nn.Linear(self.d,self.num_classes)

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
    
    def get_losses(self, label, pred):
        loss = self.loss_func(pred, label)
        return loss


    def add_prompts(self, x, P):
        # P: (B,m,d)
        P = P.to(x.device)
        P = P.permute(1,0,2)  # (m,B,d)
        x = torch.cat([x[:1,:,:], P, x[1:,:,:]], dim=0) # prepend after CLS
        return x

    

    def forward(self,x):

        # x -> (B,C,H,W)
        B = x.shape[0]
        visual = self.visual #extract only CLIP's ViT

        # Patchify: conv projection

        # (B, C, H, W) -> (B, dim, grid, grid)
        x = visual.conv1(x)
  
        # (B, dim, grid, grid) -> (B, dim, N) grid**2 =N 
        x = x.reshape(B, x.shape[1], -1)  
        # (B, dim, N) -> (B, N, dim)
        x = x.permute(0, 2, 1)     

        #CLS token
        cls_tokens = visual.class_embedding.to(x.dtype) + torch.zeros(
            B, 1, self.d, dtype=x.dtype, device=x.device
        ) 

        # (B, N, dim) -> (B, N+1, dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, d)

        # Positional embeddings
        pos_embed = visual.positional_embedding.to(x.dtype)
        x = x + pos_embed
       
        x = visual.ln_pre(x)

        # (B, 1+N, dim ) -> (1+N, B, dim)
        x = x.permute(1, 0, 2) 
        
        # Add learnable visual prompts
        if self.vpt_type == "shallow":
            P = self.prompts.unsqueeze(0).expand(B, -1, -1)  # (B, m, dim)
            
            # (1+N, B, dim) -> (1+m+N, B, dim)
            x = self.add_prompts(x, P)

            # run first block with prompts

            #  (1+m+N, B, dim) ->  (1+m+N, B, dim)
            x = visual.transformer.resblocks[0](x)
            # drop prompts

            # (1+m+N, B, dim) -> (1+N, B, dim)
            x = torch.cat([x[:1, :, :], x[1+self.m:, :, :]], dim=0)
            # rest blocks
            for blk in visual.transformer.resblocks[1:]:
                x = blk(x)
      
        elif self.vpt_type == "deep":
            for l, blk in enumerate(visual.transformer.resblocks):

                #Add new learnable prompts at every transformer layer
                P = self.prompts[l].unsqueeze(0).expand(B, -1, -1)  # (B, m, dim)
            
                # (1+N, B, dim) -> (1+m+N, B, dim)
                x = self.add_prompts(x, P)

                #  (1+m+N, B, dim) ->  (1+m+N, B, dim)
                x = blk(x)
                # drop prompts

                # (1+m+N, B, dim) -> (1+N, B, dim)
                x = torch.cat([x[:1, :, :], x[1+self.m:, :, :]], dim=0)
                

        # (1+N, B, dim) - > (B, 1+N, dim)
        x = x.permute(1, 0, 2)  # LND -> NLD
        #Extract only CLS token

        #(B, 1+N, dim) -> (B,dim)
        x = visual.ln_post(x[:, 0, :])

        #Classification Layer
       
        #(B,dim) -> (B,2)
        pred = self.fc(x)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': x}
        return pred_dict

