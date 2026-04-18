"""
Reference:
@inproceedings{kundu2025towards,
  title={Towards a universal synthetic video detector: From face or background manipulations to fully ai-generated content},
  author={Kundu, Rohit and Xiong, Hao and Mohanty, Vishal and Balachandran, Athula and Roy-Chowdhury, Amit K},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={28050--28060},
  year={2025}
}
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
import importlib


class UNITE_Detector(nn.Module):
    def __init__(self,config):
        super(UNITE_Detector, self).__init__()

        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.num_classes = config["backbone"]["num_classes"]
        depth = config["model"]["depth"]
        drop_rate = config["model"]["dropout1"]
        attn_drop_rate = config["model"]["dropout2"]
        dropout = config["model"]["dropout3"]
        # Freeze backbone weights
        
        for param in self.backbone.parameters():
            param.requires_grad = False 
        embed_dim = 512
        self.vid_transformer = VideoTransformer(num_patches = 196 , num_classes=2, embed_dim=embed_dim, depth=depth,
                 num_heads=16, mlp_ratio=4., drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                   norm_layer=nn.LayerNorm, num_frames=8, dropout=dropout)
        self.fc = nn.Linear(embed_dim,self.num_classes)

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

    def forward(self,x):
        
        sequence_output = self.backbone(x)
        video_level_features = self.vid_transformer.forward(sequence_output)
        cls = self.fc(video_level_features)
        # Probability score for binary classification
        prob = torch.softmax(cls, dim=1)[:, 1]

        # Build prediction dictionary
        pred_dict = {
            'cls': cls,
            'prob': prob,
            'feat': video_level_features
        }

        return pred_dict

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4, drop = 0., attn_drop=0.,
                 drop_path = 0.1, act_layer = nn.GELU, norm_layer = nn.LayerNorm):
        super().__init__()
        

        self.norm_attn = norm_layer(dim)
        self.attn = Attention(dim = dim, num_heads = num_heads, attn_drop= attn_drop, proj_drop= drop)
        self.fc = nn.Linear(dim, dim)

        self.norm_mlp = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                    act_layer=act_layer, drop=drop)
        

    def forward(self, x, B, T):
        # B: batch_size, T: frames, N: patches per frame
        # x: (B, T*N + 1, dim)

        res = self.attn(self.norm_attn(x))
        x = x + res
        x = x + self.mlp(self.norm_mlp(x))
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop =0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, dim * 3, bias = False)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self,x):
        #(B, N*T +1,dim)  set N*T+1=L
        B, N, C = x.shape

        #(B, L,dim) -> (B, L, 3*dim )
        qkv = self.qkv(x) 
        #(B, L, 3*dim) -> (B, L, 3, NH, HD)    NH = number of heads, HD = head dimension
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        
        # (B, L, 3, NH, HD) -> (3, B, NH, L, HD)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, B, NH, L, HD) -> 3 arrays of (B, NH, L, HD)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # (B, NH, L, L)
        weight = (q @ k.transpose(-2, -1)) * self.scale

        attn = F.softmax(weight, dim =-1)
        attn = self.attn_drop(attn)

        # (B, NH, L, L) @ (B, NH, L, HD) -> (B, NH, L, HD)
        out = attn @ v 
        
        # (B, NH, L, HD) -> (B, L, NH, HD)
        out = out.transpose(1, 2)

        # (B, L, NH, HD) -> (B, N, dim)
        out = out.reshape(B,N,C)

        #(B, L, dim) -> (B, L, dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class VideoTransformer(nn.Module):
    """ 
    Vision Transformer
    """

    def __init__(self, num_patches, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                   norm_layer=nn.LayerNorm, num_frames=8, dropout=0.):

        super().__init__()
        self.depth = depth
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_frames = num_frames
        self.num_patches = num_patches
        # Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.time_drop = nn.Dropout(p=drop_rate)
        # Attention Blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

       
    def forward(self, x):
        # x shape: (B*T, N, dim)  where N=num_patches per frame
        BT = x.shape[0]
        T = self.num_frames
        N = self.num_patches
        B = BT // T

        # cls_tokens -> (B*T, 1, dim)
        cls_tokens = self.cls_token.expand(BT, -1, -1)
        # (B*T, N, dim) -> (B*T, N+1, dim)
        x = torch.cat((cls_tokens, x), dim=1)
    
        x = x + self.pos_embed # add spatial pos embedding
        x = self.pos_drop(x)
        # Time Embeddings

        #Seperate cls token
        cls_tokens = x[:B, 0, :].unsqueeze(1)

        # (B*T, N+1, dim) -> (B*T, N, dim)
        x = x[:,1:]

        # (B*T, N, dim) -> (B*N, T, dim)
        x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)

        x = x + self.time_embed # Add temporal embedding
        x = self.time_drop(x)

        # (B*N, T, dim) -> (B, N*T, dim)
        x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)

        # (B, N*T, dim) -> (B, N*T + 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T)

        x = self.norm(x)
        #return only cls token
        return x[:, 0]
    