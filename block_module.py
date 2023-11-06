from timm.layers.mlp import Mlp
from timm.layers.drop import DropPath
import torch
import torch.nn as nn
from attention_module import Attention

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.,
        attn_drop=0.,
        init_values=None,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, x_2=None):
        #x = x + self.drop_path1(self.ls1(self.attn(self.norm1(f1))))
        if x_2 == None:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            #x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), self.norm1(x))))
            #x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x_2), self.norm1(x_2))))
        else:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), self.norm1(x_2))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x