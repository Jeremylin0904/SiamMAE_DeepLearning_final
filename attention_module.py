from timm.layers.config import use_fused_attn
from torch.jit import Final
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.,
        proj_drop=0.,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward_decoder(self, f1, f2):
        B, N, C = f1.shape
        qkv_1 = self.qkv(f1).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv_1.unbind(0)
        q1, k1 = self.q_norm(q1), self.k_norm(k1)

        qkv_2 = self.qkv(f2).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv_2.unbind(0)
        q2, k2 = self.q_norm(q2), self.k_norm(k2)
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q2, k1, v1,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q2 = q2 * self.scale
            attn = q2 @ k1.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v1

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_encoder(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x_cross_attn = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            x_cross_attn = x_cross_attn.transpose(1, 2).reshape(B, N, C)

            qkv = self.qkv(x_cross_attn).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)

            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )

        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, f1, f2=None):
        if f2 == None:
            x = self.forward_encoder(f1)
        else:
            x = self.forward_decoder(f1, f2)
        return x
        
        