import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import to_2tuple


"""
Reference:
https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

Create 2D grid representing the positional embeddings of the frames.

embed_dim is the dimention of the embedding for encoder / decoder,
grid_size is the squareroot of the number of patches.
"""

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class Mlp(nn.Module):
  def __init__(
      self,
      in_dim,
      hidden_dim,
      act_layer=nn.GELU,
      linear_layer=nn.Linear,
      bias=True,
  ):
    super().__init__()
    bias = to_2tuple(bias)
    out_dim = in_dim

    self.l1 = linear_layer(in_dim, hidden_dim, bias=bias[0])
    self.act = act_layer()
    self.l2 = linear_layer(hidden_dim, out_dim, bias=bias[1])

  def forward(self, x):
    x = self.l1(x)
    x = self.act(x)
    x = self.l2(x)
    return x

class EncoderBlock(nn.Module):
  def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
    ):
    super().__init__()
    self.norm1 = norm_layer(dim)

    self.norm2 = norm_layer(dim)
    self.mlp = mlp_layer(
        in_dim=dim,
        hidden_dim=int(dim * mlp_ratio),
        act_layer=act_layer,
    )

    self.num_heads = num_heads
    self.head_dim = dim // num_heads

    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.proj = nn.Linear(dim, dim)

  def attention(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3,
                              self.num_heads,
                              self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    x = F.scaled_dot_product_attention(q, k, v)

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    return x

  def forward(self, x):
    x = x + self.attention(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x

class PatchEmbed(nn.Module):
  def __init__(
      self,
      img_size=224,
      patch_size=16,
      in_chans=3,
      embed_dim=768,
      bias=True
  ):
    super().__init__()
    self.img_size = to_2tuple(img_size)
    self.patch_size = to_2tuple(patch_size)
    self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
    self.num_patches = self.grid_size[0] * self.grid_size[1]
    self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

  def forward(self, x):
    B, C, H, W = x.shape
    x = self.proj(x)
    x = x.flatten(2).transpose(1, 2)
    return x

class DecoderBlock(nn.Module):
  def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            act_layer=act_layer,
        )

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

  def cross_attention(self, x1, x2):
    B, N, C = x1.shape
    qkv_1 = self.qkv(x1).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q1, k1, v1 = qkv_1.unbind(0)

    qkv_2 = self.qkv(x2).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q2, k2, v2 = qkv_2.unbind(0)

    x = F.scaled_dot_product_attention(q2, k1, v1)

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)

    return x

  def self_attention(self, x):
    B, N, C = x.shape
    qkv_ = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv_.unbind(0)

    x = F.scaled_dot_product_attention(q, k, v)

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    return x

  def forward(self, x1, x2, cross_self=True, cross=False, joint=False):
    if cross_self:
      x = x2 + self.cross_attention(self.norm1(x1), self.norm1(x2))
      x = x + self.self_attention(self.norm2(x))
    elif cross:
      x = x2 + self.cross_attention(self.norm1(x1), self.norm1(x2))
    if joint:
      x = x1 + self.mlp(self.norm3(x1))
    else:
      x = x + self.mlp(self.norm3(x))
    return x

"""
Reference when building base arcitecture:
https://github.com/facebookresearch/mae/blob/main/models_mae.py
"""
class SiameseAutoencoderViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_layer=nn.LayerNorm
        ):
        super().__init__()

        # Encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        self.initialize_weights()

    def initialize_weights(self):

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio, joint_encoder=False):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        if joint_encoder:
          bmask = torch.zeros_like(mask, dtype=torch.bool)
          x_masked = x.clone()
          x_masked[~bmask] = 0

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, joint_encoder=False, joint_decoder=False):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        if mask_ratio > 0 or joint_decoder or joint_encoder:
            if joint_encoder:
              x1_idx = int(x.shape[0] / 2)
              x2_idx = x.shape[0]
              x1, mask1, ids_restore1 = self.random_masking(x[:x1_idx], mask_ratio=0)
              x2, mask2, ids_restore2 = self.random_masking(x[x1_idx:x2_idx], mask_ratio=mask_ratio)
              x = torch.vstack([x1, x2])
              mask = torch.vstack([mask1, mask2])
              ids_restore = torch.vstack([ids_restore1, ids_restore2])
            else:
              x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if mask_ratio > 0 or joint_decoder or joint_encoder:
            return x, mask, ids_restore
        else:
           return x

    def forward_decoder(self, f1, f2, ids_restore_2, joint_decoder=False):

        x_1 = self.decoder_embed(f1)
        x_1 = x_1 + self.decoder_pos_embed

        x_2 = self.decoder_embed(f2)
        mask_tokens = self.mask_token.repeat(f2.shape[0], ids_restore_2.shape[1] + 1 - x_2.shape[1], 1)
        x_2_ = torch.cat([x_2[:, 1:, :], mask_tokens], dim=1)
        x_2_ = torch.gather(x_2_, dim=1, index=ids_restore_2.unsqueeze(-1).repeat(1, 1, x_2.shape[2]))
        x_2 = torch.cat([x_2[:, :1, :], x_2_], dim=1)

        x_2 = x_2 + self.decoder_pos_embed

        if joint_decoder:
          x_1 = torch.vstack([x_1, x_2])

        for blk in self.decoder_blocks:
            x_2 = blk(x_1, x_2)

        x = self.decoder_norm(x_2)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask, joint_decoder=False):
        if joint_decoder:
          target = self.patchify(imgs[:, :, :, :])
        else:
          target = self.patchify(imgs[:, 1, :, :, :])

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()

        return loss

    def forward(self, imgs, mask_ratio=0.95, joint_encoder=False, joint_decoder=False):
        if joint_encoder:
          imgs_joint = torch.vstack([imgs[:, 0], imgs[:, 1]])
          latent, mask, ids_restore = self.forward_encoder(imgs_joint.float(), mask_ratio=mask_ratio)
          l1_idx = int(latent.shape[0] / 2)
          l2_idx = latent.shape[0]
          latent_1 = latent[:l1_idx]
          latent_2 = latent[l1_idx:l2_idx]
          ids_restore_2 = ids_restore[l1_idx:l2_idx]
          mask_1 = mask[:l1_idx]
          mask_2 = mask[l1_idx:l2_idx]
        elif joint_decoder:
          latent_1, mask_1, _ = self.forward_encoder(imgs[:, 0].float(), mask_ratio=0)
          latent_2, mask_2, ids_restore_2 = self.forward_encoder(imgs[:, 1].float(), mask_ratio=mask_ratio)
        else:
          latent_1 = self.forward_encoder(imgs[:, 0].float(), mask_ratio=0)
          latent_2, mask_2, ids_restore_2 = self.forward_encoder(imgs[:, 1].float(), mask_ratio=mask_ratio)

        if joint_decoder:
          imgs = torch.vstack([imgs[:, 0], imgs[:, 1]])
          mask_2 = torch.vstack([mask_1, mask_2])
        pred = self.forward_decoder(latent_1, latent_2, ids_restore_2)
        loss = self.forward_loss(imgs, pred, mask_2)
        return loss, pred


# Different model definitons
def sim_mae_vit_small_patch16_dec512d8b(**kwargs):
    model = SiameseAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def sim_mae_vit_small_patch8_dec512d8b(**kwargs):
    model = SiameseAutoencoderViT(
        patch_size=8, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def sim_mae_vit_tiny_patch16_dec512d8b(**kwargs):
    model = SiameseAutoencoderViT(
        patch_size=16, embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def sim_mae_vit_tiny_patch8_dec512d8b(**kwargs):
    model = SiameseAutoencoderViT(
        patch_size=8, embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model