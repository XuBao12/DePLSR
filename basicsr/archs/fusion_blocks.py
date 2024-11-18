import torch
from torch import nn
from timm.models.vision_transformer import DropPath, Mlp

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        (B, N1, C), N2 = x1.shape, x2.shape[1]
        q = self.q(x1).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x2).reshape(B, N2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x1 = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        x1 = self.proj(x1)
        x1 = self.proj_drop(x1)
        return x1, attn

class CrossAttention_Fusion(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., dim_ratio=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = int(dim * dim_ratio)

        self.attn_i = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.attn_d = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)

        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.k = nn.Linear(dim * 2, self.dim, bias=qkv_bias)
        self.v = nn.Linear(dim * 2, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, xi, xd):
        bs, nx = x.shape[0], x.shape[1]
        if xi.dim() == 2:
            xi = xi.unsqueeze(1)
        ni = xi.shape[1]
        if xd.dim() == 2:
            xd = xd.unsqueeze(1)
        nd = xd.shape[1]

        xi, _ = self.attn_i(xi, x)     # Attention between x and xi (xi is q, x is kv)
        xd, _ = self.attn_d(xd, x)     # Attention between x and xd (xd is q, x is kv)

        # Dense Interactions
        xid = torch.cat((
            xi.unsqueeze(2).repeat(1, 1, nd, 1),
            xd.unsqueeze(1).repeat(1, ni, 1, 1),
        ), dim=3).flatten(1, 2)

        q = self.q(x).reshape(bs, nx, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(xid).reshape(bs, ni*nd, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(xid).reshape(bs, ni*nd, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).flatten(2)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, xi, xd, attn

class FusionBlock(nn.Module):
    def __init__(self, dim, num_heads, attn_ratio=0.25, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1_x = norm_layer(dim)
        self.norm1_xd = norm_layer(dim)
        self.norm1_xi = norm_layer(dim)
        self.attn = CrossAttention_Fusion(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, dim_ratio=attn_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, xi, xd, return_attention=False):
        '''
        x: (B, (h w), C)
        xi: (B, ni, C) ni=1
        xd: (B, nd, C) nd=5
        '''
        x, xi, xd = self.norm1_x(x), self.norm1_xi(xi), self.norm1_xd(xd)
        x_fusion, xi, xd, attn = self.attn(x, xi, xd)
        x = x + self.drop_path(x_fusion)
        if return_attention:
            return attn

        x_fusion = self.mlp(self.norm2(x))
        x = x + self.drop_path(x_fusion)
        return x, xi, xd
