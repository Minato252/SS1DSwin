import torch
import torch.nn as nn

import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads, dim_head, dropout, withoutlinear=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        if not withoutlinear:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            )
        self.withoutlinear = withoutlinear

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        q = q * self.scale
        dots = (q @ k.transpose(-2, -1))


        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)


        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        if self.withoutlinear:
            return out
        out = self.to_out(out)
        return out


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)


        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = torch.einsum('bcij,bcjk->bcik', attn, v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


def window_partition(x, window_size):
    B, N, C = x.shape
    x = x.view(B, N // window_size, window_size, C)
    windows = x.view(-1, window_size, C)
    return windows


def window_reverse(windows, window_size, N):
    B = int(windows.shape[0] / (N / window_size))
    x = windows.view(B, N // window_size, window_size, -1)
    x = x.view(B, N, -1)
    return x


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


class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if self.input_resolution <= self.window_size:
            self.shift_size = 0
            self.window_size = self.input_resolution
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            N = self.input_resolution
            img_mask = torch.zeros((1, N, 1))
            n_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0

            for n in n_slices:
                img_mask[:, n, :] = cnt
                cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(
                2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):

        N = self.input_resolution
        B, L, C = x.shape
        assert L == N, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x,
                                     self.window_size)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        shifted_x = window_reverse(attn_windows, self.window_size, N)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            x = shifted_x
        x = x.view(B, N, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, group, output_dim,
                 mlp_ratio=.125, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,

                 normAll=False, PosEmb=None, ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.norm_layer = norm_layer
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,)
            for i in range(depth)])

        self.resNorm = nn.LayerNorm([self.input_resolution, self.dim, 2])
        self.resLinear = nn.Linear(self.dim * 2, self.dim)
        self.resLinearPreX = nn.Linear(self.dim, self.dim)
        self.resLinearX = nn.Linear(self.dim, self.dim)


        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer, group=group,
                                         outputdim=output_dim, normAll=normAll, PosEmb=PosEmb)
        else:
            self.downsample = None

    def forward(self, x):


        pre_x = x

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)


        x = torch.cat((pre_x.unsqueeze(-1), x.unsqueeze(-1)), dim=-1)
        x = self.resNorm(x)
        pre_x, x = x[:, :, :, 0], x[:, :, :, 1]

        pre_x = self.resLinearPreX(pre_x)
        x = self.resLinearX(x)
        x = torch.cat((pre_x, x), dim=-1)
        x = self.resLinear(x)

        if self.downsample is not None:


            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class TokenMerging(nn.Module):

    def __init__(self, input_resolution, dim, group, outputdim, norm_layer=nn.LayerNorm, normAll=False, PosEmb="Post"):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(group * dim, outputdim * dim, bias=False)
        if normAll:
            self.norm = norm_layer([input_resolution // group, group * dim])
        else:
            self.norm = norm_layer(group * dim)

        self.PosEmb = PosEmb
        if PosEmb == "Post" or PosEmb == "Pre":
            self.posPreEmb = nn.Parameter(torch.randn((input_resolution // group, group * dim)))
            self.posPostEmb = nn.Parameter(torch.randn((input_resolution // group, dim * outputdim)))
        self.group = group

    def forward(self, x):
        """
        x: B, N, C
        """
        N = self.input_resolution
        B, L, C = x.shape
        assert L == N, "input feature has wrong size"
        assert N % self.group == 0, f"x size ({N}) are not even."

        xlist = []
        for i in range(self.group):
            xlist.append(x[:, i::self.group, :])

        x = torch.cat(xlist, -1)
        x = x.view(B, -1, self.group * C)

        x = self.norm(x)
        if self.PosEmb == "Pre":
            x += self.posPreEmb
        x = self.reduction(x)
        if self.PosEmb == "Post":
            x += self.posPostEmb
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class SwinT(nn.Module):
    def __init__(self, image_size, near_band, num_patches, patch_dim, num_classes, band, dim, heads,
                 pool='cls', dropout=0., emb_dropout=0.,  mode='ViT',
                 group=2, outputdim=1,
                 depths=[2, 2, 6, 2],
                 window_size=10,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, use_checkpoint=False, mlp_ratio=.125,

                 ):

        super().__init__()

        num_heads = [heads * (outputdim ** i) for i in range(len(depths))]

        self.band = band
        self.image_size = image_size
        self.mode = mode
        self.dim = dim
        self.patch_dim = patch_dim
        self.near_band = near_band
        self.num_patches = num_patches

        self.embedding_by_msa_pos = nn.Parameter(torch.randn(1, image_size ** 2, near_band))
        self.embedding_by_msa = Attention(dim=near_band, dim_head=self.dim // 4, heads=4,
                                          dropout=dropout)

        self.patch_to_embedding = nn.Linear(patch_dim, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.num_layers = len(depths)
        self.num_features = self.dim * (outputdim ** (self.num_layers - 1))
        self.norm = norm_layer(self.num_features)
        self.out_num_patches = (num_patches // (group ** (self.num_layers - 1)))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Linear(self.num_features, num_classes)
        )

        assert num_patches % (
                group ** (self.num_layers - 1)) == 0, "tokenmerging，group不是num_patches的整数倍"
        self.layers = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        resNorm = nn.LayerNorm([self.dim, 2])
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(self.dim * outputdim ** i_layer),
                               input_resolution=num_patches // (group ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               group=group,
                               output_dim=outputdim,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=TokenMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               )
            self.layers.append(layer)

    def forward(self, x, mask=None):

        b, _, c = x.shape

        band_patch = self.near_band
        n = self.num_patches
        x = x.contiguous().view(b,self.image_size,self.image_size,band_patch,n).permute(0,4,1,2,3).contiguous().view(b*n,self.image_size ** 2,band_patch)
        x += self.embedding_by_msa_pos
        x = self.embedding_by_msa(x)
        x = x.contiguous().view(b*n, band_patch * self.image_size ** 2)
        x = self.patch_to_embedding(x)
        x = x.view(b,n,self.dim)

        x += self.pos_embedding[:, : n]
        x = self.dropout(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

        x = self.norm(x)
        x = self.to_latent(torch.flatten(self.avgpool(x.transpose(1, 2)), 1))
        return self.mlp_head(x)
