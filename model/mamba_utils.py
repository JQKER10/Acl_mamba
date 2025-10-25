import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

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
class PatchEmbedding3d(nn.Module):
    def __init__(self, d_model, patch_size=(2,4,4), in_chans=1, conv_bias=True, norm_layer=None, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.patch_size = to_2tuple(patch_size)
        
        self.proj = nn.Conv3d(in_chans, d_model, kernel_size=(patch_size[0], patch_size[1], patch_size[2]),
                              stride=(patch_size[0], patch_size[1], patch_size[2]), bias=conv_bias, **factory_kwargs)
        
        self.norm = norm_layer(d_model) if norm_layer else nn.Identity()

    def forward(self,x):
        x = self.proj(x)  # B, C, D, H, W
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2).transpose(1, 2)  # B, D*H*W, C
        x = self.norm(x)
        
        return x
    
def window_partition(x, window_size):
    """
    Args:
        x: (B,D, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D//window_size, window_size, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6,7).contiguous().view(-1,window_size, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, D, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, window_size, C)
        window_size (int): Window size
        D (int): Depth of image
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    B=int(windows.shape[0] / (D * H * W / window_size / window_size / window_size))
    windows = windows.view(B, D // window_size, H // window_size, W // window_size, window_size, window_size, window_size, -1)
    x = windows.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout3d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act() if act else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x
