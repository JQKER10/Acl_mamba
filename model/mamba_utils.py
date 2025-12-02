import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F

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
    
class StemLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,      # dùng 3 thay vì 7
        stride=2,
        padding=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = norm_layer(out_channels)
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MultiKernelConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_sizes=(3, 5, 7),
        activation=nn.SiLU,
        bias: bool = False,
        **factory_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes

        # depthwise conv cho mỗi kernel size, giữ nguyên C
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=k,
                stride=1,
                padding=k // 2,
                groups=in_channels,
                bias=bias,
                **factory_kwargs,
            )
            for k in kernel_sizes
        ])

        self.bn = nn.BatchNorm2d(in_channels, **factory_kwargs)
        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mỗi nhánh: (B,C,H,W) -> (B,C,H,W)
        conv_outputs = [conv(x) for conv in self.convs]
        x = sum(conv_outputs)         # vẫn (B,C,H,W)
        x = self.bn(x)
        x = self.act(x)
        return x

def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:

    batch_size, height, width, num_channels = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, height, width, groups, channels_per_group)

    x = torch.transpose(x, 3, 4).contiguous()

    # flatten
    x = x.view(batch_size, height, width, -1)

    return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super().__init__()
        # 2x downsample bằng conv stride 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x):  # x: (B, C_in, H, W)
        x = self.conv(x)   # (B, C_out, H/2, W/2)
        x = self.norm(x)
        return x


class NeighborAttentionFusion(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_neighbors: int = 2,
                 reduction: int = 4,
                 use_spatial: bool = True,
                 out_channels: int = None):
        super().__init__()
        self.in_channels = in_channels
        self.num_neighbors = num_neighbors
        self.use_spatial = use_spatial

        # K*C kênh sau khi ghép
        fused_in = in_channels * num_neighbors

        # Channel attention MLP
        hidden = max(fused_in // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(fused_in, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, fused_in, bias=False)
        )

        # Spatial attention
        if use_spatial:
            self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
            self.spatial_bn = nn.BatchNorm2d(1)
        else:
            self.spatial_conv = None
            self.spatial_bn = None

        # Output conv 1x1 fuse neighbors
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels

        self.fuse_conv = nn.Conv2d(fused_in, out_channels, kernel_size=1, bias=False)
        self.fuse_bn = nn.BatchNorm2d(out_channels)
        self.fuse_act = nn.ReLU(inplace=True)

    def forward(self, x_neighbors: torch.Tensor) -> torch.Tensor:
        """
        x_neighbors: (B, K, H, W, C)
        Return:
            x_fused: (B, H, W, C_fused)
        """
        B, K, H, W, C = x_neighbors.shape

        assert K == self.num_neighbors, f"Expected {self.num_neighbors} neighbors, got {K}"
        assert C == self.in_channels, f"Expected in_channels={self.in_channels}, got {C}"


        # (B,K,H,W,C) -> (B,K,C,H,W) -> (B,K*C,H,W)
        x = x_neighbors.permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(B, K * C, H, W)

        # Channel attention
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(B, K * C)
        max_pool = F.adaptive_max_pool2d(x, 1).view(B, K * C)

        avg_weight = self.mlp(avg_pool)
        max_weight = self.mlp(max_pool)

        channel_att = torch.sigmoid(avg_weight + max_weight).view(B, K * C, 1, 1)
        x = x * channel_att

        # Spatial attention
        if self.use_spatial and self.spatial_conv is not None:
            avg_pool_sp = torch.mean(x, dim=1, keepdim=True)          # (B,1,H,W)
            max_pool_sp, _ = torch.max(x, dim=1, keepdim=True)        # (B,1,H,W)
            sp = torch.cat([avg_pool_sp, max_pool_sp], dim=1)         # (B,2,H,W)

            spatial_att = self.spatial_conv(sp)
            spatial_att = self.spatial_bn(spatial_att)
            spatial_att = torch.sigmoid(spatial_att)
            x = x * spatial_att

        # Fuse neighbors bằng conv 1x1
        x = self.fuse_conv(x)
        x = self.fuse_bn(x)
        x = self.fuse_act(x)

        # (B, C_fused, H, W) -> (B, H, W, C_fused)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x




