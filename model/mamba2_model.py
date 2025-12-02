import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from .mamba_utils import MultiKernelConv, StemLayer,Mlp, Downsample, NeighborAttentionFusion
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
import copy
from typing import Optional, Tuple, List, Union, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count,squeeze_excitation
from torch.utils.checkpoint import checkpoint
import collections

def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse




class InterSlicePositionEmbedding(nn.Module):
    """
    Positional embedding theo trục lát (Z) cho 2.5D:
    - x: (B, num_slices, L, C)
    - pos_type: 'learnable' | 'sincos' | 'none'
    """
    def __init__(self, num_slices: int, embed_dim: int, pos_type: str = 'learnable'):
        super().__init__()
        self.num_slices = num_slices
        self.embed_dim = embed_dim
        self.pos_type = pos_type

        if pos_type == 'learnable':
            # 1 token PE cho mỗi lát: (1, K, C)
            self.pos_embed = nn.Parameter(torch.zeros(1, num_slices, embed_dim))
            trunc_normal_(self.pos_embed, std=0.02)

        elif pos_type == 'none':
            self.pos_embed = None

        else:
            raise ValueError(f"Unknown pos_type: {pos_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, num_slices, L, C)
        """
        if self.pos_embed is None:
            return x

        B, K, L, C = x.shape
        assert C == self.embed_dim, \
            f"embed_dim={self.embed_dim} nhưng x.shape[-1]={C}"

        # đảm bảo không lệch device/dtype
        pe = self.pos_embed[:, :K].to(dtype=x.dtype, device=x.device)  # (1,K,C)

        # broadcast: (1,K,1,C) + (B,K,L,C)
        x = x + pe.unsqueeze(2)
        return x


class SSD(nn.Module):
    """
    Intra-slice SSD theo style VSSD:
      - Dùng dt (B,L,nheads) + A (nheads) để sinh dA (B,H,L,1)
      - Non-causal linear attention duality kiểu non_casual_linear_attn (bản rút gọn)
    """

    def __init__(
        self,
        d_model,
        ssd_expand=2,
        d_state=64,
        headdim=64,
        ngroups=-1,           # không dùng nữa, giữ cho tương thích
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        activation=nn.SiLU,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        # ===== DIMENSIONS =====
        self.d_model = d_model
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * self.d_model)
        self.headdim = headdim
        self.d_state = d_state

        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim

        # flags giống VSSD
        self.ssd_aexp = kwargs.get("ssd_aexp", True)
        self.ssd_positive_dA = kwargs.get("ssd_positve_dA", True)
        self.ssd_norm_dA = kwargs.get("ssd_norm_da", True)

        self.act = activation()

        # ===== INPUT PROJECTION =====
        # X: d_inner, B: d_state, C: d_state, dt: nheads
        d_in_proj = self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=bias, **factory_kwargs)

        # Optional depthwise conv trên [B,C] để lấy spatial context
        self.dwconv = MultiKernelConv(
            in_channels=2 * self.d_state,
            kernel_sizes=[3, 5, 7],
            activation=activation,
            bias=bias,
            **factory_kwargs,
        )

        # ===== dt & A & D =====
        # dt bias (per-head)
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)          # (H,)
        self.dt_bias._no_weight_decay = True

        # A: (nheads,) như VSSD (trường hợp A có 1 state per head)
        A = torch.empty(self.nheads, **factory_kwargs).uniform_(*A_init_range)
        self.A = nn.Parameter(A)                     # (H,)

        # D skip (per-head)
        self.D = nn.Parameter(torch.ones(self.nheads, **factory_kwargs))

        # ===== Gating & output =====
        self.gate_proj = nn.Linear(self.d_inner, 2 * self.d_inner, bias=bias, **factory_kwargs)
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def _build_dA(self, dt: torch.Tensor) -> torch.Tensor:
        """
        dt: (B, L, H) -> sau đó permute thành (B, H, L)
        Trả về dA: (B, H, L, 1) giống với non_casual_linear_attn (nhánh linear_norm=False).
        """
        # (B,L,H) -> (B,H,L)
        dt = dt.permute(0, 2, 1)  # (B,H,L)

        # A: (H,) -> (1,H,1,1)
        A = self.A.view(1, -1, 1, 1)  # (1,H,1,1)

        dA = dt.unsqueeze(-1) * A     # (B,H,L,1)

        if self.ssd_aexp:
            dA = 1.0 / dA.exp()
        if self.ssd_positive_dA:
            dA = -dA
        if self.ssd_norm_dA:
            # chuẩn hóa theo chiều L giống code VSSD (dim=-2)
            dA = dA / (dA.sum(dim=-2, keepdim=True) + 1e-6)

        return dA  # (B,H,L,1)

    def forward(self, u: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        u: (B, L, C) hoặc (B, C, H, W)
        H, W: spatial size (đã biết trước hoặc truyền vào)
        Trả về: (B, L, C)
        """
        # reshape input
        if u.dim() == 4:
            B, C, H_in, W_in = u.shape
            assert H == H_in and W == W_in
            u = u.flatten(2).transpose(1, 2)  # (B,L,C)

        B, L, C = u.shape
        assert L == H * W, f"seq_len {L} != H*W {H*W}"

        # ==== projection ====
        xbcdt = self.in_proj(u)  # (B,L, d_inner + 2*d_state + nheads)
        X, BC, dt = torch.split(
            xbcdt,
            [self.d_inner, 2 * self.d_state, self.nheads],
            dim=-1,
        )

        # ==== spatial conv trên [B,C] ====
        BC_2d = BC.view(B, H, W, 2 * self.d_state).permute(0, 3, 1, 2)  # (B,2*d_state,H,W)
        BC_2d = self.dwconv(BC_2d)                                      # (B,2*d_state,H,W)
        BC = BC_2d.permute(0, 2, 3, 1).reshape(B, L, 2 * self.d_state)  # (B,L,2*d_state)

        B_key, C_q = torch.split(BC, [self.d_state, self.d_state], dim=-1)  # (B,L,d_state)x2

        # ==== chuẩn bị V, Q, K ====
        # V: (B,H,L,D)
        V = X.view(B, L, self.nheads, self.headdim).permute(0, 2, 1, 3)

        # K, Q như VSSD: (B,1,L,d_state)
        K = B_key.view(B, 1, L, self.d_state)
        Q = C_q.view(B, 1, L, self.d_state)

        # ==== dA ====
        dA = self._build_dA(dt)  # (B,H,L,1)

        # ==== non-linear_norm=False nhánh của VSSD ====
        # scale V theo dA
        V_scaled = V * dA  # (B,H,L,D)

        # KV = K^T @ V_scaled -> (B,H,d_state,D)
        KV = torch.matmul(
            K.transpose(-2, -1),   # (B,1,d_state,L) -> broadcast H
            V_scaled,              # (B,H,L,D)
        )  # (B,H,d_state,D)

        # x = Q @ KV -> (B,H,L,D)
        x = torch.matmul(Q, KV)  # (B,H,L,D)

        # skip như VSSD: x + V * D
        D = self.D.view(1, -1, 1, 1)  # (1,H,1,1)
        x = x + V * D                 # (B,H,L,D)

        # (B,H,L,D) -> (B,L,d_inner)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, L, self.d_inner)
        y = x
        # ===== Gating + output =====
        w, z = self.gate_proj(y).chunk(2, dim=-1)
        y = self.act(w) * z

        # residual + out
        y = self.norm(y)
        out = self.out_proj(y)  # (B,L,d_model)

        return out

    

class Inter_slice_SSD(nn.Module):
    """
    Inter-slice SSD (cross-slice) dùng dA_vec learnable:
      - center -> Q
      - neighbor_fused -> K, V
      - dA_vec (H,d_state) đóng vai trò foreground selector
      - Linear NC-SSD: S = K^T (Q⊙V), Y = Q·S, + skip D*V
    """

    def __init__(
        self,
        d_model,
        ssd_expand=2,
        d_state=64,
        headdim=64,
        norm_layer=nn.LayerNorm,
        act_layer=nn.SiLU,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        factory = dict(device=device, dtype=dtype)

        self.d_model = d_model
        self.d_inner = int(ssd_expand * d_model)
        self.headdim = headdim
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.d_state = d_state

        self.act = act_layer()

        # Center slice -> Q
        self.q_proj = nn.Linear(
            d_model,
            self.nheads * self.d_state,
            bias=bias,
            **factory,
        )

        # Neighbor fused -> K (state) + V (value)
        self.kv_proj = nn.Linear(
            d_model,
            self.nheads * self.d_state + self.d_inner,
            bias=bias,
            **factory,
        )

        # dA_vec: learnable foreground selector (H,d_state)
        self.ssd_aexp = kwargs.get("ssd_aexp", True)
        self.ssd_positive_dA = kwargs.get("ssd_positve_dA", True)
        self.ssd_norm_dA = kwargs.get("ssd_norm_da", True)

        self.dA_vec = nn.Parameter(
            torch.zeros(self.nheads, self.d_state, **factory)
        )
        nn.init.normal_(self.dA_vec, std=0.02)

        # D skip per head
        self.D = nn.Parameter(torch.ones(self.nheads, 1, 1, **factory))

        # Gating & out
        self.gate_proj = nn.Linear(self.d_inner, 2 * self.d_inner, bias=bias, **factory)
        self.norm = norm_layer(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory)

    def _process_dA(self) -> torch.Tensor:
        """
        dA_vec: (H,d_state) -> dA: (1,H,1,d_state)
        """
        dA = self.dA_vec  # (H,d_state)
        if self.ssd_aexp:
            dA = 1.0 / dA.exp()
        if self.ssd_positive_dA:
            dA = dA.abs()
        if self.ssd_norm_dA:
            dA = dA / (dA.sum(dim=-1, keepdim=True) + 1e-6)
        return dA.unsqueeze(0).unsqueeze(2)  # (1,H,1,d_state)

    def forward(self, x_center: torch.Tensor, x_neighbor: torch.Tensor, H: int, W: int):
        """
        x_center:  (B,H,W,C)
        x_neighbor:(B,H,W,C)    # đã fuse tất cả neighbors
        """
        B, H2, W2, C = x_center.shape
        assert H == H2 and W == W2 and C == self.d_model

        L = H * W

        # flatten
        xc = x_center.reshape(B, L, C)     # (B,L,C)
        xn = x_neighbor.reshape(B, L, C)   # (B,L,C)

        # Q from center
        Q = self.q_proj(xc)                # (B,L,H*d_state)
        Q = Q.view(B, L, self.nheads, self.d_state).transpose(1, 2)  # (B,H,L,d_state)

        # K,V from neighbor
        KV = self.kv_proj(xn)              # (B,L,H*d_state + d_inner)
        Kv_state = self.nheads * self.d_state
        K_n, V_n = torch.split(KV, [Kv_state, self.d_inner], dim=-1)

        K_n = K_n.view(B, L, self.nheads, self.d_state).transpose(1, 2)      # (B,H,L,d_state)
        V_n = V_n.view(B, L, self.nheads, self.headdim).transpose(1, 2)      # (B,H,L,D)

        # dA
        dA = self._process_dA()            # (1,H,1,d_state)

        # ===== NC-SSD linear cross-slice =====
        # Qs = Q ⊙ dA
        Qs = Q * dA                        # (B,H,L,d_state)

        # S = K^T (Qs ⊙ V) -> (B,H,d_state,D)
        S = torch.einsum(
            "bhld, bhld, bhle -> bhde",
            K_n, Qs, V_n
        )

        # Y = Q · S -> (B,H,L,D)
        Y = torch.einsum(
            "bhld, bhde -> bhle",
            Q, S
        )

        # Skip per-head
        Y = Y + self.D * V_n               # (B,H,L,D)

        # Merge heads
        Y = Y.permute(0, 2, 1, 3).contiguous().view(B, L, self.d_inner)  # (B,L,d_inner)

        # Gating + out_proj (giống intra-slice)
        
        w, z = self.gate_proj(y).chunk(2, dim=-1)
        y = self.act(w) * z

        y = self.norm(y)
        y = y + Y
        out = self.out_proj(y)             # (B,L,C)

        return out.view(B, H, W, C)


class SSDBlock(nn.Module):
    r""" SSD Block (intra-slice), dùng được cho cả center & neighbor.

    Args:
        dim (int): số kênh embedding (d_model).
        input_resolution (tuple[int]): (H, W) mặc định nếu không truyền H,W ở forward.
        num_heads (int): số head nội bộ cho SSD (dim phải chia hết cho num_heads).
        mlp_ratio (float): tỉ lệ hidden dim của MLP so với dim.
        drop (float): dropout trong MLP.
        drop_path (float): stochastic depth.
        act_layer (nn.Module): hàm kích hoạt trong MLP.
        norm_layer (nn.Module): lớp chuẩn hoá (LayerNorm).
        ssd_expansion (int): hệ số expand nội bộ của SSD (ssd_expand).
        ssd_ngroups (int): số group trong SSD (ngroups), -1 = auto = d_inner//headdim.
        d_state (int): kích thước state của SSD.
        use_cpe (bool): dùng CPE depthwise conv 3×3 trước/sau SSD.
        **kwargs: truyền xuống SSD nếu cần thêm tham số.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        ssd_expansion=2,
        ssd_ngroups=-1,
        d_state=64,
        use_cpe=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (H, W)
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_cpe = use_cpe

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        headdim = dim // num_heads

        self.ssd = SSD(
            d_model=dim,
            ssd_expand=ssd_expansion,
            d_state=d_state,
            headdim=headdim,
            ngroups=ssd_ngroups,
            **kwargs,
        )

        # Norm + MLP
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Convolutional Positional Encoding (optional)
        if self.use_cpe:
            # depthwise conv 3×3
            self.cpe1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
            self.cpe2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        else:
            self.cpe1 = None
            self.cpe2 = None

    def _apply_cpe(self, x, conv, H, W):
        """x: (B,L,C), conv: Conv2d depthwise, return (B,L,C)"""
        if conv is None:
            return x
        B, L, C = x.shape
        x_img = x.transpose(1, 2).reshape(B, C, H, W)    # (B,C,H,W)
        x_img = conv(x_img)
        x = x_img.reshape(B, C, L).transpose(1, 2)       # (B,L,C)
        return x

    def forward(self, x: torch.Tensor, H: int = None, W: int = None) -> torch.Tensor:
        """
        x: (B, L, C) với L = H * W.
        H, W: nếu không truyền thì dùng self.input_resolution.
        """
        B, L, C = x.shape
        if H is None or W is None:
            H, W = self.input_resolution
        assert L == H * W, f"input feature has wrong size, got L={L}, H*W={H*W}"

        
        if self.use_cpe:
            x = x + self._apply_cpe(x, self.cpe1, H, W)


        shortcut = x
        x_norm = self.norm1(x)
        x_ssd = self.ssd(x_norm, H, W)     
        x = shortcut + self.drop_path(x_ssd)

        
        if self.use_cpe:
            x = x + self._apply_cpe(x, self.cpe2, H, W)

        
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class InterSliceSSDBlock(nn.Module):
    r""" Inter-slice SSD Block với neighbor đã fuse (1 feature map).

    Dùng sau khi đã encode intra-slice (SSDBlock) + fuse neighbors:
      - x_center       : (B, H, W, C)
      - x_neighbor_fused: (B, H, W, C)

    Cấu trúc:
      PreNorm(center) -> Inter_slice_SSD(core) -> +skip(center)
      -> FFN (MLP) -> +skip
    """

    def __init__(
        self,
        dim,
        input_resolution,
        mlp_ratio=2.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        ssd_expand=2,
        d_state=64,
        headdim=64,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (H, W)

        # Core inter-slice SSD: dùng dA_vec learnable, neighbor = 1 map
        self.cross_ssd = Inter_slice_SSD(
            d_model=dim,
            ssd_expand=ssd_expand,
            d_state=d_state,
            headdim=headdim,
            norm_layer=norm_layer,
            act_layer=act_layer,
            bias=bias,
            device=device,
            dtype=dtype,
            **kwargs,
        )

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x_center: torch.Tensor,        # (B, H, W, C)
        x_neighbor_fused: torch.Tensor, # (B, H, W, C)
        H: int = None,
        W: int = None,
    ) -> torch.Tensor:
        B, Hc, Wc, C = x_center.shape
        if H is None or W is None:
            H, W = self.input_resolution
        assert Hc == H and Wc == W
        assert C == self.dim, f"dim mismatch: got {C}, expected {self.dim}"

        # PreNorm trên center
        x_flat = x_center.view(B, H * W, C)
        x_norm_flat = self.norm1(x_flat)
        x_norm = x_norm_flat.view(B, H, W, C)

        # Inter-slice SSD: (center_norm, neighbor_fused) -> delta
        delta = self.cross_ssd(x_norm, x_neighbor_fused, H, W)  # (B,H,W,C)

        # Residual 1
        x = x_center + self.drop_path(delta)  # (B,H,W,C)

        # FFN branch
        xf = x.view(B, H * W, C)
        xf_ffn = self.mlp(self.norm2(xf))
        xf = xf + self.drop_path(xf_ffn)

        return xf.view(B, H, W, C)


class DualStreamEncoder2D(nn.Module):
    """
    Dual-stream 2D encoder cho 2.5D:

    Pipeline:
      - Input:
          x_center:   (B, C_in, H, W)          # lát trung tâm
          x_neighbors:(B, K, C_in, H, W)       # K lát lân cận

      - Bước 0 (chung):
          + StemLayer cho center và từng neighbor slice.
          + Thêm slice positional embedding cho từng lát (center & neighbor).
          + NeighborAttentionFusion gộp K láng giềng → 1 feature map neighbor fused.

      - Mỗi stage i:
          + Chạy depths[i] SSDBlock cho center (có grad).
          + Chạy depths[i] SSDBlock cho neighbor fused (EMA, no grad).
          + 1 InterSliceSSDBlock: fuse center với neighbor fused.
          + Lưu feature center sau inter-slice làm skip cho decoder.
          + Downsample center & neighbor fused, sang stage i+1.
    """

    def __init__(
        self,
        img_size=(256, 256),
        in_channels=1,
        dims=(64, 128, 256, 512),
        depths=(2, 2, 2, 2),
        num_heads=(2, 4, 8, 8),
        neighbor_slices: int = 2,
        ssd_expansion=2,
        d_state=64,
        use_cpe=True,
        momentum=0.99,
        use_naf: bool = True,
        slice_pe_type: str = "learnable",  # 'learnable' / 'sincos' / 'none'
        **ssd_kwargs,
    ):
        super().__init__()
        self.num_stages = len(dims)
        self.momentum = momentum
        self.neighbor_slices = neighbor_slices
        self.use_naf = use_naf

        # stem chung
        self.stem = StemLayer(
            in_channels=in_channels,
            out_channels=dims[0],
            kernel_size=7,
            stride=2,
            padding=3,
            norm_layer=nn.BatchNorm2d,
        )

        # slice position embedding cho stage 0 (sau stem)
        # center: 1 slice
        self.slice_pe_center = InterSlicePositionEmbedding(
            num_slices=1, embed_dim=dims[0], pos_type=slice_pe_type
        )
        # neighbors: K slices
        self.slice_pe_neighbors = InterSlicePositionEmbedding(
            num_slices=neighbor_slices, embed_dim=dims[0], pos_type=slice_pe_type
        )

        # Neighbor attention fusion cho stage 0 (gộp K neighbor slices)
        if self.use_naf:
            self.neighbor_fusion0 = NeighborAttentionFusion(
                in_channels=dims[0],
                num_neighbors=neighbor_slices,
                reduction=4,
                use_spatial=True,
                out_channels=dims[0],
            )
        else:
            self.neighbor_fusion0 = None

        H, W = img_size
        H0, W0 = H // 2, W // 2  # do stem stride=2

        # Mỗi stage:
        #   - center_blocks[i]  : list[SSDBlock], length = depths[i]
        #   - neighbor_blocks[i]: EMA copy của center_blocks[i]
        #   - inter_blocks[i]   : 1 InterSliceSSDBlock / stage
        #   - downs_center / downs_neighbor: PatchMerging
        self.center_blocks = nn.ModuleList()
        self.neighbor_blocks = nn.ModuleList()
        self.inter_blocks = nn.ModuleList()
        self.downs_center = nn.ModuleList()
        self.downs_neighbor = nn.ModuleList()

        for i in range(self.num_stages):
            Hi = H0 // (2 ** i)
            Wi = W0 // (2 ** i)
            Cin = dims[i]
            Cout = dims[i + 1] if i < self.num_stages - 1 else dims[i]

            # SSDBlock cho center
            center_stage = nn.ModuleList(
                [
                    SSDBlock(
                        dim=Cin,
                        input_resolution=(Hi, Wi),
                        num_heads=num_heads[i],
                        mlp_ratio=4.0,
                        drop=0.0,
                        drop_path=0.0,
                        act_layer=nn.GELU,
                        norm_layer=nn.LayerNorm,
                        ssd_expansion=ssd_expansion,
                        ssd_ngroups=-1,
                        d_state=d_state,
                        use_cpe=use_cpe,
                        **ssd_kwargs,
                    )
                    for _ in range(depths[i])
                ]
            )
            self.center_blocks.append(center_stage)

            # neighbor = EMA copy
            neighbor_stage = copy.deepcopy(center_stage)
            self.neighbor_blocks.append(neighbor_stage)

            # Inter-slice SSD: 1 block / stage
            inter_block = InterSliceSSDBlock(
                dim=Cin,
                input_resolution=(Hi, Wi),
                mlp_ratio=2.0,
                drop=0.0,
                drop_path=0.0,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                ssd_expand=ssd_expansion,
                d_state=d_state,
                headdim=Cin // num_heads[i],
            )
            self.inter_blocks.append(inter_block)

            # Downsample cho 2 stream
            if i < self.num_stages - 1:
                self.downs_center.append(
                    Downsample(Cin, Cout, norm_layer=nn.BatchNorm2d)
                )
                self.downs_neighbor.append(
                    Downsample(Cin, Cout, norm_layer=nn.BatchNorm2d)
                )

    @torch.no_grad()
    def _update_neighbor(self):
        """
        EMA update: copy weights từ center_blocks sang neighbor_blocks.
        Gọi trước mỗi forward.
        """
        if self.momentum is None:
            return
        for cs, ns in zip(self.center_blocks, self.neighbor_blocks):
            for p_c, p_n in zip(cs.parameters(), ns.parameters()):
                p_n.data = p_n.data * self.momentum + p_c.data * (1.0 - self.momentum)

    def forward(
        self,
        x_center: torch.Tensor,       # (B, C_in, H, W)
        x_neighbors: torch.Tensor,    # (B, K, C_in, H, W)
    ):
        self._update_neighbor()

        B, K, C_in, H, W = x_neighbors.shape
        assert K == self.neighbor_slices, f"Expected {self.neighbor_slices} neighbors, got {K}"

        # ----- 0. Stem cho center -----
        x_c = self.stem(x_center)   # (B, C0, H0, W0)
        B, C0, H0, W0 = x_c.shape

        # ----- 0. Stem cho từng neighbor slice -----
        x_neighbors = x_neighbors.view(B * K, C_in, H, W)     # (B*K, C_in, H, W)
        x_neighbors = self.stem(x_neighbors)                  # (B*K, C0, H0, W0)
        x_neighbors = x_neighbors.view(B, K, C0, H0, W0)      # (B, K, C0, H0, W0)

        # ----- 1. Slice position embedding cho center -----
        # chuyển center thành (B, 1, L, C0) để dùng InterSlicePositionEmbedding
        L0 = H0 * W0
        xc_flat = x_c.permute(0, 2, 3, 1).reshape(B, 1, L0, C0)    # (B,1,L0,C0)
        xc_flat = self.slice_pe_center(xc_flat)                    # (B,1,L0,C0)
        xc_flat = xc_flat[:, 0]                                    # (B,L0,C0)
        x_c = xc_flat.view(B, H0, W0, C0).permute(0, 3, 1, 2)      # (B,C0,H0,W0)

        # ----- 1. Slice position embedding cho neighbor slices -----
        xn_flat = x_neighbors.permute(0, 1, 3, 4, 2).reshape(B, K, L0, C0)  # (B,K,L0,C0)
        xn_flat = self.slice_pe_neighbors(xn_flat)                          # (B,K,L0,C0)
        x_neighbors = xn_flat.view(B, K, H0, W0, C0)                        # (B,K,H0,W0,C0)

        # ----- 2. NeighborAttentionFusion: gộp K lát neighbor thành 1 map -----
        if self.neighbor_fusion0 is not None:
            # NAF expect (B,K,H,W,C)
            nei_fused_hw = self.neighbor_fusion0(x_neighbors)     # (B,H0,W0,C0)
            x_n = nei_fused_hw.permute(0, 3, 1, 2)                # (B,C0,H0,W0)
        else:
            # fallback: lấy mean theo K
            x_n = x_neighbors.mean(dim=1).permute(0, 4, 2, 3)     # (B,C0,H0,W0)

        skips = []

        # ----- 3. Các stage SSDBlock + InterSliceSSDBlock + Downsample -----
        for i in range(self.num_stages):
            blocks_c = self.center_blocks[i]
            blocks_n = self.neighbor_blocks[i]
            inter = self.inter_blocks[i]

            B_, C_, H_, W_ = x_c.shape
            assert C_ == blocks_c[0].dim

            # 3.1 Center: chạy hết depths[i] SSDBlock (intra-slice)
            xc_flat = x_c.permute(0, 2, 3, 1).reshape(B_, H_ * W_, C_)
            for blk_c in blocks_c:
                xc_flat = blk_c(xc_flat, H_, W_)
            x_c = xc_flat.view(B_, H_, W_, C_).permute(0, 3, 1, 2)

            # 3.2 Neighbor fused: chạy hết depths[i] SSDBlock (EMA, no grad)
            with torch.no_grad():
                xn_flat = x_n.permute(0, 2, 3, 1).reshape(B_, H_ * W_, C_)
                for blk_n in blocks_n:
                    xn_flat = blk_n(xn_flat, H_, W_)
                x_n = xn_flat.view(B_, H_, W_, C_).permute(0, 3, 1, 2)

            # 3.3 Inter-slice SSD 1 lần cho cả stage
            xc_hw = x_c.permute(0, 2, 3, 1).contiguous()  # (B_,H_,W_,C_)
            xn_hw = x_n.permute(0, 2, 3, 1).contiguous()  # (B_,H_,W_,C_)

            xc_hw = inter(xc_hw, xn_hw, H_, W_)           # (B_,H_,W_,C_)  (resblock + FFN)
            x_c = xc_hw.permute(0, 3, 1, 2)               # (B_,C_,H_,W_)

            # 3.4 Lưu skip cho decoder
            skips.append(x_c)

            # 3.5 Downsample nếu chưa phải stage cuối
            if i < self.num_stages - 1:
                x_c = self.downs_center[i](x_c)
                x_n = self.downs_neighbor[i](x_n)

        center_last = x_c
        return center_last, skips



class Bottleneck(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

    def forward(self, x):
        return self.block(x)
    
class SSDDecoder(nn.Module):
    """
    Multi-scale UNETR-style decoder + deep supervision (sau mỗi upsample).
    feats[0] = largest H, feats[-1] = smallest H
    """
    def __init__(self, dims, out_channels, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        S = len(dims)

        # Up blocks: S-1 → S-2 → ... → 0
        self.up_blocks = nn.ModuleList([
            UnetrUpBlock(
                spatial_dims=2,
                in_channels=dims[i],
                out_channels=dims[i-1],
                kernel_size=3,
                stride=2,
                norm_name="instance",
                res_block=True,
            )
            for i in range(S-1, 0, -1)
        ])

        # Output chính
        self.out_head = UnetOutBlock(
            spatial_dims=2,
            in_channels=dims[0],
            out_channels=out_channels
        )

        # Deep supervision: dec_feats[1]..dec_feats[S-2]
        if deep_supervision and S > 2:
            self.ds_heads = nn.ModuleList([
                UnetOutBlock(
                    spatial_dims=2,
                    in_channels=dims[i],
                    out_channels=out_channels
                )
                for i in range(1, S-1)
            ])
        else:
            self.ds_heads = None

    def forward(self, feats):
        S = len(feats)
        x = feats[-1]   # bắt đầu ở đáy encoder/bottleneck

        dec_feats = [None] * S
        dec_feats[-1] = x

        # upsample
        for idx, up in enumerate(self.up_blocks):
            enc_idx = S - 2 - idx    # mapping ngược encoder
            skip = feats[enc_idx]
            x = up(x, skip)
            dec_feats[enc_idx] = x

        # output chính
        out = self.out_head(dec_feats[0])

        # không dùng deep supervision
        if not self.deep_supervision or self.ds_heads is None:
            return out

        ds_outs = []
        for j, head in enumerate(self.ds_heads):
            i = j + 1        # map 1→dec_feats[1], 2→dec_feats[2], ...
            ds_outs.append(head(dec_feats[i]))

        return out, ds_outs

        
class MAMBA2(nn.Module):

    def __init__(
        self,
        img_size=(256,256),
        in_channels=1,
        out_channels=2,
        dims=(64,128,256,512),
        depths=(2,2,2,2),
        num_heads=(2,4,8,8),
        **ssd_kwargs
    ):
        super().__init__()

        self.encoder = DualStreamEncoder2D(
            img_size=img_size,
            in_channels=in_channels,
            dims=dims,
            depths=depths,
            num_heads=num_heads,
            **ssd_kwargs
        )

        self.bottleneck = Bottleneck(dim=dims[-1])

        self.decoder = SSDDecoder(
            dims=dims,
            out_channels=out_channels,
            deep_supervision=True
        )

    def forward(self, x_center, x_neighbors):
        center_last, skips = self.encoder(x_center, x_neighbors)

        # bottleneck
        x_bot = self.bottleneck(center_last)
        skips[-1] = x_bot

        out = self.decoder(skips)
        return out
    
    
    def compute_flops(self, x_center: torch.Tensor, x_neighbors: torch.Tensor):
        """
        Tính FLOPs và số params của toàn bộ model MAMBA2
        với input cụ thể (đúng shape, đúng device).

        Usage:
            model = MAMBA2(...)
            x_c = torch.randn(1, 1, 256, 256).cuda()
            x_n = torch.randn(1, K, 1, 256, 256).cuda()
            flops, params = model.compute_flops(x_c, x_n)
        """
        self.eval()
        with torch.no_grad():
            flops = FlopCountAnalysis(self, (x_center, x_neighbors))
            total_flops = flops.total()

        total_params = sum(p.numel() for p in self.parameters())
        return total_flops, total_params