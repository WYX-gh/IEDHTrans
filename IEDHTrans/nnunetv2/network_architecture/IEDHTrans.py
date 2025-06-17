import torch
import torch.nn as nn
from timm.models.layers import DropPath
import math
import triton
import triton.language as tl
from typing import Optional, Union
from flash_attn import flash_attn_func


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
    


@triton.jit
def rotary_kernel(
    OUT,  
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,  
    # Matrix dimensions
    seqlen,
    nheads,
    rotary_dim,
    seqlen_ro,
    CACHE_KEY_SEQLEN,
    # strides
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    # Meta-parameters
    BLOCK_K: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2

    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        OUT = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
    rk = tl.arange(0, BLOCK_K)
    rk_half = tl.arange(0, BLOCK_K // 2)

    if not INTERLEAVED:
        # Load the 1st and 2nd halves of X, do calculation, then store to 1st and 2nd halves of OUT
        X = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        cos = tl.load(
            COS, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=1.0
        ).to(tl.float32)
        sin = tl.load(
            SIN, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=0.0
        ).to(tl.float32)
        x0 = tl.load(
            X, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0
        ).to(tl.float32)
        x1 = tl.load(
            X + rotary_dim_half * stride_x_headdim,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        # write back result
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim)
        tl.store(OUT, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
        tl.store(
            OUT + rotary_dim_half * stride_out_headdim,
            o1,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
        )
    else:
        # We don't want to load X[0, 2, 4, ...] and X[1, 3, 5, ...] separately since both are slow.
        # Instead, we load x0 = X[0, 1, 2, 3, ...] and x1 = X[1, 0, 3, 2, ...].
        # Loading x0 will be fast but x1 will be slow.
        # Then we load cos = COS[0, 0, 1, 1, ...] and sin = SIN[0, 0, 1, 1, ...].
        # Then we do the calculation and use tl.where to pick put the right outputs for the even
        # and for the odd indices.
        rk_swap = rk + ((rk + 1) % 2) * 2 - 1  # 1, 0, 3, 2, 5, 4, ...
        rk_repeat = tl.arange(0, BLOCK_K) // 2
        X0 = X + (rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim)
        X1 = X + (rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        cos = tl.load(
            COS,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            SIN,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        x0 = tl.load(X0, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim), other=0.0).to(
            tl.float32
        )
        x1 = tl.load(
            X1, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim), other=0.0
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        x0_cos = x0 * cos
        x1_sin = x1 * sin
        out = tl.where(rk[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim)
        tl.store(OUT, out, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim))


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> torch.Tensor:
    """
    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer or integer tensor of size (batch,)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Returns:
        y: (batch, seqlen, nheads, headdim)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None, "If cu_seqlens is passed in, then max_seqlen must be passed"
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "Only support headdim <= 256"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

    assert (
        cos.dtype == sin.dtype
    ), f"cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}"
    assert (
        x.dtype == cos.dtype
    ), f"Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}"

    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    BLOCK_K = (
        32
        if rotary_dim <= 32
        else (64 if rotary_dim <= 64 else (128 if rotary_dim <= 128 else 256))
    )
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch, nheads)  # noqa
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 64 else 4)

    # Need this, otherwise Triton tries to launch from cuda:0 and we get
    # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    with torch.cuda.device(x.device.index):
        rotary_kernel[grid](
            output,  # data ptrs
            x,
            cos,
            sin,
            cu_seqlens,
            seqlen_offsets,
            seqlen,  # shapes
            nheads,
            rotary_dim,
            seqlen_ro,
            seqlen // 128,  # key for triton cache (limit number of compilations)
            output.stride(0) if not is_varlen else 0,  # batch_strides if not varlen else 0
            output.stride(-3),  # seqlen_stride or total_seqlen_stride
            output.stride(-2),  # nheads_stride
            output.stride(-1),  # headdim_stride
            x.stride(0) if not is_varlen else 0,  # batch_strides if not varlen else 0
            x.stride(-3),  # seqlen stride or total_seqlen_stride
            x.stride(-2),  # nheads stride
            x.stride(-1),  # headdim stride
            BLOCK_K,
            isinstance(seqlen_offsets, torch.Tensor),
            is_varlen,
            interleaved,
            conjugate,
            BLOCK_M,
        )
    return output

class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        out = apply_rotary(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=inplace,
        )
        if isinstance(seqlen_offsets, int):
            # Can't save int with save_for_backward
            ctx.save_for_backward(cos, sin, cu_seqlens)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        # TD [2023-09-02]: For some reason Triton (2.0.0.post1) errors with
        # "[CUDA]: invalid device context", and cloning makes it work. Idk why. Triton 2.1.0 works.
        if not ctx.interleaved and not ctx.inplace:
            do = do.clone()
        dx = apply_rotary(
            do,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=ctx.interleaved,
            inplace=ctx.inplace,
            conjugate=True,
        )
        return dx, None, None, None, None, None, None, None


def apply_rotary_emb(
    x, cos, sin, interleaved=False, inplace=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None
):
  
    cos = cos.to(dtype=x.dtype).contiguous()
    sin = sin.to(dtype=x.dtype).contiguous()
    
    return ApplyRotaryEmb.apply(
        x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
    )

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)



class MultiheadDiffAttn(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads=8,
        num_kv_heads=None,
        qkv_bias=True,
        attn_drop=0.,
        proj_drop=0.
    ):
        super().__init__()
        self.embed_dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = dim // num_heads
        self.attn_drop_rate = attn_drop
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.subln = RMSNorm(dim, eps=1e-5)

    def forward(self, x, cos=None, sin=None):
        B, N, C = x.shape

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            x = x.to(torch.bfloat16)
            
            q = self.q_proj(x)  # [B, N, C]
            k = self.k_proj(x)  # [B, N, num_kv_heads*head_dim]
            v = self.v_proj(x)  # [B, N, num_kv_heads*head_dim]

            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_kv_heads, self.head_dim)
            v = v.view(B, N, self.num_kv_heads, self.head_dim)

            q = apply_rotary_emb(q, cos, sin, interleaved=True)
            k = apply_rotary_emb(k, cos, sin, interleaved=True)

            q = q.transpose(1, 2)
            k = repeat_kv(k.transpose(1, 2), self.n_rep)
            v = repeat_kv(v.transpose(1, 2), self.n_rep)
            
            q1, q2 = torch.chunk(q, 2, dim=-1)
            k1, k2 = torch.chunk(k, 2, dim=-1)
            v1, v2 = torch.chunk(v, 2, dim=-1)

            dropout_p = self.attn_drop_rate if self.training else 0.0
            softmax_scale = float(self.scaling) 

            attn11 = flash_attn_func(
                q1, k1, v1, 
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=True
            )
            attn12 = flash_attn_func(
                q1, k1, v2,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=True
            )
            attn1 = torch.cat([attn11, attn12], dim=-1)
            
            attn21 = flash_attn_func(
                q2, k2, v1,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=True
            )
            attn22 = flash_attn_func(
                q2, k2, v2,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=True
            )
            attn2 = torch.cat([attn21, attn22], dim=-1)

            lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
            lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
            attn = attn1 - (lambda_1 - lambda_2 + self.lambda_init) * attn2

        attn = attn.to(x.dtype)
        
        x = attn.transpose(1, 2).reshape(B, N, C)
        x = self.subln(x)
        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, depth, mlp_ratio=4,  
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = MultiheadDiffAttn(  
            dim=dim, 
            depth=depth,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=drop,
        )

    def forward(self, x, cos=None, sin=None):  
        x = x + self.drop_path(self.attn(self.norm1(x), cos, sin)) 
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    


class ResidualConvBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.InstanceNorm3d(planes)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.InstanceNorm3d(planes)
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out
    

class CBAM3D(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        # Channel Attention
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction_ratio, 1),
            nn.ReLU(),
            #nn.Dropout3d(0.1),  
            nn.Conv3d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.spatial_gate = nn.Sequential(
            nn.Conv3d(channels, 1, 1),
            nn.InstanceNorm3d(1),
            nn.ReLU(),
            #nn.Dropout3d(0.1), 
            nn.Conv3d(1, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel-wise weighting
        channel_weight = self.channel_gate(x)
        x = x * channel_weight
        
        # Spatial-wise weighting
        spatial_weight = self.spatial_gate(x)
        return x * spatial_weight

    
class MPFI(nn.Module):
    def __init__(self, cx1, cx2, c_out):
        super().__init__()
        self.conv1 = nn.Conv3d(cx1, c_out//2, kernel_size=1)  # cx1 → c_out/2
        self.conv2 = nn.Conv3d(cx2, c_out//2, kernel_size=1)  # cx2 → c_out/2
        
        self.cbam = CBAM3D(c_out) 
        
        self.residual_block = nn.Sequential(
            nn.Conv3d(c_out, c_out, 3, padding=1),
            nn.InstanceNorm3d(c_out),
            nn.LeakyReLU(0.2),
            nn.Conv3d(c_out, c_out, 3, padding=1),
            nn.InstanceNorm3d(c_out),
            nn.LeakyReLU(0.2)
        )
        
        self.out_conv = nn.Conv3d(c_out, c_out, kernel_size=1)
        self.conv_x2 = nn.Conv3d(c_out, cx2, kernel_size=1)
        self.conv_x1 = nn.Conv3d(c_out, cx1, kernel_size=1)

    def forward(self, x1, x2):

        original_x1, original_x2 = x1, x2
        

        x1_calibrated = self.conv1(x1)  # cx1 → c_out/2
        x2_calibrated = self.conv2(x2)  # cx2 → c_out/2
        

        fused = torch.cat([x1_calibrated, x2_calibrated], dim=1)  # c_out/2 + c_out/2 = c_out
        
        out = self.residual_block(fused)  # 输出通道=c_out
        
        fuse_a = self.cbam(out)  # 保持通道数=c_out

        y = self.conv_x2(fuse_a)  # c_out → cx2
        x2_new = y + original_x2  # 保持原始通道数
        
        x = self.conv_x1(fuse_a)  # c_out → cx1
        x1_new = x + original_x1  # 保持原始通道数
        
        return [x1_new, x2_new]




def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1), 
        nn.InstanceNorm3d(out_dim),
        activation,
        ResidualConvBlock(out_dim, out_dim, stride=1),  
        activation
    )

def conv_3d_NoDown(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, 3,1,1),  
        nn.InstanceNorm3d(out_dim),
        activation,
        ResidualConvBlock(out_dim, out_dim, stride=1), 
        activation
    )

def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=(1, 2, 2), padding=1), 
        nn.InstanceNorm3d(out_dim),
        activation,
        ResidualConvBlock(in_dim, out_dim, stride=1),
        activation
    )


def conv_trans_block_z_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_dim),
        activation,
        #nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=False)
        nn.ConvTranspose3d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.InstanceNorm3d(out_dim),
        activation,
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_dim),
        activation
        )

def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_dim),
        activation,
        nn.ConvTranspose3d(out_dim, out_dim, kernel_size=3, stride=(1,2,2), padding=1, output_padding=(0,1,1)), 
        #nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)
        nn.InstanceNorm3d(out_dim),
        activation,
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_dim),
        activation
        )


class TokenSeg(nn.Module):
    def __init__(self, inch=2, outch=2,  base_channel=32, hidden_size=256, imgsize=[64,192,192], TransformerLayerNum=3):
        super().__init__()
        self.base_channel = base_channel
        self.hidden_size = hidden_size
        self.imgsize = imgsize
        activation = nn.LeakyReLU(0.2)
        self.outch=outch

        self.modalities = inch 
        # encoder1
        self.encoder1_layer1 = conv_block_2_3d(self.base_channel, self.base_channel, activation)
        self.encoder1_layer2 = conv_block_3d(self.base_channel*2, self.base_channel * 4, activation)
        self.encoder1_layer3 = conv_block_3d(self.base_channel * 8, self.base_channel * 16, activation)

        # encoder2
        self.encoder2_layer1 = conv_block_2_3d(self.base_channel, self.base_channel, activation)
        self.encoder2_layer2 = conv_block_3d(self.base_channel*2, self.base_channel * 4, activation)
        self.encoder2_layer3 = conv_block_3d(self.base_channel * 8, self.base_channel * 16, activation)

         # DHFormer
        self.transformer_blocks = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(
                    dim=hidden_size, 
                    num_heads=8, 
                    depth=i,  
                    mlp_ratio=4,
                    drop_path=0.1
                ) for _ in range(TransformerLayerNum)
            ]) for i in range(3)  
        ])



        self.mlp_adjust_c1 = nn.Linear(base_channel*32, hidden_size)
        self.mlp_adjust_c2 = nn.Linear(base_channel*32, hidden_size)
        self.mlp_adjust_c3 = nn.Linear(base_channel*32, hidden_size)


        self.up_c2 = nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=False)
        self.up_c3 = nn.Upsample(scale_factor=(4,4,4), mode='trilinear', align_corners=False)


        self.fusion = nn.Sequential(
            nn.Conv3d(hidden_size*3, 256, kernel_size=1),
            nn.InstanceNorm3d(256), 
            activation)
        

        self.downsample1 = conv_block_3d(self.base_channel*32, self.base_channel*32, activation)
        self.downsample2 = conv_block_3d(self.base_channel*32, self.base_channel*32, activation)
          

        self.conv_3d_NoDown1 = conv_3d_NoDown(1, self.base_channel, activation)#用于对原始图像进行卷积处理
        self.conv_3d_NoDown2 = conv_3d_NoDown(1, self.base_channel, activation)#用于对原始图像进行卷积处理
        self.conv_3d_NoDown3 = conv_3d_NoDown(self.base_channel * 3 , self.base_channel * 1 , activation)#用于最后output前的卷积


        self.trans_1 = conv_trans_block_z_3d(self.base_channel*8, self.base_channel * 4, activation)
        self.trans_2 = conv_trans_block_z_3d(self.base_channel * 12, self.base_channel * 4, activation)
        self.trans_3 = conv_trans_block_3d(self.base_channel * 4 + self.base_channel * 2, self.base_channel, activation)# +self.base_channeel * 2以满足x1_layer1和x2_layer1的通道数
        self.out = nn.Conv3d(self.base_channel * 1, self.outch,  kernel_size=1)
                                 


        #MPFI
        self.mpfi_1 = MPFI(self.base_channel, self.base_channel , self.base_channel * 2)
        self.mpfi_2 = MPFI(self.base_channel * 4, self.base_channel * 4, self.base_channel * 8)


        self.decoder_out1=nn.Conv3d(self.base_channel * 4 + self.base_channel * 2, self.outch, 3,1,1)
        self.decoder_out2=nn.Conv3d(self.base_channel * 12, self.outch, 3,1,1)
        self.decoder_out3=nn.Conv3d(self.base_channel * 8 , self.outch, 3,1,1)
        self.decoder_out4=nn.Conv3d(self.base_channel * 8 , self.outch , 3,1,1)


    def transformer_layer(self, feature, transformer_blocks, mlp_layer):
        B, C, D, H, W = feature.shape
        x_seq = feature.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]
        mlp_layer = mlp_layer.to(feature.device)
        x_seq = mlp_layer(x_seq) 

        seqlen = D * H * W
        rotary_dim = self.hidden_size // self.transformer_blocks[0][0].attn.num_heads

        theta = 1.0 / (10000 ** torch.linspace(0, 1, rotary_dim//2, device=feature.device, dtype=feature.dtype))
        pos = torch.arange(seqlen, device=feature.device, dtype=feature.dtype)[:, None]

        cos = torch.cos(pos * theta[None, :])
        sin = torch.sin(pos * theta[None, :])

        for block in transformer_blocks:
            x_seq = block(x_seq, cos, sin) 
                
        return x_seq.permute(0, 2, 1).view(B, -1, D, H, W)
       
    def forward(self, x):

        modality1, modality2 = torch.chunk(x, 2, dim=1)

        modality1_conv = self.conv_3d_NoDown1(modality1)
        modality2_conv = self.conv_3d_NoDown2(modality2)


        x1 = self.encoder1_layer1(modality1_conv)
        x1_layer1 = x1  #  channel 32, [D, H/2, W/2]

        x2 = self.encoder2_layer1(modality2_conv)
        x2_layer1 = x2  #  channel 32, [D, H/2, W/2]


        mpfi1 = self.mpfi_1(x1_layer1,x2_layer1)
        x1 = torch.cat([x1,mpfi1[0]],dim=1)
        x2 = torch.cat([x2,mpfi1[1]],dim=1)


        x1 = self.encoder1_layer2(x1)
        x1_layer2 = x1  # channel 128, [D/2, H/4, W/4]

        x2 = self.encoder2_layer2(x2)
        x2_layer2 = x2  #  channel 128, [D/2, H/4, W/4]


        mpfi2 = self.mpfi_2(x1_layer2,x2_layer2)
        x1 = torch.cat([x1,mpfi2[0]],dim=1)
        x2 = torch.cat([x2,mpfi2[1]],dim=1)

        x1 = self.encoder1_layer3(x1)
        x1_layer3 = x1  #   channel 512, [D/4, H/8, W/8]

        x2 = self.encoder2_layer3(x2)
        x2_layer3 = x2  # channel 512, [D/4, H/8, W/8]


        x1 = torch.cat([x1_layer3, x2_layer3], dim=1)  # channel 1024, [D/4, H/8, W/8]
        x2 = self.downsample1(x1)# channel 1024, [D/8, H/16, W/16]
        x3 = self.downsample2(x2) # channel 1024, [D/16, H/32, W/32]


        c1 = self.transformer_layer(x1, self.transformer_blocks[0], self.mlp_adjust_c1 ) # channel 256 ,[D/4,H/8,W/8]
        c2 = self.transformer_layer(x2, self.transformer_blocks[1], self.mlp_adjust_c2  )# channel 256 ,[D/8,H/16,W/16]
        c3 = self.transformer_layer(x3, self.transformer_blocks[2], self.mlp_adjust_c3  )# channel 256 ,[D/16,H/32,W/32]
        out4=c2


        c2 = self.up_c2(c2)   # channel 256 [D/4,H/8,W/8]
        c3 = self.up_c3(c3)   # channel 256 [D/4,H/8,W/8]


        fused = torch.cat([c1, c2, c3], dim=1)
        trans_fused_feature = self.fusion(fused) # channel 256, [D/4, H/8, W/8]
        x=trans_fused_feature # # channel 256, [D/4, H/8, W/8]
        out3 = x
        


        x = self.trans_1(x)  # channel 128, [D/2, H/4, W/4]
        x = torch.cat([x, x1_layer2, x2_layer2], dim=1)    channel 128*3, [D/2, H/4, W/4]
        out2 = x


        x = self.trans_2(x)  # channel 128, [D, H/2, W/2]
        x = torch.cat([x, x1_layer1, x2_layer1], dim=1)  #  channel 128+64 ， [D, H/2, W/2]
        out1 = x


        x = self.trans_3(x)  # channel 32, [D, H, W]
        x = torch.cat([x, modality1_conv, modality2_conv], dim=1)  #   channel 96, [D, H, W]
        x = self.conv_3d_NoDown3(x)  # channel 32, [D, H, W]

        x = self.out(x)

        x1=self.decoder_out1(out1)
        x2=self.decoder_out2(out2)
        x3=self.decoder_out3(out3)
        x4=self.decoder_out4(out4)
       
        return {
        'seg_output': [x, x1, x2, x3, x4],
        'features': {
            'cross_modality': [
                (x1_layer1, x2_layer1),
                (x1_layer2, x2_layer2),
                (x1_layer3, x2_layer3),
            ],
            'cross_scale': {
                'modality1': [
                    (x1_layer1, x1_layer2),
                    (x1_layer2, x1_layer3),
                ],
                'modality2': [
                    (x2_layer1, x2_layer2),
                    (x2_layer2, x2_layer3),
                ]
            }
        }
    }
