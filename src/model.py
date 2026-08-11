"""
Model zoo for the KLA image restoration challenge.

Primary model: NAFNetRestorer
  - NAFNet blocks (Chen et al., ECCV 2022) with noise-aware input
  - Encoder-decoder at native resolution + 2x pixel-shuffle SR to 256x256
  - Pure ML: LayerNorm, depthwise conv, SimpleGate, channel attention

Legacy models kept for backward-compat with existing checkpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_noise_evidence_map(x):
    """Pixels outside [0,1] are evidence of clipping / extreme noise."""
    return torch.relu(x - 1.0) + torch.relu(-x)


# ── NAFNet components ──────────────────────────────────────────────────────────

class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for BCHW tensors."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    """Split channels in half; use second half to gate the first."""
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    """
    Non-linear Activation Free Block (Chen et al., ECCV 2022).

    Sub-module 1 – spatial mixing:
      LayerNorm -> 1x1 -> DW-3x3 -> SimpleGate -> SCA -> 1x1  (+beta residual)
    Sub-module 2 – channel FFN:
      LayerNorm -> 1x1 -> SimpleGate -> 1x1  (+gamma residual)

    No ReLU/GELU needed — SimpleGate provides the non-linearity.
    Learnable beta/gamma scalars let each block learn its own residual weight.
    """
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw = c * dw_expand
        ff = c * ffn_expand

        self.norm1 = LayerNorm2d(c)
        self.pw1   = nn.Conv2d(c,      dw,      1)
        self.dw    = nn.Conv2d(dw,     dw,      3, padding=1, groups=dw)
        self.sg1   = SimpleGate()
        self.sca   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw // 2, dw // 2, 1),
        )
        self.pw2   = nn.Conv2d(dw // 2, c, 1)
        self.beta  = nn.Parameter(torch.ones(1, c, 1, 1))

        self.norm2 = LayerNorm2d(c)
        self.ff1   = nn.Conv2d(c,      ff,      1)
        self.sg2   = SimpleGate()
        self.ff2   = nn.Conv2d(ff // 2, c, 1)
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1))

    def forward(self, inp):
        # spatial branch
        x = self.norm1(inp)
        x = self.pw1(x)
        x = self.dw(x)
        x = self.sg1(x)
        x = x * self.sca(x)
        x = self.pw2(x)
        y = inp + x * self.beta

        # FFN branch
        x = self.norm2(y)
        x = self.ff1(x)
        x = self.sg2(x)
        x = self.ff2(x)
        return y + x * self.gamma


# ── V2 components: transformer bottleneck for long-range context ───────────────

class MDTA(nn.Module):
    """
    Multi-Dconv Head Transposed Attention (Restormer, Zamir et al., CVPR 2022).

    Attention is computed over the channel dimension instead of the spatial
    dimension, so cost is O(C^2) not O((HW)^2) -- independent of resolution.
    This gives every pixel a global receptive field (useful for periodic /
    repeating structure like wafer patterns or ripples spanning the frame)
    at a cost cheap enough to use inside the network, not just at the
    lowest-resolution bottleneck.
    """
    def __init__(self, c, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.qkv_dw = nn.Conv2d(c * 3, c * 3, 3, padding=1, groups=c * 3)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)

        def split_heads(t):
            return t.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """MDTA + NAFBlock-style gated FFN, pre-norm, learnable residual scales."""
    def __init__(self, c, num_heads=4, ffn_expand=2):
        super().__init__()
        self.norm1 = LayerNorm2d(c)
        self.attn  = MDTA(c, num_heads=num_heads)
        self.beta  = nn.Parameter(torch.ones(1, c, 1, 1))

        self.norm2 = LayerNorm2d(c)
        ff = c * ffn_expand
        self.ff1   = nn.Conv2d(c, ff, 1)
        self.sg    = SimpleGate()
        self.ff2   = nn.Conv2d(ff // 2, c, 1)
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1))

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) * self.beta
        y = self.ff2(self.sg(self.ff1(self.norm2(x))))
        return x + y * self.gamma


# ── Primary model ──────────────────────────────────────────────────────────────

class NAFNetRestorer(nn.Module):
    """
    Noise-aware encoder-decoder with NAFNet blocks + 2x pixel-shuffle SR.

    Input:  (B, 1, H,   W)   raw noisy grayscale  (values may exceed [0,1])
    Output: (B, 1, 2H, 2W)   clean grayscale

    Default config (width=32, enc=(2,2,4), mid=8, dec=(4,2,2)):
      Scales:  H -> H/2 -> H/4 -> H/8 (bottleneck) -> H/4 -> H/2 -> H -> 2H
      ~5 M parameters — strong quality with fast inference.

    Larger config (width=48, enc=(2,2,4), mid=12, dec=(4,2,2)):
      ~11 M parameters — highest quality, slower training.
    """
    def __init__(
        self,
        width=32,
        enc_blks=(2, 2, 4),
        middle_blks=8,
        dec_blks=(4, 2, 2),
    ):
        super().__init__()

        # noise-aware head: clamped image + evidence map -> width channels
        self.head = nn.Conv2d(2, width, 3, padding=1)

        # encoder: each stage processes then strides down
        self.encoders = nn.ModuleList()
        self.downs     = nn.ModuleList()
        ch = width
        for n in enc_blks:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2))  # learnable downsample
            ch *= 2

        # bottleneck
        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blks)])

        # decoder: pixel-shuffle upsample + skip add + NAFBlocks
        self.ups      = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n in dec_blks:
            # Conv(ch -> ch*2) then PixelShuffle(2) -> ch//2 channels at 2x res
            self.ups.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, 1),
                nn.PixelShuffle(2),
            ))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))

        # 2x super-resolution upsample (native res -> 2x output res)
        self.sr_up = nn.Sequential(
            nn.Conv2d(ch, ch * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )

        self.tail = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(self, x_raw):
        evidence = compute_noise_evidence_map(x_raw)
        x = torch.cat([x_raw.clamp(0, 1), evidence], dim=1)
        x = self.head(x)

        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        for dec, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip
            x = dec(x)

        x = self.sr_up(x)
        return self.tail(x)


class NAFNetRestorerV2(nn.Module):
    """
    NAFNetRestorer + three additions aimed at closing the remaining fidelity
    gap, all still pure learned components (no classical denoising/filtering
    step anywhere in the path):

      1. Global residual: the network adds its output on top of a bicubic
         upsample of the input rather than predicting the image from
         scratch. This is the standard residual-learning convention used
         in super-resolution nets (EDSR, RCAN) -- bicubic interpolation
         supplies no restoration on its own (it doesn't remove noise), it
         just gives the network an identity starting point so its limited
         capacity is spent on the noise-removal / detail-recovery residual
         instead of relearning large-scale structure it already has for free.
      2. Transformer bottleneck: TransformerBlock (MDTA channel attention)
         replaces NAFBlock at the lowest-resolution stage, giving the
         network a global receptive field there for long-range / periodic
         structure that local convolutions can't see.
      3. Deep supervision taps: each decoder stage exposes a 1-channel aux
         prediction so a loss can be applied at every scale during training,
         improving gradient flow to early layers. Call forward(x,
         return_aux=True) during training; plain forward(x) at inference
         ignores them.

    Not backward-compatible with NAFNetRestorer checkpoints -- train fresh.
    """
    def __init__(
        self,
        width=48,
        enc_blks=(2, 2, 4),
        middle_blks=12,
        dec_blks=(4, 2, 2),
        num_heads=4,
    ):
        super().__init__()

        self.head = nn.Conv2d(2, width, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for n in enc_blks:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2))
            ch *= 2

        self.middle = nn.Sequential(
            *[TransformerBlock(ch, num_heads=num_heads) for _ in range(middle_blks)]
        )

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.aux_heads = nn.ModuleList()
        for n in dec_blks:
            self.ups.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, 1),
                nn.PixelShuffle(2),
            ))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
            self.aux_heads.append(nn.Conv2d(ch, 1, 3, padding=1))

        self.sr_up = nn.Sequential(
            nn.Conv2d(ch, ch * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.tail = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(self, x_raw, return_aux=False):
        base = F.interpolate(x_raw.clamp(0, 1), scale_factor=2, mode="bicubic", align_corners=False)

        evidence = compute_noise_evidence_map(x_raw)
        x = torch.cat([x_raw.clamp(0, 1), evidence], dim=1)
        x = self.head(x)

        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        aux_outputs = []
        for dec, up, skip, aux_head in zip(self.decoders, self.ups, skips[::-1], self.aux_heads):
            x = up(x)
            x = x + skip
            x = dec(x)
            if return_aux:
                aux_outputs.append(aux_head(x))

        x = self.sr_up(x)
        out = self.tail(x) + base

        if return_aux:
            return out, aux_outputs
        return out


# ── Legacy models (kept for backward-compat with existing checkpoints) ─────────

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


class RestorationNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, num_res_blocks=8):
        super().__init__()
        self.head = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_res_blocks)]
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.ReLU(inplace=True),
        )
        self.tail = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.head(x)
        feat = self.res_blocks(feat)
        feat = self.upsample(feat)
        return self.tail(feat)


class NoiseAwareRestorationNet(nn.Module):
    def __init__(self, out_channels=1, base_channels=64, num_res_blocks=8):
        super().__init__()
        self.head = nn.Conv2d(2, base_channels, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_res_blocks)]
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.ReLU(inplace=True),
        )
        self.tail = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x_raw):
        evidence = compute_noise_evidence_map(x_raw)
        x_clamped = torch.clamp(x_raw, 0, 1)
        x = torch.cat([x_clamped, evidence], dim=1)
        feat = self.head(x)
        feat = self.res_blocks(feat)
        feat = self.upsample(feat)
        return self.tail(feat)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetRestorationNet(nn.Module):
    def __init__(self, base_channels=48):
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(2, c)
        self.down1 = nn.Conv2d(c, c, 4, stride=2, padding=1)
        self.enc2 = ConvBlock(c, c * 2)
        self.down2 = nn.Conv2d(c * 2, c * 2, 4, stride=2, padding=1)
        self.bottleneck = ConvBlock(c * 2, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)
        self.dec1 = ConvBlock(c * 2, c)
        self.final_upsample = nn.Sequential(
            nn.Conv2d(c, c * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.ReLU(inplace=True),
        )
        self.tail = nn.Conv2d(c, 1, kernel_size=3, padding=1)

    def forward(self, x_raw):
        evidence = compute_noise_evidence_map(x_raw)
        x_clamped = torch.clamp(x_raw, 0, 1)
        x = torch.cat([x_clamped, evidence], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        out = self.final_upsample(d1)
        return self.tail(out)
