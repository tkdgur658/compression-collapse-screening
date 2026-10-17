"""Predictive boundary estimator network (PBE-Net).

Clean re-implementation of the legacy `Phase57_LapSkipSup` + DW-Sep bottleneck
(agent_T1X_model_arch.py, agent_T1X_model_phase44.py, models/RJWNet_MSP3_SymB*.py).
Module and parameter names are kept identical so legacy checkpoints load without remapping;
tools/verify_legacy.py checks that outputs match the legacy code bit for bit.

Architecture (c_list = [128, 192, 256, 384, 512] for the base tier):
  encoder stage i : [1x1 conv] -> RepConv(3x3 DW) -> (eBPB at stages 1-2) -> PRCM gate
  bottleneck      : 7x7 DW conv + BN + ReLU + 1x1 conv + BN + ReLU
  skips           : s_i = e_i - Up(W_i * e_{i+1}) for i = 1..3 (W_i zero-init), s_4 = e_4
  decoder         : plain stages, additive skips, deep-supervision heads gt_conv1-4
  train-only heads: shead_i (1x1, C_i -> 1) on s_1..s_3 for the matching loss
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RepConv(nn.Module):
    """k x k depthwise conv + 1x1 branch + identity BN branch (structural re-parameterisation)."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv_kxk = nn.Conv2d(channels, channels, kernel_size, 1, kernel_size // 2, groups=channels, bias=False)
        self.bn_kxk = nn.BatchNorm2d(channels)
        self.conv_1x1 = nn.Conv2d(channels, channels, 1, 1, 0, groups=channels, bias=False)
        self.bn_1x1 = nn.BatchNorm2d(channels)
        self.bn_identity = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn_kxk(self.conv_kxk(x))
        out += self.bn_1x1(self.conv_1x1(x))
        out += self.bn_identity(x)
        return self.activation(out)


class PRCM(nn.Module):
    """Channel gate from global context projected on a small learned basis."""

    def __init__(self, channels, num_basis=8, dropout_rate=0.5):
        super().__init__()
        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        self.fuser = nn.Linear(num_basis, channels, bias=False)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        coeff = self.coeff_dropout(x.mean(dim=[2, 3]) @ self.basis.t())
        return x * self.fuser(coeff).sigmoid()[..., None, None]


class EBPB(nn.Module):
    """Efficient band-preserving block: B(x) = H x + Up(psi(P x)), P = avg-pool stride r."""

    def __init__(self, channels, r=4):
        super().__init__()
        self.pool4 = nn.AvgPool2d(r, r)
        self.branch1 = nn.Identity()  # high band H x, kept by identity
        self.branch3 = nn.Sequential(  # psi on the low band
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        lo = self.pool4(x)
        high = x - F.interpolate(lo, size=size, mode='bilinear', align_corners=False)
        return self.branch1(high) + F.interpolate(self.branch3(lo), size=size, mode='bilinear', align_corners=False)


class InvResBlock(nn.Module):
    """MobileNetV2 inverted residual at full resolution (block-ablation control for EBPB)."""

    def __init__(self, channels, t=0.5):
        super().__init__()
        h = max(8, int(channels * t))
        self.b = nn.Sequential(
            nn.Conv2d(channels, h, 1, bias=False), nn.BatchNorm2d(h), nn.ReLU6(True),
            nn.Conv2d(h, h, 3, padding=1, groups=h, bias=False), nn.BatchNorm2d(h), nn.ReLU6(True),
            nn.Conv2d(h, channels, 1, bias=False), nn.BatchNorm2d(channels))

    def forward(self, x):
        return x + self.b(x)


class DWSepBottleneck(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 7, padding=3, groups=ch, bias=False); self.bn1 = nn.BatchNorm2d(ch)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False); self.bn2 = nn.BatchNorm2d(ch); self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn2(self.pw(self.act(self.bn1(self.dw(x))))))


def init_weights(m):
    """Legacy RJWNet initialisation (applied before the band blocks and bottleneck are attached)."""
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)


def stage(in_ch, out_ch, band=False, num_basis=8, dropout_rate=0.5):
    layers = [nn.Conv2d(in_ch, out_ch, 1, bias=False)] if in_ch != out_ch else []
    layers.append(RepConv(out_ch, 3))
    if band:
        layers.append(nn.Identity())  # placeholder, replaced by the band block after init
    layers.append(PRCM(out_ch, num_basis, dropout_rate))
    return nn.Sequential(nn.Sequential(*layers))


class Base(nn.Module):
    def __init__(self, c, num_classes=1):
        super().__init__()
        self.encoder1 = stage(3, c[0], band=True); self.encoder2 = stage(c[0], c[1], band=True)
        self.encoder3 = stage(c[1], c[2]); self.encoder4 = stage(c[2], c[3]); self.encoder5 = stage(c[3], c[4])
        self.gt_conv1 = nn.Conv2d(c[3], num_classes, 1); self.gt_conv2 = nn.Conv2d(c[2], num_classes, 1)
        self.gt_conv3 = nn.Conv2d(c[1], num_classes, 1); self.gt_conv4 = nn.Conv2d(c[0], num_classes, 1)
        self.decoder1 = stage(c[4], c[3]); self.decoder2 = stage(c[3], c[2])
        self.decoder3 = stage(c[2], c[1]); self.decoder4 = stage(c[1], c[0])
        self.final = nn.Conv2d(c[0], num_classes, 1)
        self.apply(init_weights)
        self.bottleneck = DWSepBottleneck(c[4])


class PBENet(nn.Module):
    """skip: 'predictive' (s_i = e_i - Up(W_i e_{i+1})) or 'plain' (s_i = e_i).
    block: 'ebpb' or 'mbv2' at encoder stages 1-2."""

    def __init__(self, c_list=(128, 192, 256, 384, 512), skip='predictive', block='ebpb', num_classes=1):
        super().__init__()
        c = list(c_list)
        self.skip = skip
        self.base = Base(c, num_classes)
        make = {'ebpb': EBPB, 'mbv2': InvResBlock}[block]
        self.base.encoder1[0][2] = make(c[0]); self.base.encoder2[0][2] = make(c[1])
        self.proj1 = nn.Conv2d(c[1], c[0], 1, bias=False)
        self.proj2 = nn.Conv2d(c[2], c[1], 1, bias=False)
        self.proj3 = nn.Conv2d(c[3], c[2], 1, bias=False)
        for m in (self.proj1, self.proj2, self.proj3):
            nn.init.zeros_(m.weight)
        self.shead1 = nn.Conv2d(c[0], 1, 1); self.shead2 = nn.Conv2d(c[1], 1, 1); self.shead3 = nn.Conv2d(c[2], 1, 1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, x, return_aux=False):
        b = self.base
        e1 = F.max_pool2d(b.encoder1(x), 2); e2 = F.max_pool2d(b.encoder2(e1), 2)
        e3 = F.max_pool2d(b.encoder3(e2), 2); e4 = F.max_pool2d(b.encoder4(e3), 2)
        e5 = b.bottleneck(b.encoder5(e4))
        if self.skip == 'predictive':
            s1 = e1 - self._up(self.proj1(e2), e1); s2 = e2 - self._up(self.proj2(e3), e2)
            s3 = e3 - self._up(self.proj3(e4), e3)
        else:
            s1, s2, s3 = e1, e2, e3
        s4 = e4
        up = lambda t: F.interpolate(t, scale_factor=2, mode='bilinear', align_corners=True)
        d4 = b.decoder1(e5) + s4
        d3 = up(b.decoder2(d4)) + s3
        d2 = up(b.decoder3(d3)) + s2
        d1 = up(b.decoder4(d2)) + s1
        out = up(b.final(d1))
        if not return_aux:
            return out
        hw = x.shape[-2:]
        side = [F.interpolate(g(d), hw, mode='bilinear', align_corners=True)
                for g, d in ((b.gt_conv1, d4), (b.gt_conv2, d3), (b.gt_conv3, d2), (b.gt_conv4, d1))]
        side = (side[0], side[0], side[1], side[2], side[3])  # legacy GT tuple order and weights
        heads = [self.shead1(s1), self.shead2(s2), self.shead3(s3)]
        return out, side, heads
