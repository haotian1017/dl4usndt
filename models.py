# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================================
# 自带轻量模型（不依赖 segmentation_models_pytorch）
# ==================================================================
class DSConv(nn.Module):
    """深度可分离卷积：参数量和计算量约为标准卷积的 1/8~1/9"""

    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, stride, padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = DSConv(in_ch, out_ch)
        self.c2 = DSConv(out_ch, out_ch)

    def forward(self, x):
        return self.c2(self.c1(x))


class AttentionGate(nn.Module):
    """跳跃连接上的注意力门：抑制背景特征，突出缺陷相关的浅层细节"""

    def __init__(self, g_ch, x_ch, mid_ch):
        super().__init__()
        self.wg = nn.Conv2d(g_ch, mid_ch, 1, bias=False)
        self.wx = nn.Conv2d(x_ch, mid_ch, 1, bias=False)
        self.psi = nn.Sequential(nn.Conv2d(mid_ch, 1, 1, bias=False), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        a = self.relu(self.wg(g) + self.wx(x))
        return x * self.psi(a)


class TinyUNet(nn.Module):
    """轻量 U-Net：depthwise separable + 可选 attention gate。

    base=32 -> 约 1.1M 参数
    base=16 -> 约 0.28M 参数（nano）
    """

    def __init__(self, in_channels=1, base=32, depth=4, use_ag=True, out_channels=1):
        super().__init__()
        ch = [base * (2 ** i) for i in range(depth)]
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        c_in = in_channels
        for c in ch:
            self.downs.append(Block(c_in, c))
            c_in = c

        self.bottleneck = Block(ch[-1], ch[-1] * 2)

        self.ups = nn.ModuleList()
        self.ags = nn.ModuleList() if use_ag else None
        for i in reversed(range(depth)):
            up_in = ch[-1] * 2 if i == depth - 1 else ch[i + 1]
            self.ups.append(nn.ConvTranspose2d(up_in, ch[i], 2, stride=2))
            if use_ag:
                self.ags.append(AttentionGate(ch[i], ch[i], max(ch[i] // 4, 8)))

        self.heads = nn.ModuleList()
        for i in reversed(range(depth)):
            self.heads.append(Block(ch[i] * 2, ch[i]))

        self.final = nn.Conv2d(ch[0], out_channels, 1)
        self.use_ag = use_ag

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)

        for i, (up, head) in enumerate(zip(self.ups, self.heads)):
            x = up(x)
            s = skips[-(i + 1)]
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            if self.use_ag:
                s = self.ags[i](x, s)
            x = head(torch.cat([x, s], dim=1))
        return self.final(x)


# ==================================================================
# 统一入口
# ==================================================================
def build_model(cfg):
    """按配置构建模型。返回 (model, is_external_encoder)"""
    arch = cfg.arch.lower()

    if arch == "smp_unet":
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise ImportError(
                "arch=smp_unet 需要安装 segmentation_models_pytorch：\n"
                "  pip install segmentation-models-pytorch\n"
                "若不想装依赖，请把 arch 改成 tiny_unet（自带轻量 U-Net）"
            )
        model = smp.Unet(
            encoder_name=cfg.encoder,
            encoder_weights=cfg.encoder_weights,
            in_channels=cfg.in_channels,
            classes=1,
            activation=None,       # 输出 logits，配合 BCEWithLogits / Tversky
        )
        return model, True

    if arch == "tiny_unet":
        return TinyUNet(cfg.in_channels, base=32, use_ag=True), False
    if arch == "tiny_unet_nano":
        return TinyUNet(cfg.in_channels, base=16, use_ag=False), False

    raise ValueError(f"未知 arch: {cfg.arch}")


def param_groups(model, cfg, external_encoder=True):
    """预训练编码器用更小的学习率，decoder 用大学习率"""
    if not external_encoder:
        return [{"params": model.parameters(), "lr": cfg.lr}]
    enc = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc}
    dec = [p for p in model.parameters() if id(p) not in enc_ids]
    return [
        {"params": enc, "lr": cfg.encoder_lr},
        {"params": dec, "lr": cfg.lr},
    ]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    from config import Config
    cfg = Config()
    for a in ["tiny_unet", "tiny_unet_nano"]:
        c = Config(arch=a)
        m, _ = build_model(c)
        print(f"{a}: {count_params(m)/1e6:.3f} M 参数")
