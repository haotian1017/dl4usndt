# -*- coding: utf-8 -*-

from dataclasses import dataclass, asdict
import json


@dataclass
class Config:
    # ================= 数据 =================
    data_root: str = "./data"          # 数据根目录
    img_dir: str = "images"            # 原图目录（单通道灰度，png/tif/bmp/jpg）
    mask_dir: str = "masks"            # 标注目录，与 image 同名，0=背景 255=缺陷
    img_ext: str = ".png"
    max_cache: int = 0            # 内存中缓存已预处理样本的数量；200 张小图可设 400 加速

    # ================= 模型 =================
    # smp_unet        : segmentation_models_pytorch，编码器可换，推荐主力方案
    # tiny_unet       : 自带轻量 U-Net（depthwise separable，约 1.1M），无外部依赖
    # tiny_unet_nano  : 极简版（约 0.28M），给算力很弱的边缘板卡
    arch: str = "smp_unet"
    encoder: str = "mobilenet_v3_small"   # 备选: efficientnet-b0 / resnet18 / mobilenet_v3_large
    encoder_weights: str = "imagenet"     # 小数据集务必保留预训练
    in_channels: int = 1                  # C-SAM 为单通道灰度图

    # ================= 训练 =================
    patch_size: int = 512         # 训练裁剪尺寸；原图小于此值时自动用全图
    fg_crop_prob: float = 0.75    # 裁剪窗口"必须包含缺陷"的概率（小目标关键）
    epochs: int = 150
    batch_size: int = 8
    lr: float = 1e-3              # decoder / 自带模型 的学习率
    encoder_lr: float = 1e-4      # 预训练编码器用更小的学习率
    weight_decay: float = 1e-4
    amp: bool = True              # 混合精度
    num_workers: int = 4
    folds: int = 5                # k 折交叉验证；设为 1 则按 val_ratio 单次划分
    val_ratio: float = 0.2
    early_stop_patience: int = 30

    # ================= 损失 =================
    # combo: FocalTversky + 带权 BCE（推荐）
    # dice_bce: Dice + BCE（warmup 阶段用）
    # focal_tversky / tversky / dice
    loss: str = "combo"
    loss_warmup_epochs: int = 5   # 前 N 个 epoch 先用 dice_bce 稳住，再切主损失
    # BCE 中缺陷像素的正样本权重。缺陷占比越小，这个值要越大（经验：1/缺陷占比 量级）
    pos_weight: float = 20.0
    # Tversky: TP / (TP + alpha*FP + beta*FN)，alpha+beta=1
    # beta 越大 → 越惩罚漏检 → 召回越高（工业漏检代价大，故 beta > alpha）
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    tversky_gamma: float = 1.3333  # Focal 指数，>1 聚焦难样本
    dice_smooth: float = 1.0

    # ================= 预处理（超声专用）=================
    pct_low: float = 1.0          # 百分位裁剪下界，抗极值
    pct_high: float = 99.0        # 百分位裁剪上界
    use_log: bool = False         # 对数变换（乘性散斑→加性）。设备已做检波对数时保持 False
    denoise: str = "bilateral"    # none | bilateral | srad
    bilateral_d: int = 5
    bilateral_sigma_color: float = 25.0
    bilateral_sigma_space: float = 25.0
    srad_iters: int = 15          # 简化 SRAD 迭代次数
    srad_dt: float = 0.15
    clahe: bool = True            # 局部对比度增强，对低对比度缺陷提升明显
    clahe_clip: float = 2.0       # 别调太大，否则放大散斑噪声
    clahe_tile: int = 8

    # ================= 数据增强 =================
    copy_paste_prob: float = 0.5  # 缺陷粘贴（小数据集性价比最高的一招）
    copy_paste_max: int = 3       # 单张最多粘贴几个缺陷
    elastic_prob: float = 0.3
    speckle_prob: float = 0.3     # 乘性散斑噪声模拟
    speckle_sigma: float = 0.15
    brightness_prob: float = 0.5
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    rot90_p: float = 0.5

    # ================= 推理 =================
    infer_patch: int = 512
    infer_overlap: float = 0.25   # 滑窗重叠率，边界更平滑
    tta: bool = True              # D4 测试时增强
    prob_thresh: float = 0.5      # 概率阈值（可用 val 集搜索）
    min_area_px: int = 20         # 图像级"有/无"判定：小于该面积的连通域视为噪声
    min_area_ratio: float = 0.0   # 也可用占比：连通域面积 / 图像面积 的下限
    out_dir: str = "./outputs"

    seed: int = 42

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return Config(**d)


def cfg_from_args(args, cls=Config):
    """把 argparse 的 Namespace 里非 None 的字段覆盖到 Config 上"""
    cfg = cls()
    for k, v in vars(args).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
