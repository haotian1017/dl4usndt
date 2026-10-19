# -*- coding: utf-8 -*-

import numpy as np
import cv2

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def read_gray(path):
    """读取单通道灰度图，保持原始位深返回 float32"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def read_mask(path, thresh=127):
    """读取标注 mask，返回 0/1 的 float32"""
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"无法读取标注: {path}")
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return (m > thresh).astype(np.float32)


def robust_normalize(img, pct_low=1.0, pct_high=99.0):
    """百分位裁剪 + min-max 归一化到 [0,1]，抗极值、抗增益漂移"""
    lo, hi = np.percentile(img, pct_low), np.percentile(img, pct_high)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    out = (img - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def log_transform(img, eps=1e-6):
    """对数变换：把乘性散斑变成加性噪声，便于后续滤波。

    注意：多数 C-SAM 设备输出的灰度图已经做过对数检波，此时再取对数反而压平对比度，
    默认关闭，只有当你的原始图是线性幅度回波时才打开。
    """
    x = np.clip(img, eps, None)
    x = np.log1p(x)
    return robust_normalize(x)


def bilateral_denoise(img01, d=5, sigma_color=25.0, sigma_space=25.0):
    """双边滤波：保边去噪。sigma_color 基于 0-255 尺度，故先放大"""
    u8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    out = cv2.bilateralFilter(u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return out.astype(np.float32) / 255.0


def srad(img01, n_iter=15, dt=0.15):
    """简化版 SRAD（Speckle Reducing Anisotropic Diffusion）。

    各向异性扩散：在均匀区平滑（抑制散斑），在梯度大的边缘处停止扩散（保住缺陷边界）。
    扩散系数用局部梯度统计自适应估计，不需要手调阈值。
    """
    I = np.clip(img01, 0, 1).astype(np.float32).copy()
    for _ in range(n_iter):
        # 四邻域差分（边界用 edge 复制等效的 roll 后修正）
        dN = np.roll(I, 1, axis=0) - I
        dS = np.roll(I, -1, axis=0) - I
        dE = np.roll(I, -1, axis=1) - I
        dW = np.roll(I, 1, axis=1) - I
        dN[0, :] = dS[-1, :] = 0.0
        dE[:, -1] = dW[:, 0] = 0.0

        # 用当前图像的梯度尺度自适应设定扩散门限 q0（替代手工阈值）
        g = np.sqrt(dN ** 2 + dS ** 2 + dE ** 2 + dW ** 2)
        q0 = np.percentile(g, 75)
        q0_sq = max(q0 ** 2, 1e-8)

        cN = np.exp(-(dN ** 2) / q0_sq)
        cS = np.exp(-(dS ** 2) / q0_sq)
        cE = np.exp(-(dE ** 2) / q0_sq)
        cW = np.exp(-(dW ** 2) / q0_sq)

        I = I + dt * (cN * dN + cS * dS + cE * dE + cW * dW)
    return np.clip(I, 0, 1).astype(np.float32)


def apply_clahe(img01, clip=2.0, tile=8):
    """限制对比度直方图均衡：拉开弱缺陷的低对比度细节。clip 越大噪声越明显"""
    u8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    return clahe.apply(u8).astype(np.float32) / 255.0


def preprocess(img_raw, cfg=None, is_mask=False):
    """完整预处理流水线。

    Args:
        img_raw: 原始灰度图（float32，任意位深）
        cfg: Config；为 None 时使用默认参数
        is_mask: True 时跳过（mask 不做任何强度变换）
    Returns:
        float32 [0,1] 图像
    """
    if is_mask:
        return (img_raw > 127).astype(np.float32)
    if cfg is None:
        from config import Config
        cfg = Config()

    x = robust_normalize(img_raw, cfg.pct_low, cfg.pct_high)
    if cfg.use_log:
        x = log_transform(x)
    if cfg.denoise == "bilateral":
        x = bilateral_denoise(x, cfg.bilateral_d, cfg.bilateral_sigma_color,
                              cfg.bilateral_sigma_space)
    elif cfg.denoise == "srad":
        x = srad(x, cfg.srad_iters, cfg.srad_dt)
    if cfg.clahe:
        x = apply_clahe(x, cfg.clahe_clip, cfg.clahe_tile)
    return x.astype(np.float32)


def normalize_for_model(x01, mean=None, std=None):
    """送入网络前的标准化。默认用 ImageNet 统计量的一半强度（灰度图方差更小）"""
    if mean is None:
        mean, std = 0.449, 0.226
    return ((x01 - mean) / std).astype(np.float32)


def to_3ch(x01):
    """单通道复制成三通道，用于吃 ImageNet 预训练权重"""
    return np.repeat(x01[None, ...], 3, axis=0)
