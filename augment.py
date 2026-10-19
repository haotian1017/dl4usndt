# -*- coding: utf-8 -*-

import numpy as np
import cv2


# ---------------------------------------------------------------- 基础几何
def random_d4(img, mask, rng):
    """随机 D4 群变换（90°旋转 × 翻转），像素精确对齐，不需要插值"""
    k = rng.integers(0, 4)
    img = np.rot90(img, k, axes=(0, 1)).copy()
    mask = np.rot90(mask, k, axes=(0, 1)).copy()
    if rng.random() < 0.5:
        img = np.flipud(img).copy()
        mask = np.flipud(mask).copy()
    if rng.random() < 0.5:
        img = np.fliplr(img).copy()
        mask = np.fliplr(mask).copy()
    return img, mask


def elastic_deform(img, mask, rng, alpha=40.0, sigma=6.0):
    """弹性形变：模拟样品轻微形变/扫描畸变。mask 用最近邻，避免产生灰阶"""
    h, w = img.shape[:2]
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    img_d = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)
    mask_d = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_REFLECT)
    return img_d, (mask_d > 0.5).astype(np.float32)


def random_scale(img, mask, rng, scale_range=(0.9, 1.1)):
    """轻微缩放（注意：缺陷本身面积很小，缩放幅度要小，别把缺陷缩没了）"""
    s = rng.uniform(*scale_range)
    h, w = img.shape[:2]
    nh, nw = max(8, int(round(h * s))), max(8, int(round(w * s)))
    img_s = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask_s = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    if (nh, nw) != (h, w):   # 居中裁剪/回补到原尺寸
        if s > 1:
            oy, ox = (nh - h) // 2, (nw - w) // 2
            img_s, mask_s = img_s[oy:oy + h, ox:ox + w], mask_s[oy:oy + h, ox:ox + w]
        else:
            canvas_i = np.zeros((h, w), img_s.dtype)
            canvas_m = np.zeros((h, w), mask_s.dtype)
            oy, ox = (h - nh) // 2, (w - nw) // 2
            canvas_i[oy:oy + nh, ox:ox + nw] = img_s
            canvas_m[oy:oy + nh, ox:ox + nw] = mask_s
            img_s, mask_s = canvas_i, canvas_m
    return img_s, (mask_s > 0.5).astype(np.float32)


# ---------------------------------------------------------------- 灰度 / 噪声
def add_speckle(img, rng, sigma=0.15):
    """乘性散斑噪声：I' = I * (1 + N(0, sigma))，符合超声成像物理"""
    n = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img * (1.0 + n), 0.0, 1.0).astype(np.float32)


def brightness_contrast(img, rng, b_range=0.12, c_range=0.15):
    b = rng.uniform(-b_range, b_range)
    c = rng.uniform(1.0 - c_range, 1.0 + c_range)
    return np.clip((img - 0.5) * c + 0.5 + b, 0.0, 1.0).astype(np.float32)


def gamma_jitter(img, rng, g_range=(0.85, 1.18)):
    g = rng.uniform(*g_range)
    return np.clip(img, 0, 1) ** g


# ---------------------------------------------------------------- 缺陷粘贴
def build_defect_bank(samples, min_area=9, max_side=512):
    """从全部缺陷图中抠出"缺陷小块"，构成本地缺陷库。

    Args:
        samples: [(img01(H,W), mask01(H,W)), ...]
    Returns:
        bank: [(patch_img(p,p), patch_mask(p,p)), ...]  patch 为缺陷外接正方框 + padding
    """
    bank = []
    for img, mask in samples:
        m = (mask > 0.5).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue
            pad = max(4, int(0.25 * max(w, h)))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(m.shape[1], x + w + pad), min(m.shape[0], y + h + pad)
            # 裁成正方形，便于后续随机旋转
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            side = min(max(x1 - x0, y1 - y0) + 2, max_side)
            sx0, sy0 = max(0, cx - side // 2), max(0, cy - side // 2)
            sx1, sy1 = min(m.shape[1], sx0 + side), min(m.shape[0], sy0 + side)
            pi = img[sy0:sy1, sx0:sx1].copy()
            pm = m[sy0:sy1, sx0:sx1].copy().astype(np.float32)
            if pm.sum() < min_area:
                continue
            bank.append((pi, pm))
    return bank


def _match_intensity(src, src_mask, dst_roi):
    """把粘贴块的背景亮度对齐到目标区域，避免"贴上去的方块"一眼假。

    只利用缺陷区域之外的像素估计亮度差，缺陷本身保持原样。
    """
    bg_src = src[src_mask < 0.5]
    bg_dst = dst_roi[src_mask < 0.5]
    if bg_src.size < 20 or bg_dst.size < 20:
        return src
    delta = float(bg_dst.mean() - bg_src.mean())
    return np.clip(src + delta, 0.0, 1.0).astype(np.float32)


def copy_paste(img, mask, bank, rng, max_num=3):
    """把缺陷库里的小块随机变换后贴到当前图上（允许贴到已有缺陷的图上）。"""
    if not bank:
        return img, mask
    img = img.copy()
    mask = mask.copy()
    H, W = img.shape[:2]
    for _ in range(rng.integers(1, max_num + 1)):
        pi, pm = bank[rng.integers(0, len(bank))]
        # 随机 D4 + 轻微缩放，增加多样性
        pi, pm = random_d4(pi, pm, rng)
        s = rng.uniform(0.8, 1.25)
        ph, pw = pi.shape[:2]
        nh, nw = max(4, int(ph * s)), max(4, int(pw * s))
        if nh >= H or nw >= W:
            continue
        pi = cv2.resize(pi, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pm = cv2.resize(pm, (nw, nh), interpolation=cv2.INTER_NEAREST)

        oy = rng.integers(0, H - nh + 1)
        ox = rng.integers(0, W - nw + 1)
        roi_i = img[oy:oy + nh, ox:ox + nw]
        roi_m = mask[oy:oy + nh, ox:ox + nw]

        # 只在"目标位置本来没有缺陷"的地方贴，避免标签冲突
        conflict = (roi_m > 0.5) & (pm > 0.5)
        pm_use = pm.copy()
        pm_use[conflict] = 0.0
        if pm_use.sum() < 5:
            continue

        pi = _match_intensity(pi, pm, roi_i)

        # 软边界融合：mask 边缘高斯羽化，消除硬拼缝
        soft = cv2.GaussianBlur(pm_use, (0, 0), 1.2)
        soft = np.clip(soft * 1.6, 0, 1)
        img[oy:oy + nh, ox:ox + nw] = roi_i * (1 - soft) + pi * soft
        mask[oy:oy + nh, ox:ox + nw] = np.clip(roi_m + pm_use, 0, 1)
    return img.astype(np.float32), (mask > 0.5).astype(np.float32)


# ---------------------------------------------------------------- 裁剪
def pad_to(img, mask, size):
    """小于 patch 尺寸的图补边（反射填充），保证能组成一个 batch"""
    h, w = img.shape[:2]
    ph = max(0, size - h)
    pw = max(0, size - w)
    if ph == 0 and pw == 0:
        return img, mask
    img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    return img, mask


def foreground_biased_crop(img, mask, size, rng, fg_prob=0.75):
    """前景偏置裁剪：以 fg_prob 的概率保证窗口内含有缺陷像素。

    这是让小目标训练收敛的关键。否则 200 张图里绝大多数 patch 是纯背景，
    模型会直接学成"全背景输出"（这就是极端不平衡下 BCE/Focal 也会失效的原因）。
    """
    H, W = img.shape[:2]
    ys, xs = np.where(mask > 0.5)
    use_fg = (len(ys) > 0) and (rng.random() < fg_prob)

    if H <= size and W <= size:
        img, mask = pad_to(img, mask, size)
        return img[:size, :size], mask[:size, :size]

    if use_fg:
        idx = rng.integers(0, len(ys))
        cy, cx = int(ys[idx]), int(xs[idx])
        # 让缺陷落在窗口内的随机位置，而不是总在中心（增强位置鲁棒性）
        top = cy - rng.integers(size // 4, size * 3 // 4 + 1)
        left = cx - rng.integers(size // 4, size * 3 // 4 + 1)
    else:
        top = rng.integers(0, H - size + 1)
        left = rng.integers(0, W - size + 1)

    top = int(np.clip(top, 0, H - size))
    left = int(np.clip(left, 0, W - size))
    return img[top:top + size, left:left + size], mask[top:top + size, left:left + size]


# ---------------------------------------------------------------- 组合
def train_augment(img, mask, rng, cfg, bank=None):
    """训练期增强总入口（输入均为 float32 [0,1] 的 H×W）"""
    img, mask = random_d4(img, mask, rng)

    if rng.random() < cfg.elastic_prob:
        img, mask = elastic_deform(img, mask, rng)
    if rng.random() < 0.3:
        img, mask = random_scale(img, mask, rng)
    if rng.random() < cfg.brightness_prob:
        img = brightness_contrast(img, rng)
    if rng.random() < 0.3:
        img = gamma_jitter(img, rng)
    if rng.random() < cfg.speckle_prob:
        img = add_speckle(img, rng, cfg.speckle_sigma)
    if bank and rng.random() < cfg.copy_paste_prob:
        img, mask = copy_paste(img, mask, bank, rng, cfg.copy_paste_max)

    img, mask = foreground_biased_crop(img, mask, cfg.patch_size, rng, cfg.fg_crop_prob)
    return img.astype(np.float32), (mask > 0.5).astype(np.float32)
