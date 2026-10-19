# -*- coding: utf-8 -*-

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from preprocess import read_gray, read_mask, preprocess, normalize_for_model, IMG_EXTS
from augment import train_augment, build_defect_bank, pad_to


def scan_dataset(root, img_dir="images", mask_dir="masks"):
    """扫描数据目录，返回样本列表"""
    idir, mdir = Path(root) / img_dir, Path(root) / mask_dir
    if not idir.is_dir():
        raise FileNotFoundError(f"找不到图像目录: {idir}")
    files = sorted([p for p in idir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not files:
        raise FileNotFoundError(f"{idir} 下没有图像")

    samples = []
    for p in files:
        mp = mdir / p.name
        if not mp.exists():
            # 也允许 mask 用其它后缀
            cands = [mdir / (p.stem + e) for e in IMG_EXTS]
            mp = next((c for c in cands if c.exists()), None)
        if mp is None:
            raise FileNotFoundError(f"缺少标注: {mdir / p.name}")
        samples.append({"img": str(p), "mask": str(mp), "name": p.stem})
    return samples


def kfold_split(n, folds=5, seed=42):
    """自实现 KFold，避免多装一个 sklearn 依赖"""
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    folds_list = [[] for _ in range(folds)]
    for i, v in enumerate(idx):
        folds_list[i % folds].append(v)
    return folds_list


def train_val_split(n, val_ratio=0.2, seed=42):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    nv = max(1, int(round(n * val_ratio)))
    return idx[nv:], idx[:nv]


class CSAMDataset(Dataset):
    """C-SAM 分割数据集。

    train 模式：读图 → 超声预处理 → Copy-Paste/几何/散斑增强 → 前景偏置裁剪 → patch
    val  模式：读图 → 超声预处理 → 不增强，返回整图（验证时用滑窗推理评估）
    """

    def __init__(self, samples, cfg, mode="train", defect_bank=None):
        self.samples = samples
        self.cfg = cfg
        self.mode = mode
        self.bank = defect_bank
        self._cache = {}

    def __len__(self):
        return len(self.samples)

    def _load(self, i):
        """带缓存的读取 + 预处理"""
        if i in self._cache:
            return self._cache[i]
        s = self.samples[i]
        img = preprocess(read_gray(s["img"]), self.cfg, is_mask=False)
        mask = read_mask(s["mask"])
        if self.cfg.max_cache and len(self._cache) < self.cfg.max_cache:
            self._cache[i] = (img, mask)
        return img, mask

    def __getitem__(self, i):
        img, mask = self._load(i)
        rng = np.random.default_rng(self.cfg.seed + i * 7919
                                    + (0 if self.mode == "train" else 12345))

        if self.mode == "train":
            img, mask = train_augment(img, mask, rng, self.cfg, self.bank)
        else:
            # 验证集不做随机变换，但补到 patch 的整数倍便于滑窗
            img, mask = pad_to(img, mask, self.cfg.patch_size)

        img = normalize_for_model(img)
        x = torch.from_numpy(img[None, ...].astype(np.float32))    # 1×H×W
        y = torch.from_numpy(mask[None, ...].astype(np.float32))
        return x, y


def build_bank(samples, cfg):
    """构建缺陷库（用于 Copy-Paste）。只在训练集内部的样本上构建，避免信息泄漏"""
    pairs = []
    for s in samples:
        try:
            img = preprocess(read_gray(s["img"]), cfg)
            mask = read_mask(s["mask"])
            if mask.sum() > 0:
                pairs.append((img, mask))
        except Exception as e:
            print(f"[warn] 跳过 {s['img']}: {e}")
    return build_defect_bank(pairs)


def make_loader(samples, cfg, mode="train", bank=None, batch_size=None, shuffle=None):
    ds = CSAMDataset(samples, cfg, mode=mode, defect_bank=bank)
    bs = batch_size or cfg.batch_size
    sh = (mode == "train") if shuffle is None else shuffle
    return DataLoader(
        ds, batch_size=bs, shuffle=sh,
        num_workers=cfg.num_workers, pin_memory=True,
        drop_last=(mode == "train"),
        persistent_workers=(cfg.num_workers > 0),
    )
