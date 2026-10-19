# -*- coding: utf-8 -*-

import numpy as np


def _bin(x, thr=0.5):
    return (np.asarray(x) > thr).astype(np.float32)


def confusion(pred, gt, thr=0.5):
    p, g = _bin(pred, thr), _bin(gt, thr)
    tp = float((p * g).sum())
    fp = float((p * (1 - g)).sum())
    fn = float(((1 - p) * g).sum())
    tn = float(((1 - p) * (1 - g)).sum())
    return tp, fp, fn, tn


def dice(pred, gt, thr=0.5, smooth=1e-6):
    tp, fp, fn, _ = confusion(pred, gt, thr)
    return (2 * tp + smooth) / (2 * tp + fp + fn + smooth)


def iou(pred, gt, thr=0.5, smooth=1e-6):
    tp, fp, fn, _ = confusion(pred, gt, thr)
    return (tp + smooth) / (tp + fp + fn + smooth)


def precision(pred, gt, thr=0.5, smooth=1e-6):
    tp, fp, _, _ = confusion(pred, gt, thr)
    return (tp + smooth) / (tp + fp + smooth)


def recall(pred, gt, thr=0.5, smooth=1e-6):
    tp, _, fn, _ = confusion(pred, gt, thr)
    return (tp + smooth) / (tp + fn + smooth)


def _connected_components(mask):
    """不依赖 scipy 的连通域标记（8 邻域 BFS）。mask: 0/1 uint8"""
    from collections import deque
    h, w = mask.shape
    lab = np.zeros((h, w), dtype=np.int32)
    cur = 0
    for i in range(h):
        for j in range(w):
            if mask[i, j] and lab[i, j] == 0:
                cur += 1
                q = deque([(i, j)])
                lab[i, j] = cur
                while q:
                    y, x = q.popleft()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and lab[ny, nx] == 0:
                                lab[ny, nx] = cur
                                q.append((ny, nx))
    return lab, cur


def pro_score(pred, gt, thr=0.5, max_regions=None):
    """PRO：每个真实缺陷区域的平均覆盖率。全图无缺陷时返回 nan（由调用方跳过）"""
    p, g = _bin(pred, thr), _bin(gt, thr)
    if g.sum() == 0:
        return float("nan")
    lab, n = _connected_components(g.astype(np.uint8))
    if n == 0:
        return float("nan")
    areas = np.array([(lab == i).sum() for i in range(1, n + 1)], dtype=np.float64)
    # 忽略碎片噪声区域（<4 px），避免个别像素级标注把 PRO 拉飞
    keep = [i + 1 for i, a in enumerate(areas) if a >= 4]
    if not keep:
        keep = list(range(1, n + 1))
    if max_regions is not None and len(keep) > max_regions:
        keep = sorted(keep, key=lambda i: -areas[i - 1])[:max_regions]
    vals = [float(p[lab == i].sum()) / float(areas[i - 1]) for i in keep]
    return float(np.mean(vals))


def all_metrics(pred, gt, thr=0.5):
    d = {
        "dice": dice(pred, gt, thr),
        "iou": iou(pred, gt, thr),
        "precision": precision(pred, gt, thr),
        "recall": recall(pred, gt, thr),
    }
    pr = pro_score(pred, gt, thr)
    if pr == pr:   # 非 nan
        d["pro"] = pr
    return d


def image_level_decision(prob, thr=0.5, min_area=20, min_ratio=0.0):
    """由概率图给出"有/无缺陷"的图像级判定 + 缺陷面积统计。

    工业现场真正需要的往往是这个二值结论。判定逻辑：
    1) 阈值化 → 2) 去掉面积过小的连通域（散斑残留）→ 3) 仍有残留则判为"有缺陷"
    """
    from collections import deque
    m = (prob > thr).astype(np.uint8)
    h, w = m.shape
    visited = np.zeros_like(m)
    largest = 0
    kept = np.zeros_like(m)
    for i in range(h):
        for j in range(w):
            if m[i, j] and not visited[i, j]:
                q = deque([(i, j)])
                visited[i, j] = 1
                comp = []
                while q:
                    y, x = q.popleft()
                    comp.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = 1
                                q.append((ny, nx))
                if len(comp) >= max(min_area, int(min_ratio * h * w)):
                    largest = max(largest, len(comp))
                    for y, x in comp:
                        kept[y, x] = 1
    return {
        "has_defect": bool(largest > 0),
        "defect_area_px": int(kept.sum()),
        "defect_ratio": float(kept.sum()) / float(h * w),
        "largest_component_px": int(largest),
        "mask": kept,
    }
