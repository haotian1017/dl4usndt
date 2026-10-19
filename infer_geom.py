# -*- coding: utf-8 -*-



def gauss_weight(h, w, sigma_scale=0.125, floor=0.05):
    """中心高、边缘低的高斯权重窗，用于融合重叠区域、消除拼接缝。

    floor: 权重下界（相对峰值）。必须 > 0，否则只被窗口边缘覆盖一次的像素会出现
           wsum=0 → 除零 → 图像边界出现黑边/概率塌陷。
    """
    import numpy as np
    yy = np.linspace(-1, 1, h, dtype=np.float32)
    xx = np.linspace(-1, 1, w, dtype=np.float32)
    gy = np.exp(-(yy ** 2) / (2 * sigma_scale ** 2))
    gx = np.exp(-(xx ** 2) / (2 * sigma_scale ** 2))
    g = (gy[:, None] * gx[None, :]).astype(np.float32)
    g = np.maximum(g, floor)
    return (g / g.max()).astype(np.float32)


def window_grid(H, W, patch, overlap=0.25):
    """生成滑窗左上角坐标。保证最后一格贴边，从而完整覆盖整图。

    Returns:
        ys, xs: 行/列起始坐标列表
    """
    stride = max(32, int(patch * (1 - overlap)))
    if H <= patch and W <= patch:
        return [0], [0]
    ys = list(range(0, max(1, H - patch + 1), stride))
    xs = list(range(0, max(1, W - patch + 1), stride))
    if ys[-1] != H - patch:
        ys.append(max(0, H - patch))
    if xs[-1] != W - patch:
        xs.append(max(0, W - patch))
    return ys, xs
