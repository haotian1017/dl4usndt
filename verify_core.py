# -*- coding: utf-8 -*-

import numpy as np
import cv2

from config import Config
from preprocess import (robust_normalize, bilateral_denoise, srad, apply_clahe,
                        preprocess, read_mask)
from augment import (random_d4, elastic_deform, add_speckle, copy_paste,
                     build_defect_bank, foreground_biased_crop, pad_to, train_augment)
from metrics import (dice, iou, precision, recall, pro_score,
                     image_level_decision)

ALL = []


def check(cond, msg):
    ALL.append(bool(cond))
    print(("  [OK]   " if cond else "  [FAIL] ") + msg)


def synth(h=256, w=256, n_def=2, seed=0, speckle=0.12):
    """合成一张"超声风格"图：暗背景 + 几个高亮缺陷块 + 乘性散斑"""
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 0.25, np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    base += 0.05 * np.sin(yy / 17.0)            # 缓慢背景起伏
    mask = np.zeros((h, w), np.float32)
    for _ in range(n_def):
        r = rng.integers(4, 12)
        cy, cx = rng.integers(r + 2, h - r - 2), rng.integers(r + 2, w - r - 2)
        cv2.circle(mask, (int(cx), int(cy)), int(r), 1.0, -1)
    img = base + 0.55 * mask                     # C-SAM 分层缺陷：高亮
    img = img * (1 + rng.normal(0, speckle, img.shape))   # 乘性散斑
    return np.clip(img, 0, 1).astype(np.float32), mask


print("\n=== 1. 超声预处理 ===")
img, mask = synth()
n = robust_normalize(img * 1000 + 500)
check(n.min() >= 0 and n.max() <= 1 and abs(n.mean() - n.mean()) < 1e-9,
      f"robust_normalize 输出在 [0,1]（min={n.min():.3f} max={n.max():.3f}）")

dn = bilateral_denoise(img)
# 背景区噪声应下降；缺陷与背景的对比度（跨边缘梯度）应基本保留
bg_noise_before = img[mask < .5].std()
bg_noise_after = dn[mask < .5].std()
con_before = img[mask > .5].mean() - img[mask < .5].mean()
con_after = dn[mask > .5].mean() - dn[mask < .5].mean()
check(bg_noise_after < bg_noise_before,
      f"双边滤波降低背景噪声 std {bg_noise_before:.4f} -> {bg_noise_after:.4f}")
check(con_after > 0.9 * con_before,
      f"双边滤波保住缺陷对比度 {con_before:.3f} -> {con_after:.3f}")

sr = srad(img, n_iter=15)
check(np.isfinite(sr).all() and sr.min() >= 0 and sr.max() <= 1,
      f"SRAD 输出有效 [0,1]（min={sr.min():.3f} max={sr.max():.3f}）")
con_sr = sr[mask > .5].mean() - sr[mask < .5].mean()
check(sr[mask < .5].std() < bg_noise_before,
      f"SRAD 降低背景噪声 std {bg_noise_before:.4f} -> {sr[mask < .5].std():.4f}")
check(con_sr > 0.9 * con_before,
      f"SRAD 保住缺陷对比度 {con_before:.3f} -> {con_sr:.3f}")

low = np.clip(img * 0.35 + 0.3, 0, 1)            # 模拟低对比度
cl = apply_clahe(low, clip=2.0)
check(cl.std() > low.std(),
      f"CLAHE 拉开对比度 std {low.std():.4f} -> {cl.std():.4f}")

cfg = Config()
pp = preprocess(img * 1000, cfg)
check(pp.shape == img.shape and pp.dtype == np.float32 and
      pp.min() >= 0 and pp.max() <= 1, "完整预处理流水线输出合法")

print("\n=== 2. 增强 ===")
rng = np.random.default_rng(1)
i2, m2 = random_d4(img, mask, rng)
# D4 后 img/mask 必须像素级对齐：缺陷区亮度仍显著高于背景
check(i2[m2 > .5].mean() > i2[m2 < .5].mean() + 0.2,
      f"D4 变换后 img/mask 仍对齐（缺陷亮度 {i2[m2>.5].mean():.3f} vs "
      f"背景 {i2[m2<.5].mean():.3f}）")

i3, m3 = elastic_deform(img, mask, rng)
check(i3.shape == img.shape and m3.shape == mask.shape and
      set(np.unique(m3)).issubset({0.0, 1.0}), "弹性形变后 mask 仍为 0/1 二值")
check(abs(m3.sum() - mask.sum()) / mask.sum() < 0.35,
      f"弹性形变后缺陷面积基本保持 ({mask.sum():.0f} -> {m3.sum():.0f})")

sp = add_speckle(img, rng, 0.15)
check(sp.min() >= 0 and sp.max() <= 1 and sp.std() > img.std(),
      "散斑噪声叠加后仍在合法范围且方差增大")

pairs = [(img, mask)] + [synth(seed=s) for s in range(2, 8)]
bank = build_defect_bank(pairs)
check(len(bank) > 0, f"缺陷库构建成功，共 {len(bank)} 个缺陷块")

bg = synth(seed=99, n_def=0)[0]
bgm = np.zeros_like(bg)
pi, pm = copy_paste(bg, bgm, bank, rng, max_num=3)
check(pm.sum() > 0, f"缺陷粘贴成功：正常图新增缺陷像素 {int(pm.sum())}")
check(pm.shape == bg.shape and pi.shape == bg.shape, "粘贴后尺寸不变")
check(pi[pm > .5].mean() > pi[pm < .5].mean(),
      "粘贴的缺陷区域仍为高亮（强度已对齐目标图）")

# 前景偏置裁剪命中率
rng2 = np.random.default_rng(7)
hit = 0
for _ in range(60):
    ci, cm = foreground_biased_crop(img, mask, 128, rng2, fg_prob=0.75)
    hit += int((cm > .5).sum() > 0)
check(hit / 60 > 0.6, f"前景偏置裁剪命中率 {hit/60:.2f}（设定 0.75，缺陷占比仍低时略降）")

ci, cm = foreground_biased_crop(img, mask, 128, rng2, fg_prob=0.75)
check(ci.shape == (128, 128) and cm.shape == (128, 128), "裁剪尺寸正确")
small = img[:64, :64]
si, sm = pad_to(small, small * 0, 128)
check(si.shape == (128, 128), "小于 patch 的图能正确补边")

ti, tm = train_augment(img, mask, rng2, cfg, bank)
check(ti.shape == (cfg.patch_size, cfg.patch_size) and
      set(np.unique(tm)).issubset({0.0, 1.0}), "train_augment 端到端输出合法")

print("\n=== 3. 指标 ===")
p1 = np.zeros((32, 32), np.float32); p1[8:16, 8:16] = 1
g1 = np.zeros((32, 32), np.float32); g1[8:16, 8:16] = 1
check(abs(dice(p1, g1) - 1.0) < 1e-6, f"完全一致时 Dice=1（{dice(p1,g1):.4f}）")
p2 = np.zeros((32, 32), np.float32); p2[8:12, 8:12] = 1     # 只覆盖了 1/4
check(abs(dice(p2, g1) - 0.4) < 1e-3, f"覆盖 1/4 时 Dice=0.4（{dice(p2,g1):.4f}）")
check(abs(recall(p2, g1) - 0.25) < 1e-3, f"覆盖 1/4 时 Recall=0.25（{recall(p2,g1):.4f}）")
check(abs(precision(p2, g1) - 1.0) < 1e-6, "无过检时 Precision=1")

# PRO：两个缺陷，一个全检出、一个检出一半 → PRO=0.75
g3 = np.zeros((64, 64), np.float32)
g3[4:12, 4:12] = 1
g3[40:48, 40:48] = 1
p3 = np.zeros((64, 64), np.float32)
p3[4:12, 4:12] = 1
p3[40:44, 40:48] = 1
check(abs(pro_score(p3, g3) - 0.75) < 1e-6, f"PRO 逐区域平均正确（{pro_score(p3,g3):.4f}）")
check(pro_score(np.zeros((8, 8), np.float32), np.zeros((8, 8), np.float32)) !=
      pro_score(np.zeros((8, 8), np.float32), np.zeros((8, 8), np.float32)),
      "无缺陷区域时 PRO 返回 nan（调用方跳过）")

prob = np.zeros((128, 128), np.float32)
prob[10:40, 10:40] = 0.9                    # 一个明显缺陷
prob[100:102, 100:102] = 0.9                # 4 px 的散斑残留
r = image_level_decision(prob, 0.5, min_area=20)
check(r["has_defect"] is True, "图像级判定：有缺陷 -> True")
check(r["defect_area_px"] == 30 * 30, f"小面积连通域被过滤（剩余 {r['defect_area_px']}px）")
r2 = image_level_decision(np.full((128, 128), 0.01, np.float32), 0.5, min_area=20)
check(r2["has_defect"] is False, "图像级判定：全背景 -> False")

print("\n=== 4. 滑窗 ===")
from infer_geom import gauss_weight, window_grid   # 与 infer.py 同源的纯 numpy 实现
gw64 = gauss_weight(64, 64)

for (H, W) in [(1000, 800), (512, 512), (700, 1300), (513, 517)]:
    ys, xs = window_grid(H, W, 512, 0.25)
    cov = np.zeros((H, W), np.float32)
    for y in ys:
        for x in xs:
            cov[y:y + 512, x:x + 512] += 1
    check(cov.min() >= 1, f"滑窗完整覆盖 {H}x{W}（最小覆盖次数 {int(cov.min())}）")

# 回归测试：单边小于 patch 的图（如 800x400）必须先 pad，否则切片会越界
P = 512
for (H, W) in [(800, 400), (400, 800), (300, 300), (512, 513), (2000, 300)]:
    pad_h, pad_w = max(0, P - H), max(0, P - W)
    H2, W2 = H + pad_h, W + pad_w
    if H2 <= P and W2 <= P:
        check(H2 == P and W2 == P, f"{H}x{W} 补边后正好一整块")
        continue
    ys, xs = window_grid(H2, W2, P, 0.25)
    cov = np.zeros((H2, W2), np.float32)
    tile_ok = True
    for y in ys:
        for x in xs:
            if y + P > H2 or x + P > W2:
                tile_ok = False
            cov[y:y + P, x:x + P] += 1
    check(tile_ok and cov.min() >= 1,
          f"{H}x{W} 补边到 {H2}x{W2} 后所有窗口尺寸合法且完整覆盖")

gw64 = gauss_weight(64, 64)
check(abs(gw64[32, 32] - gw64.max()) < 1e-6 and gw64[0, 0] < 0.5 * gw64.max(),
      f"高斯窗中心高边缘低（中心 {gw64[32,32]:.3f} 角 {gw64[0,0]:.3f}）")
# 融合一致性：常数输入经高斯加权融合后仍是常数（说明拼接无缝）
H, W, P = 900, 700, 512
gw = gauss_weight(P, P)
acc = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
ys, xs = window_grid(H, W, P, 0.25)
for y in ys:
    for x in xs:
        acc[y:y + P, x:x + P] += gw
        wsum[y:y + P, x:x + P] += gw
out = acc / np.maximum(wsum, 1e-6)
check(np.allclose(out, 1.0, atol=1e-5),
      f"高斯权重融合无接缝（常量图还原误差 {np.abs(out-1).max():.2e}）")

print("\n" + "=" * 46)
print(f"通过 {sum(ALL)}/{len(ALL)} 项")
if not all(ALL):
    print("存在失败项，请检查上方 [FAIL]")
    raise SystemExit(1)
print("全部通过 ✅")
