# -*- coding: utf-8 -*-

import argparse
import os
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from config import Config, cfg_from_args
from preprocess import read_gray, preprocess, normalize_for_model, IMG_EXTS
from models import build_model
from infer_geom import gauss_weight, window_grid


# ---------------------------------------------------------------- 滑窗
def sliding_window_predict(model, img01, patch=512, overlap=0.25, device="cuda",
                           tta=False, amp=False):
    """对整图做滑窗推理，返回与输入同尺寸的概率图 [0,1]。

    img01: 已预处理、未标准化的 float32 [0,1] 图
    """
    model.eval()
    H, W = img01.shape[:2]

    # 先补到不小于 patch（否则 H>patch 但 W<patch 时切片会越界），最后裁回原尺寸
    pad_h, pad_w = max(0, patch - H), max(0, patch - W)
    if pad_h or pad_w:
        img01 = cv2.copyMakeBorder(img01, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    H2, W2 = img01.shape[:2]

    if H2 <= patch and W2 <= patch:
        p = _predict_one(model, img01[:patch, :patch], device, tta, amp)
        return p[:H, :W]

    prob = np.zeros((H2, W2), dtype=np.float32)
    wsum = np.zeros((H2, W2), dtype=np.float32)
    gw = gauss_weight(patch, patch)
    ys, xs = window_grid(H2, W2, patch, overlap)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                tile = img01[y:y + patch, x:x + patch]
                p = _predict_one(model, tile, device, tta, amp)
                prob[y:y + patch, x:x + patch] += p * gw
                wsum[y:y + patch, x:x + patch] += gw

    wsum[wsum < 1e-6] = 1e-6
    return (prob / wsum).astype(np.float32)[:H, :W]


def _predict_one(model, tile, device, tta=False, amp=False):
    """单块推理；tta=True 时做 D4 平均"""
    def _fwd(a):
        t = torch.from_numpy(normalize_for_model(a)[None, None, ...]).to(device)
        if amp and device.startswith("cuda"):
            with torch.autocast("cuda"):
                out = model(t)
        else:
            out = model(t)
        out = F.interpolate(out, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(out)[0, 0].float().cpu().numpy()

    if not tta:
        return _fwd(tile)

    ps = [_fwd(tile)]
    for k in (1, 2, 3):
        r = np.rot90(tile, k).copy()
        pr = _fwd(r)
        ps.append(np.rot90(pr, -k).copy())
    f = np.flipud(tile).copy()
    ps.append(np.flipud(_fwd(f)).copy())
    return np.mean(ps, axis=0)


# ---------------------------------------------------------------- 加载
def load_model(ckpt, cfg, device="cuda"):
    model, _ = build_model(cfg)
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("model", sd)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------- 主流程
def predict_file(model, path, cfg, device="cuda", save_mask_to=None):
    raw = read_gray(path)
    img01 = preprocess(raw, cfg)
    prob = sliding_window_predict(model, img01, cfg.infer_patch, cfg.infer_overlap,
                                  device, cfg.tta, amp=False)
    from metrics import image_level_decision
    res = image_level_decision(prob, cfg.prob_thresh, cfg.min_area_px, cfg.min_area_ratio)
    res["prob_map"] = prob
    if save_mask_to:
        cv2.imwrite(str(save_mask_to), (res["mask"] * 255).astype(np.uint8))
    return res


def export_onnx(cfg, ckpt, out_path, img_size=512, opset=12):
    """导出 ONNX（动态 batch/尺寸），供 TensorRT / 边缘板卡部署"""
    model, _ = build_model(cfg)
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd.get("model", sd))
    model.eval()
    dummy = torch.randn(1, cfg.in_channels, img_size, img_size)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch", 2: "h", 3: "w"},
                      "logits": {0: "batch", 2: "h", 3: "w"}},
        opset_version=opset, do_constant_folding=True,
    )
    print(f"ONNX 已导出: {out_path}")


def main():
    ap = argparse.ArgumentParser("C-SAM 缺陷分割推理")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="图像文件或目录")
    ap.add_argument("--config", default=None, help="训练时保存的 config.json")
    ap.add_argument("--out_dir", default="./outputs/pred")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prob_thresh", type=float, default=None)
    ap.add_argument("--min_area_px", type=int, default=None)
    ap.add_argument("--tta", type=int, default=None)
    ap.add_argument("--export_onnx", default=None, help="可选：导出 ONNX 到该路径")
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    for k in ["prob_thresh", "min_area_px", "tta"]:
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, bool(v) if k == "tta" else v)

    if args.export_onnx:
        export_onnx(cfg, args.ckpt, args.export_onnx)
        if not os.path.isdir(args.input):
            return

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model = load_model(args.ckpt, cfg, device)

    inp = Path(args.input)
    files = sorted([p for p in inp.iterdir() if p.suffix.lower() in IMG_EXTS]) \
        if inp.is_dir() else [inp]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in files:
        t0 = time.time()
        r = predict_file(model, f, cfg, device, save_mask_to=out_dir / f"{f.stem}_mask.png")
        dt = (time.time() - t0) * 1000
        rows.append((f.name, "有缺陷" if r["has_defect"] else "无缺陷",
                     r["defect_area_px"], r["defect_ratio"] * 100, dt))
        print(f"{f.name:30s} {rows[-1][1]:6s} 面积={r['defect_area_px']:7d}px "
              f"({r['defect_ratio']*100:.3f}%)  {dt:.0f}ms")

    n_def = sum(1 for r in rows if r[1] == "有缺陷")
    print(f"\n合计 {len(rows)} 张，判为有缺陷 {n_def} 张。mask 已保存至 {out_dir}")


if __name__ == "__main__":
    main()
