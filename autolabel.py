# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path

import numpy as np
import cv2

from preprocess import read_gray, robust_normalize, IMG_EXTS


def candidate_mask(img01, mode="bright", k=2.5, min_area=12, use_otsu=False):
    """阈值 + 形态学出候选缺陷区域"""
    x = img01
    if use_otsu:
        u8 = (np.clip(x, 0, 1) * 255).astype(np.uint8)
        thr, m = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = (m > 0).astype(np.uint8)
        if mode == "dark":
            m = 1 - m
    else:
        mu, sd = float(x.mean()), float(x.std())
        t = mu + k * sd if mode == "bright" else mu - k * sd
        m = ((x > t) if mode == "bright" else (x < t)).astype(np.uint8)

    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 1
    return out


def boxes_from_mask(m, pad=3):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        boxes.append([max(0, int(x - pad)), max(0, int(y - pad)),
                      int(w + 2 * pad), int(h + 2 * pad)])
    return boxes


def sam_refine(img01, boxes, ckpt, model_type="vit_b", device="cuda"):
    """可选：用框提示 SAM 精化 mask（需 pip install segment-anything）"""
    from segment_anything import sam_model_registry, SamPredictor
    import torch
    sam = sam_model_registry[model_type](checkpoint=ckpt)
    sam.to(device).eval()
    pred = SamPredictor(sam)
    u8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    rgb = cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB)
    pred.set_image(rgb)
    out = np.zeros(img01.shape[:2], np.uint8)
    for b in boxes:
        x, y, w, h = b
        box = np.array([x, y, x + w, y + h])
        masks, scores, _ = pred.predict(box=box, multimask_output=False)
        if len(masks):
            out[masks[0]] = 1
    return out


def to_labelme(name, img_path, h, w, boxes, masks=None):
    """导出 labelme json，便于人工微调"""
    shapes = []
    for i, (x, y, bw, bh) in enumerate(boxes):
        shapes.append({
            "label": "defect", "shape_type": "rectangle",
            "points": [[x, y], [x + bw, y + bh]], "flags": {},
        })
    return {"version": "5.0.1", "flags": {},
            "shapes": shapes, "imagePath": os.path.basename(img_path),
            "imageData": None, "imageHeight": h, "imageWidth": w}


def main():
    ap = argparse.ArgumentParser("C-SAM 半自动标注")
    ap.add_argument("--input", required=True, help="原始图目录")
    ap.add_argument("--out", required=True, help="输出 data 根目录")
    ap.add_argument("--mode", default="bright", choices=["bright", "dark"])
    ap.add_argument("--k", type=float, default=2.5, help="阈值 = mean ± k*std")
    ap.add_argument("--min_area", type=int, default=12)
    ap.add_argument("--otsu", action="store_true")
    ap.add_argument("--use_sam", action="store_true")
    ap.add_argument("--sam_ckpt", default=None)
    ap.add_argument("--sam_type", default="vit_b")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    src = Path(args.input)
    out_root = Path(args.out)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "masks").mkdir(parents=True, exist_ok=True)
    (out_root / "_labelme").mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS])
    print(f"待标注 {len(files)} 张")

    n_pos = 0
    for p in files:
        raw = read_gray(p)
        img01 = robust_normalize(raw)
        m = candidate_mask(img01, args.mode, args.k, args.min_area, args.otsu)
        if args.use_sam and args.sam_ckpt and m.sum() > 0:
            boxes = boxes_from_mask(m)
            m = sam_refine(img01, boxes, args.sam_ckpt, args.sam_type, args.device)
        cv2.imwrite(str(out_root / "images" / p.name), (np.clip(img01, 0, 1) * 255).astype(np.uint8))
        cv2.imwrite(str(out_root / "masks" / p.name), (m * 255).astype(np.uint8))
        boxes = boxes_from_mask(m)
        if boxes:
            n_pos += 1
        js = to_labelme(p.stem, p, *img01.shape[:2], boxes)
        with open(out_root / "_labelme" / f"{p.stem}.json", "w", encoding="utf-8") as f:
            json.dump(js, f, ensure_ascii=False, indent=1)
    print(f"完成：{len(files)} 张，其中 {n_pos} 张自动生成了缺陷候选。")
    print(f"输出：{out_root/'images'}  {out_root/'masks'}  {out_root/'_labelme'}")
    print("请用 Labelme / Label Studio 打开 _labelme 复核后再训练。")


if __name__ == "__main__":
    main()
