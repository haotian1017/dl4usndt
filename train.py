# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import Config
from dataset import scan_dataset, kfold_split, train_val_split, build_bank, make_loader
from losses import build_loss, warmup_loss
from models import build_model, param_groups, count_params
from metrics import dice, iou, precision, recall, pro_score
from infer import sliding_window_predict
from preprocess import read_gray, read_mask, preprocess


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model, samples, cfg, device, prob_thresh=0.5):
    """滑窗推理评估整图指标（不裁剪，避免高估）"""
    model.eval()
    agg = {"dice": [], "iou": [], "precision": [], "recall": [], "pro": []}
    for s in samples:
        img01 = preprocess(read_gray(s["img"]), cfg)
        gt = read_mask(s["mask"])
        prob = sliding_window_predict(model, img01, cfg.infer_patch, cfg.infer_overlap,
                                      device, tta=False, amp=False)
        pred = (prob > prob_thresh).astype(np.float32)
        if gt.sum() == 0 and pred.sum() == 0:
            # 真阴性：全部记满分会虚高，这里记 dice=1 但 PRO 跳过
            agg["dice"].append(1.0); agg["iou"].append(1.0)
            agg["precision"].append(1.0); agg["recall"].append(1.0)
            continue
        agg["dice"].append(dice(pred, gt))
        agg["iou"].append(iou(pred, gt))
        agg["precision"].append(precision(pred, gt))
        agg["recall"].append(recall(pred, gt))
        pr = pro_score(pred, gt)
        if pr == pr:
            agg["pro"].append(pr)
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in agg.items()}


def search_threshold(model, samples, cfg, device):
    """在验证集上搜索最优概率阈值（以 Dice 为目标）"""
    model.eval()
    best_t, best_s = 0.5, -1.0
    cache = []
    for s in samples:
        img01 = preprocess(read_gray(s["img"]), cfg)
        gt = read_mask(s["mask"])
        prob = sliding_window_predict(model, img01, cfg.infer_patch, cfg.infer_overlap,
                                      device, tta=False, amp=False)
        cache.append((prob, gt))
    for t in np.arange(0.25, 0.76, 0.05):
        scores = [dice((p > t).astype(np.float32), g) for p, g in cache]
        m = float(np.mean(scores))
        if m > best_s:
            best_s, best_t = m, float(t)
    return best_t, best_s


def train_one_fold(cfg, all_samples, train_idx, val_idx, fold, args):
    set_seed(cfg.seed + fold)
    device = args.device
    out_dir = Path(cfg.out_dir) / (f"fold{fold}" if cfg.folds > 1 else "run")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]

    # 缺陷库只在训练集内构建，防止信息泄漏
    bank = build_bank(train_samples, cfg) if cfg.copy_paste_prob > 0 else None
    print(f"[fold {fold}] 训练 {len(train_samples)} / 验证 {len(val_samples)}，"
          f"缺陷库 {len(bank) if bank else 0} 个")

    train_loader = make_loader(train_samples, cfg, "train", bank)
    # 验证走整图滑窗（见 validate），不需要 DataLoader

    model, ext_enc = build_model(cfg)
    model.to(device)
    print(f"模型参数量: {count_params(model)/1e6:.3f} M")

    criterion_main = build_loss(cfg.loss, cfg)
    criterion_warm = warmup_loss(cfg)
    optimizer = AdamW(param_groups(model, cfg, ext_enc),
                      weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.startswith("cuda")))

    writer = None
    if args.tb:
        from torch.utils.tensorboard import SummaryWriter   # 延迟导入，没装也能跑
        writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    log_path = out_dir / "log.csv"
    f_log = open(log_path, "w", newline="", encoding="utf-8")
    writer_csv = csv.writer(f_log)
    writer_csv.writerow(["epoch", "lr", "train_loss", "val_dice", "val_iou",
                         "val_precision", "val_recall", "val_pro"])

    best_score, best_epoch, no_improve = -1.0, -1, 0
    for epoch in range(cfg.epochs):
        model.train()
        crit = criterion_warm if epoch < cfg.loss_warmup_epochs else criterion_main
        running, nb = 0.0, 0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(cfg.amp and device.startswith("cuda"))):
                logits = model(x)
                if logits.shape[-2:] != y.shape[-2:]:
                    logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear",
                                            align_corners=False)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()); nb += 1

        scheduler.step()
        train_loss = running / max(nb, 1)
        lr_now = optimizer.param_groups[0]["lr"]

        do_val = (epoch % args.val_every == 0) or (epoch == cfg.epochs - 1)
        if do_val:
            m = validate(model, val_samples, cfg, device)
            # 主指标：Dice 与 PRO 的平均。PRO 更能反映"每个缺陷检出了多少"
            score = np.nanmean([m["dice"], m.get("pro", np.nan)])
            writer_csv.writerow([epoch, f"{lr_now:.2e}", f"{train_loss:.5f}",
                                 f"{m['dice']:.4f}", f"{m['iou']:.4f}",
                                 f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                                 f"{m.get('pro', float('nan')):.4f}"])
            f_log.flush()
            print(f"  ep{epoch:03d} loss={train_loss:.4f} "
                  f"dice={m['dice']:.4f} iou={m['iou']:.4f} "
                  f"P={m['precision']:.4f} R={m['recall']:.4f} "
                  f"PRO={m.get('pro', float('nan')):.4f}  ({time.time()-t0:.0f}s)")

            if writer:
                writer.add_scalar("train/loss", train_loss, epoch)
                for k, v in m.items():
                    writer.add_scalar(f"val/{k}", v, epoch)

            if score > best_score:
                best_score, best_epoch, no_improve = score, epoch, 0
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                            "epoch": epoch, "score": score},
                           out_dir / "best.pth")
                cfg.save(out_dir / "config.json")
            else:
                no_improve += 1
                if no_improve >= cfg.early_stop_patience:
                    print(f"  早停于 epoch {epoch}（最佳 epoch {best_epoch}）")
                    break
        else:
            print(f"  ep{epoch:03d} loss={train_loss:.4f}  ({time.time()-t0:.0f}s)")

    f_log.close()
    if writer:
        writer.close()

    # 用最佳权重复评 + 搜索阈值
    ckpt = out_dir / "best.pth"
    if ckpt.exists():
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd["model"])
    model.to(device)
    best_t, _ = search_threshold(model, val_samples, cfg, device)
    final = validate(model, val_samples, cfg, device, prob_thresh=best_t)
    cfg.prob_thresh = best_t
    cfg.save(out_dir / "config.json")
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "prob_thresh": best_t},
               ckpt)

    print(f"[fold {fold}] 最佳 epoch={best_epoch}  最优阈值={best_t:.2f}")
    print(f"[fold {fold}] 最终指标: " +
          "  ".join(f"{k}={v:.4f}" for k, v in final.items()))
    return {"fold": fold, "best_epoch": best_epoch, "prob_thresh": best_t,
            "best_score": float(best_score), **{f"final_{k}": v for k, v in final.items()}}


def main():
    ap = argparse.ArgumentParser("C-SAM 芯片超声缺陷分割训练")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--patch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--loss", default=None)
    ap.add_argument("--pos_weight", type=float, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument("--tb", action="store_true", help="写 tensorboard")
    ap.add_argument("--final", action="store_true",
                    help="用全量数据训最终模型（不做验证划分）")
    ap.add_argument("--config", default=None, help="从 config.json 载入超参")
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    for k, v in vars(args).items():
        if v is not None and hasattr(cfg, k) and k not in ("device", "val_every", "tb"):
            setattr(cfg, k, v)

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("[warn] 未检测到 CUDA，改用 CPU"); args.device = "cpu"

    all_samples = scan_dataset(cfg.data_root, cfg.img_dir, cfg.mask_dir)
    print(f"共扫描到 {len(all_samples)} 个样本")

    if args.final:
        cfg.folds = 1
        idx = list(range(len(all_samples)))
        res = train_one_fold(cfg, all_samples, idx, idx, 0, args)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    folds = kfold_split(len(all_samples), cfg.folds, cfg.seed)
    results = []
    for f in range(cfg.folds):
        val_idx = folds[f]
        train_idx = [i for i in range(len(all_samples)) if i not in val_idx]
        print(f"\n===== Fold {f}/{cfg.folds-1} =====")
        results.append(train_one_fold(cfg, all_samples, train_idx, val_idx, f, args))

    out = Path(cfg.out_dir)
    with open(out / "cv_results.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)

    keys = [k for k in results[0] if k.startswith("final_")]
    print("\n========== 交叉验证汇总 ==========")
    for k in keys:
        vals = [r[k] for r in results if r[k] == r[k]]
        if vals:
            print(f"{k[6:]:10s}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
                  f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")
    print(f"\n结果已保存: {out / 'cv_results.json'}")


if __name__ == "__main__":
    main()
