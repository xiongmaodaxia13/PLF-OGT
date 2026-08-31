#!/usr/bin/env python
"""no_residual 3-seed 消融的 6h AUROC/AUPRC + cluster bootstrap CI —— 第三档 (#10).

陆老师 #10: 残差分支补 3 次独立训练消融, 使结论与主结果同口径.
本脚本: no_residual 3-seed (42/52/62) ensemble, 在 test 集 (67,665) 上算
6h TCR AUROC/AUPRC + 患者层聚类 bootstrap CI.

口径与 ablations_test_6h.json (seed42) 一致, 只是 3-seed + CI.
输出: results/v4/no_residual_3seed_ci.json

用法: python scripts/eval_v4_no_residual_3seed_ci.py
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # 1=4090 (与训练同卡, 推理时4090空闲则可用)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from v6.config import DEVICE, RESULTS_DIR, configure_cuda
from v6.data.dataset import collate
from v6.data.v4_dataset import PLFOGTV4Dataset
from v6.models.plf_ogt_v4 import PLFOGTV4Model
from v6.models.v4_axes import load_v4_proxy_contract
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path(__file__).resolve().parents[1]
H6 = 5
SEEDS = [42, 52, 62]
N_BOOT = 2000
BOOT_SEED = 42
OUT_DIR = RESULTS_DIR / "v4"


def move(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v)
            for k, v in batch.items()}


def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def main():
    configure_cuda(); dev = DEVICE
    print("=" * 60, flush=True)
    print("第三档 #10: no_residual 3-seed 消融 + CI", flush=True)
    print("=" * 60, flush=True)

    spec = load_v4_proxy_contract(REPO / "configs/v4/v4_proxy_contract.json")
    ds = PLFOGTV4Dataset("test")
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"test: {len(ds)} samples", flush=True)

    # 标签 + stay_id (与 ablations_test 口径: 6h, organ now/6h 均有效)
    all_p_seeds = []
    organ_all, organ_mask_all, stays_all = [], [], []
    for si, seed in enumerate(SEEDS):
        ckpt = REPO / f"runs/v4/abl_no_residual_s5_s{seed}/best.pt"
        if not ckpt.exists():
            print(f"  ⚠ seed {seed} checkpoint 不存在: {ckpt}", flush=True); continue
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4,
                              event_layers=2, concept_layers=1, residual_layers=1,
                              transition_layers=2, dropout=0.0, n_horizons=12,
                              r_encoder_type="slot_attention", r_n_iters=3, spec=spec,
                              n_residual=0, transition_mode="modulation", proxy_bias_init=2.0)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        model.to(dev).eval()
        sp = []; ol = []; ml = []; stl = []
        with torch.inference_mode():
            for batch in loader:
                batch = move(batch, dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    out = model(batch, stage="conditioned", future_treatment_mode="actual")
                logits = out["class_logits"][:, H6, :].float().cpu().numpy()
                sp.append(softmax_np(logits)[:, 0])
                ol.append(batch["organ"].cpu().numpy()); ml.append(batch["organ_mask"].cpu().numpy())
                stl.append(batch["stay_id"].cpu().numpy())
        all_p_seeds.append(np.concatenate(sp))
        if si == 0:
            organ_all = np.concatenate(ol); organ_mask_all = np.concatenate(ml); stays_all = np.concatenate(stl)
        print(f"  seed {seed}: done", flush=True)
        del model; torch.cuda.empty_cache()

    # ensemble
    p = np.mean(all_p_seeds, axis=0)
    # 标签: 6h, organ now/6h 均有效 (与 ablations_test 一致)
    o_now = organ_all[:, 0, :]; m_now = organ_mask_all[:, 0, :]
    o_6h = organ_all[:, H6 + 1, :]; m_6h = organ_mask_all[:, H6 + 1, :]
    valid_organ = m_now * m_6h
    delta = (valid_organ * (o_6h - o_now)).sum(axis=1)
    valid_sample = valid_organ.sum(axis=1) > 0
    y = (delta >= 2).astype(float)

    # 点估计 (仅 valid_sample)
    m = valid_sample
    auc = float(roc_auc_score(y[m], p[m]))
    ap = float(average_precision_score(y[m], p[m]))
    print(f"\n点估计: AUROC={auc:.4f} AUPRC={ap:.4f} (n={m.sum()}, prev={y[m].mean()*100:.1f}%)", flush=True)

    # cluster bootstrap CI
    sv = stays_all[m]
    idx_map = {s: np.where(sv == s)[0] for s in np.unique(sv)}
    n_st = len(idx_map)
    rng = np.random.RandomState(BOOT_SEED)
    auc_b, ap_b = [], []
    yv, pv = y[m], p[m]
    for _ in range(N_BOOT):
        sampled = rng.choice(list(idx_map.keys()), size=n_st, replace=True)
        idx = np.concatenate([idx_map[s] for s in sampled])
        yy = yv[idx]
        if len(set(yy)) < 2: continue
        try:
            auc_b.append(roc_auc_score(yy, pv[idx])); ap_b.append(average_precision_score(yy, pv[idx]))
        except Exception:
            continue
    a = 0.025
    ci = {"auroc_lo": float(np.percentile(auc_b, a * 100)), "auroc_hi": float(np.percentile(auc_b, (1 - a) * 100)),
          "auprc_lo": float(np.percentile(ap_b, a * 100)), "auprc_hi": float(np.percentile(ap_b, (1 - a) * 100)),
          "n_boot": len(auc_b), "n_clusters": n_st}
    print(f"CI: AUROC ({ci['auroc_lo']:.4f}-{ci['auroc_hi']:.4f})  AUPRC ({ci['auprc_lo']:.4f}-{ci['auprc_hi']:.4f})", flush=True)

    results = {"no_residual_3seed": {"auroc_6h": auc, "auprc_6h": ap, "n": int(m.sum()),
                                     "prev": float(y[m].mean()), "ci": ci,
                                     "seeds": SEEDS}}
    # 对比 seed42 单点 (原 ablations_test_6h)
    orig = json.load(open(OUT_DIR / "ablations_test_6h.json"))["no_residual"]
    results["_vs_seed42"] = {"auroc_6h_s42": orig["auroc_6h"], "auprc_6h_s42": orig["auprc_6h"],
                             "auroc_3seed": auc, "auprc_3seed": ap,
                             "delta_auroc": auc - orig["auroc_6h"], "delta_auprc": ap - orig["auprc_6h"]}
    print(f"\n对比 seed42: ΔAUROC={auc-orig['auroc_6h']:+.4f} ΔAUPRC={ap-orig['auprc_6h']:+.4f}", flush=True)

    out = OUT_DIR / "no_residual_3seed_ci.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"保存: {out}", flush=True)


if __name__ == "__main__":
    main()
