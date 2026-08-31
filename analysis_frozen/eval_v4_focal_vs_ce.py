#!/usr/bin/env python
"""#13: focal loss 模型 vs CE 主模型 的 6h test 对比.

focal: runs/v4/abl_full_s5_s42 (full + focal_gamma=2.0)
CE:    runs/v4/full_s5_s42 (现有主模型, 普通 CE)
两者均 seed 42, 完整模型, TCR 推理, 6h test.

输出: results/v4/focal_vs_ce.json
用法: CUDA_VISIBLE_DEVICES=0 python scripts/eval_v4_focal_vs_ce.py
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

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


def move(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v)
            for k, v in batch.items()}


def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def eval_model(loader, dev, spec, ckpt_rel):
    ck = torch.load(REPO / ckpt_rel, map_location="cpu", weights_only=False)
    model = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4, n_residual=6,
                          event_layers=2, concept_layers=1, residual_layers=1,
                          transition_layers=2, dropout=0.0, n_horizons=12,
                          r_encoder_type="slot_attention", r_n_iters=3, spec=spec,
                          transition_mode="modulation", proxy_bias_init=2.0)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.to(dev).eval()
    all_p, ol, ml = [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = move(batch, dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                out = model(batch, stage="conditioned", future_treatment_mode="actual")
            logits = out["class_logits"][:, H6, :].float().cpu().numpy()
            all_p.append(softmax_np(logits)[:, 0])
            ol.append(batch["organ"].cpu().numpy()); ml.append(batch["organ_mask"].cpu().numpy())
    del model; torch.cuda.empty_cache()
    p = np.concatenate(all_p)
    organ = np.concatenate(ol); omask = np.concatenate(ml)
    o_now, m_now = organ[:, 0, :], omask[:, 0, :]
    o_6h, m_6h = organ[:, H6+1, :], omask[:, H6+1, :]
    valid = (m_now * m_6h).sum(axis=1) > 0
    delta = ((m_now * m_6h) * (o_6h - o_now)).sum(axis=1)
    y = (delta >= 2).astype(float)
    return p[valid], y[valid]


def main():
    configure_cuda(); dev = DEVICE
    print("=" * 60, flush=True)
    print("第三档 #13: focal vs CE 的 6h test 对比 (seed42, full)", flush=True)
    print("=" * 60, flush=True)
    spec = load_v4_proxy_contract(REPO / "configs/v4/v4_proxy_contract.json")
    ds = PLFOGTV4Dataset("test")
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"test: {len(ds)}\n", flush=True)

    print("[CE 主模型]...", flush=True)
    p_ce, y_ce = eval_model(loader, dev, spec, "runs/v4/full_s5_s42/best.pt")
    auc_ce = roc_auc_score(y_ce, p_ce); ap_ce = average_precision_score(y_ce, p_ce)
    print(f"  CE:    AUROC={auc_ce:.4f} AUPRC={ap_ce:.4f} (n={len(y_ce)}, prev={y_ce.mean()*100:.1f}%)", flush=True)

    print("\n[focal γ=2.0]...", flush=True)
    p_fo, y_fo = eval_model(loader, dev, spec, "runs/v4/abl_full_s5_s42/best.pt")
    auc_fo = roc_auc_score(y_fo, p_fo); ap_fo = average_precision_score(y_fo, p_fo)
    print(f"  focal: AUROC={auc_fo:.4f} AUPRC={ap_fo:.4f}", flush=True)

    # F1@锁定阈值比较 (CE 用原阈值0.155; focal 用自身 val 锁定阈值需另算, 这里用 AUPRC 主指标判)
    print(f"\n对比 (focal - CE): ΔAUROC={auc_fo-auc_ce:+.4f} ΔAUPRC={ap_fo-ap_ce:+.4f}", flush=True)
    verdict = "focal 改善" if (auc_fo > auc_ce and ap_fo > ap_ce) else ("focal 持平" if abs(ap_fo-ap_ce)<0.01 else "focal 未改善/更差")
    print(f"结论: {verdict}", flush=True)

    out = RESULTS_DIR / "v4" / "focal_vs_ce.json"
    results = {"CE": {"auroc": auc_ce, "auprc": ap_ce},
               "focal_gamma2": {"auroc": auc_fo, "auprc": ap_fo},
               "delta_focal_vs_ce": {"auroc": auc_fo-auc_ce, "auprc": ap_fo-ap_ce},
               "verdict": verdict, "n": int(len(y_ce)), "prev": float(y_ce.mean())}
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n保存: {out}", flush=True)


if __name__ == "__main__":
    main()
