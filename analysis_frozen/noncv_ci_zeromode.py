#!/usr/bin/env python
"""non-CV 判别增益 CI — zero-mode care-off 口径 (与 frozen_organ_noncv.json 一致).

allhorizon_ci_logits.npz 的 olp 是另一实现 (total ΔAUROC +0.099),
frozen_organ_noncv.json 用 future_treatment_mode="zero" (+0.081).
本脚本重推 zero-mode 3-seed logits, 复现 +0.0131 后给 CI.

输出: results/v4/frozen_noncv_ci.json (覆盖, zero-mode 口径)
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, r"F:/MIMIC3_1/V12")
sys.stdout.reconfigure(encoding="utf-8")

from v6.config import DEVICE, configure_cuda
from v6.data.dataset import collate
from v6.data.v4_dataset import PLFOGTV4Dataset
from v6.models.plf_ogt_v4 import PLFOGTV4Model
from v6.models.v4_axes import load_v4_proxy_contract

REPO = Path(r"F:/MIMIC3_1/V12")
OUT = Path(r"F:/MIMIC3_1/V13/results/v4/frozen_noncv_ci.json")
SEEDS = [42, 52, 62]; H6 = 5
N_BOOT = 2000; BOOT_SEED = 42


def move(b, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v) for k, v in b.items()}


def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def main():
    configure_cuda(); dev = DEVICE
    print("=" * 60, flush=True)
    print("non-CV CI (zero-mode care-off, 复现 frozen_organ_noncv 口径)", flush=True)
    print("=" * 60, flush=True)
    spec = load_v4_proxy_contract(REPO / "configs/v4/v4_proxy_contract.json")
    ds = PLFOGTV4Dataset("test")
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)

    # 标签
    all_organ, all_mask, all_cmask, all_stays = [], [], [], []
    for b in loader:
        all_organ.append(b["organ"].numpy()); all_mask.append(b["organ_mask"].numpy())
        all_cmask.append(b["class_mask"].numpy()); all_stays.append(b["stay_id"].numpy())
    organ = np.concatenate(all_organ); omask = np.concatenate(all_mask)
    cmask = np.concatenate(all_cmask); stays = np.concatenate(all_stays)

    # 推理: TCR (actual) + care-off (zero), 3-seed ensemble logits
    ens = {"actual": [], "zero": []}
    for mode in ["actual", "zero"]:
        for seed in SEEDS:
            ck = torch.load(REPO / f"runs/v4/full_s5_s{seed}/best.pt", map_location="cpu", weights_only=False)
            m = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4, n_residual=6,
                              event_layers=2, concept_layers=1, residual_layers=1,
                              transition_layers=2, dropout=0.0, transition_mode="modulation",
                              n_horizons=12, r_encoder_type="slot_attention", r_n_iters=3, spec=spec)
            m.load_state_dict(ck["model_state_dict"], strict=False)
            m.to(dev).eval()
            ls = []
            with torch.inference_mode():
                for b in loader:
                    b = move(b, dev)
                    with torch.autocast(dev.type, dtype=torch.bfloat16):
                        out = m(b, stage="conditioned", future_treatment_mode=mode)
                    ls.append(out["class_logits"].float().cpu().numpy())
            ens[mode].append(np.concatenate(ls))
            print(f"  {mode} seed {seed}: done", flush=True)
            del m; torch.cuda.empty_cache()

    tcr_logits = np.mean(ens["actual"], axis=0)
    co_logits = np.mean(ens["zero"], axis=0)

    o_now = organ[:, 0, :]; m_now = omask[:, 0, :]
    o_6h = organ[:, H6 + 1, :]; m_6h = omask[:, H6 + 1, :]
    mask_ncv = np.ones(6, dtype=bool); mask_ncv[1] = False
    delta_ncv = ((m_now * m_6h)[:, mask_ncv] * (o_6h[:, mask_ncv] - o_now[:, mask_ncv])).sum(axis=1)
    y_ncv = (delta_ncv >= 2).astype(float)
    valid_ncv = (m_now * m_6h)[:, mask_ncv].sum(axis=1) > 0
    m6_cls = cmask[:, H6] > 0
    v = valid_ncv & m6_cls

    tcr_p = softmax_np(tcr_logits[:, H6, :])[:, 0]
    co_p = softmax_np(co_logits[:, H6, :])[:, 0]
    yv = y_ncv[v]; tv = tcr_p[v]; cv = co_p[v]; sv = stays[v]
    print(f"\nn_valid={v.sum():,}, prev={yv.mean()*100:.2f}%", flush=True)

    # total SOFA 对照 (验证 zero-mode 复现: 应 ≈ +0.081)
    y6 = ((delta_sofa := None) or (omask[:, H6 + 1].sum(1) > 0))  # placeholder, 用 delta 数组
    # 重新取 delta_sofa
    rt_delta = np.concatenate([b["delta_sofa"].numpy() for b in DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)])
    y6_total = (rt_delta[:, H6] >= 2).astype(float)
    dt = roc_auc_score(y6_total[m6_cls], tcr_p[m6_cls]) - roc_auc_score(y6_total[m6_cls], co_p[m6_cls])
    print(f"total ΔAUROC (验证) = {dt:+.4f}  [zero-mode 应≈+0.081]", flush=True)

    def metrics(yy, tp, cp):
        auc_t = roc_auc_score(yy, tp); auc_c = roc_auc_score(yy, cp)
        ap_t = average_precision_score(yy, tp); ap_c = average_precision_score(yy, cp)
        br_t = brier_score_loss(yy, np.clip(tp, 1e-8, 1 - 1e-8))
        br_c = brier_score_loss(yy, np.clip(cp, 1e-8, 1 - 1e-8))
        return auc_t - auc_c, ap_t - ap_c, br_c - br_t

    d_auc, d_ap, d_br = metrics(yv, tv, cv)
    print(f"non-CV 点估计: ΔAUROC={d_auc:+.4f}  ΔAUPRC={d_ap:+.4f}  ΔBrier={d_br:+.4f}", flush=True)

    uniq = np.unique(sv)
    idx_map = {s: np.where(sv == s)[0] for s in uniq}
    rng = np.random.RandomState(BOOT_SEED)
    b_auc, b_ap, b_br = [], [], []
    n_skip = 0
    for _ in range(N_BOOT):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_map[s] for s in sampled])
        yy = yv[idx]
        if len(set(yy)) < 2 or yy.sum() < 3:
            n_skip += 1
            continue
        try:
            a, p, b = metrics(yy, tv[idx], cv[idx])
            b_auc.append(a); b_ap.append(p); b_br.append(b)
        except Exception:
            n_skip += 1

    out = {
        "non_cv": {
            "n_valid": int(v.sum()), "prev": float(yv.mean()),
            "delta_auroc": float(d_auc), "delta_auroc_ci": [float(np.percentile(b_auc, 2.5)), float(np.percentile(b_auc, 97.5))],
            "delta_auprc": float(d_ap), "delta_auprc_ci": [float(np.percentile(b_ap, 2.5)), float(np.percentile(b_ap, 97.5))],
            "delta_brier": float(d_br), "delta_brier_ci": [float(np.percentile(b_br, 2.5)), float(np.percentile(b_br, 97.5))],
        },
        "total_check": {"delta_auroc_total": float(dt)},
        "n_boot": len(b_auc), "n_skip": n_skip, "boot_seed": BOOT_SEED,
        "note": "non-CV ΔSOFA≥2 (排除循环器官); TCR vs care-off future_treatment_mode=zero; "
                "3-seed ensemble; paired stay-cluster bootstrap. 与 frozen_organ_noncv.json 同口径.",
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    r = out["non_cv"]
    print(f"\nΔAUROC = {r['delta_auroc']:+.4f} (95% CI {r['delta_auroc_ci'][0]:+.4f} ~ {r['delta_auroc_ci'][1]:+.4f})")
    print(f"ΔAUPRC = {r['delta_auprc']:+.4f} (95% CI {r['delta_auprc_ci'][0]:+.4f} ~ {r['delta_auprc_ci'][1]:+.4f})")
    print(f"ΔBrier = {r['delta_brier']:+.4f} (95% CI {r['delta_brier_ci'][0]:+.4f} ~ {r['delta_brier_ci'][1]:+.4f})")
    print(f"保存: {OUT}")


if __name__ == "__main__":
    main()
