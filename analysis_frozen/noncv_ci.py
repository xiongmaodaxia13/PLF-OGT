#!/usr/bin/env python
"""non-CV 判别增益的配对 cluster bootstrap CI (修改意见4 问题2 补充).

复用 allhorizon_ci_logits.npz (TCR/OLP 3-seed ensemble logits) + rollout organ 标签,
为 non-CV ΔSOFA>=2 结局的 TCR−care-off ΔAUROC / ΔAUPRC / ΔBrier 补 95% CI.

输出: results/v4/frozen_noncv_ci.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

sys.stdout.reconfigure(encoding="utf-8")

V12 = Path(r"F:/MIMIC3_1/V12")
OUT = Path(r"F:/MIMIC3_1/V13/results/v4/frozen_noncv_ci.json")
H6 = 5
N_BOOT = 2000; BOOT_SEED = 42


def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def main():
    d = np.load(r"F:/MIMIC3_1/V13/results/v4/allhorizon_ci_logits.npz", allow_pickle=True)
    tcr_logits, olp_logits = d["tcr"], d["olp"]
    stays = d["stays"]

    rt = np.load(V12 / "data/gold/labels/rollout_targets_gmu_v6.npz", allow_pickle=True)
    test_mask = rt["sample_split"].astype(str) == "confirmation"
    organ = rt["organ"][test_mask]
    omask = rt["organ_mask"][test_mask]
    assert organ.shape[0] == tcr_logits.shape[0]

    o_now = organ[:, 0, :]; m_now = omask[:, 0, :]
    o_6h = organ[:, H6 + 1, :]; m_6h = omask[:, H6 + 1, :]

    mask_ncv = np.ones(6, dtype=bool); mask_ncv[1] = False
    delta_ncv = ((m_now * m_6h)[:, mask_ncv] * (o_6h[:, mask_ncv] - o_now[:, mask_ncv])).sum(axis=1)
    y_ncv = (delta_ncv >= 2).astype(float)
    valid_ncv = (m_now * m_6h)[:, mask_ncv].sum(axis=1) > 0

    # class_mask @6h
    cmask = rt["class_mask"][test_mask]
    m6_cls = cmask[:, H6] > 0
    v = valid_ncv & m6_cls

    tcr_p = softmax_np(tcr_logits[:, H6, :])[:, 0]
    co_p = softmax_np(olp_logits[:, H6, :])[:, 0]

    print(f"n_valid={v.sum():,}, prev={y_ncv[v].mean()*100:.2f}%", flush=True)

    yv = y_ncv[v]; tv = tcr_p[v]; cv = co_p[v]; sv = stays[v]

    def metrics(yy, tp, cp):
        auc_t = roc_auc_score(yy, tp); auc_c = roc_auc_score(yy, cp)
        ap_t = average_precision_score(yy, tp); ap_c = average_precision_score(yy, cp)
        br_t = brier_score_loss(yy, np.clip(tp, 1e-8, 1 - 1e-8))
        br_c = brier_score_loss(yy, np.clip(cp, 1e-8, 1 - 1e-8))
        return auc_t - auc_c, ap_t - ap_c, br_c - br_t

    d_auc, d_ap, d_br = metrics(yv, tv, cv)
    print(f"点估计: ΔAUROC={d_auc:+.4f}  ΔAUPRC={d_ap:+.4f}  ΔBrier={d_br:+.4f}", flush=True)

    # 配对 stay-cluster bootstrap
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
            "n_valid": int(v.sum()), "prev": float(y_ncv[v].mean()),
            "delta_auroc": float(d_auc), "delta_auroc_ci": [float(np.percentile(b_auc, 2.5)), float(np.percentile(b_auc, 97.5))],
            "delta_auprc": float(d_ap), "delta_auprc_ci": [float(np.percentile(b_ap, 2.5)), float(np.percentile(b_ap, 97.5))],
            "delta_brier": float(d_br), "delta_brier_ci": [float(np.percentile(b_br, 2.5)), float(np.percentile(b_br, 97.5))],
        },
        "n_boot": len(b_auc), "n_skip": n_skip, "boot_seed": BOOT_SEED,
        "note": "non-CV ΔSOFA≥2 (排除循环器官) 结局; TCR vs care-off (zero mode); "
                "3-seed ensemble logits 缓存; paired stay-cluster bootstrap",
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    r = out["non_cv"]
    print(f"\nΔAUROC = {r['delta_auroc']:+.4f} (95% CI {r['delta_auroc_ci'][0]:+.4f} ~ {r['delta_auroc_ci'][1]:+.4f})")
    print(f"ΔAUPRC = {r['delta_auprc']:+.4f} (95% CI {r['delta_auprc_ci'][0]:+.4f} ~ {r['delta_auprc_ci'][1]:+.4f})")
    print(f"ΔBrier = {r['delta_brier']:+.4f} (95% CI {r['delta_brier_ci'][0]:+.4f} ~ {r['delta_brier_ci'][1]:+.4f})")
    ci_a = r['delta_auroc_ci']
    print(f"\nCI 是否含零: AUROC {'是' if ci_a[0] <= 0 <= ci_a[1] else '否'}")
    print(f"保存: {OUT}")


if __name__ == "__main__":
    main()
