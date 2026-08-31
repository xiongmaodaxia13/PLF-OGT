#!/usr/bin/env python
"""MIMIC-IV frozen trajectory results (主任务跨队列闭环).

3-seed ensemble TCR organ_future → 1/3/6/12h SOFA MAE + macro-MAE + persistence baseline.
与 GMUICU frozen 口径完全一致.

输出: results_mimic/frozen_mimic_trajectory.json + frozen_mimic_trajectory.csv
"""
from __future__ import annotations
import os, sys, json, csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, r"F:/MIMIC3_1/V12")
sys.stdout.reconfigure(encoding="utf-8")

from v6.config import DEVICE, configure_cuda
from v6.data.dataset import collate
from v6.data.mimic_dataset import MIMICDataset
from v6.models.plf_ogt_v4 import PLFOGTV4Model
from v6.models.v4_axes import load_v4_proxy_contract

REPO = Path(r"F:/MIMIC3_1/V12")
OUTDIR = Path(r"F:/MIMIC3_1/V13/results_mimic")
SEEDS = [42, 52, 62]
HORIZONS = [(1, 0), (3, 2), (6, 5), (12, 11)]


def move(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v) for k, v in batch.items()}


def main():
    configure_cuda(); dev = DEVICE
    print("=" * 70, flush=True)
    print("MIMIC-IV frozen trajectory (3-seed TCR ensemble)", flush=True)
    print("=" * 70, flush=True)
    spec = load_v4_proxy_contract(REPO / "configs/v4/v4_proxy_contract.json")
    ds = MIMICDataset(split="test")
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"MIMIC test: {len(ds)}\n", flush=True)

    # 标签
    all_organ, all_mask = [], []
    for b in loader:
        all_organ.append(b["organ"].numpy()); all_mask.append(b["organ_mask"].numpy())
    organ = np.concatenate(all_organ); omask = np.concatenate(all_mask)
    o_now = organ[:, 0, :]; m_now = omask[:, 0, :]

    # 3-seed TCR ensemble organ_future
    print("=== PLF TCR 3-seed ===", flush=True)
    ens_organ = []
    for seed in SEEDS:
        ck = torch.load(REPO / f"runs_mimic/v4/full_s5_s{seed}/best.pt", map_location="cpu", weights_only=False)
        model = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4, n_residual=6,
                              event_layers=2, concept_layers=1, residual_layers=1,
                              transition_layers=2, dropout=0.0, transition_mode="modulation",
                              n_horizons=12, r_encoder_type="slot_attention", r_n_iters=3, spec=spec)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        model.to(dev).eval()
        ps = []
        with torch.inference_mode():
            for b in loader:
                b = move(b, dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    out = model(b, stage="conditioned", future_treatment_mode="actual")
                ps.append(out["organ_future"][:, :12, :].float().cpu().numpy())
        ens_organ.append(np.concatenate(ps))
        print(f"  seed {seed}: done", flush=True)
        del model; torch.cuda.empty_cache()
    pred = np.mean(ens_organ, axis=0)  # (N, 12, 6)

    # 计算各时距
    print("\n=== Trajectory MAE (frozen) ===", flush=True)
    results = []
    for h, hi in HORIZONS:
        o_h = organ[:, hi + 1, :]; m_h = omask[:, hi + 1, :]
        valid = (m_now * m_h).sum(axis=1) > 0
        if valid.sum() == 0: continue

        p = pred[valid, hi, :]; t = o_h[valid]; m = m_h[valid]
        # SOFA total MAE (逐锚点绝对值均值)
        plf_sofa = np.abs((p * m).sum(axis=1) - (t * m).sum(axis=1)).mean()
        persist_sofa = np.abs((o_now[valid] * m).sum(axis=1) - (t * m).sum(axis=1)).mean()
        # macro MAE
        organ_maes_plf = [np.abs(p[m[:, o] > 0, o] - t[m[:, o] > 0, o]).mean() if (m[:, o] > 0).sum() > 0 else float("nan") for o in range(6)]
        organ_maes_per = [np.abs(o_now[valid][m[:, o] > 0, o] - t[m[:, o] > 0, o]).mean() if (m[:, o] > 0).sum() > 0 else float("nan") for o in range(6)]
        macro_plf = np.nanmean(organ_maes_plf); macro_per = np.nanmean(organ_maes_per)

        r = {"horizon": h, "valid_n": int(valid.sum()),
             "plf_sofa": float(plf_sofa), "persist_sofa": float(persist_sofa),
             "plf_macro": float(macro_plf), "persist_macro": float(macro_per),
             "delta_sofa": float(plf_sofa - persist_sofa), "delta_macro": float(macro_plf - macro_per)}
        results.append(r)
        print(f"  {h:>2}h: PLF sofa={plf_sofa:.3f} macro={macro_plf:.3f} | persist sofa={persist_sofa:.3f} macro={macro_per:.3f} | n={valid.sum()}", flush=True)

    # changed-state (6h)
    print("\n=== Changed-state (6h) ===", flush=True)
    h6 = 5
    o_6h = organ[:, h6 + 1, :]; m_6h = omask[:, h6 + 1, :]
    valid6 = (m_now * m_6h).sum(axis=1) > 0
    delta_sofa_6h = ((m_now * m_6h) * (o_6h - o_now)).sum(axis=1)
    p6 = pred[valid6, h6, :]; t6 = o_6h[valid6]; m6 = m_6h[valid6]
    changed = {"all": {}, "unchanged": {}, "changed_ge1": {}, "worsened_ge2": {}}
    for label, mask_extra in [("all", valid6),
                               ("unchanged", valid6 & (delta_sofa_6h == 0)),
                               ("changed_ge1", valid6 & (np.abs(delta_sofa_6h) >= 1)),
                               ("worsened_ge2", valid6 & (delta_sofa_6h >= 2))]:
        if mask_extra.sum() < 10: continue
        p_sofa = np.abs((p6[mask_extra[valid6]] * m6[mask_extra[valid6]]).sum(axis=1) - (t6[mask_extra[valid6]] * m6[mask_extra[valid6]]).sum(axis=1)).mean()
        per_sofa = np.abs((o_now[mask_extra] * m_6h[mask_extra]).sum(axis=1) - (o_6h[mask_extra] * m_6h[mask_extra]).sum(axis=1)).mean()
        changed[label] = {"n": int(mask_extra.sum()), "plf_sofa": float(p_sofa), "persist_sofa": float(per_sofa), "delta": float(p_sofa - per_sofa)}
        print(f"  {label}: n={mask_extra.sum()} PLF={p_sofa:.3f} persist={per_sofa:.3f} Δ={p_sofa-per_sofa:+.3f}", flush=True)

    # 保存
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_json = {"multi_horizon": results, "changed_state_6h": changed,
                "note": "MIMIC-IV 3-seed TCR ensemble, frozen, same caliber as GMUICU"}
    json.dump(out_json, open(OUTDIR / "frozen_mimic_trajectory.json", "w"), indent=2, ensure_ascii=False, default=float)

    with open(OUTDIR / "frozen_mimac_trajectory.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Horizon", "Valid n", "PLF SOFA MAE", "PLF macro MAE", "Persistence SOFA MAE", "Persistence macro MAE", "Δ SOFA", "Δ macro"])
        for r in results:
            w.writerow([f"{r['horizon']}h", r["valid_n"], f"{r['plf_sofa']:.3f}", f"{r['plf_macro']:.3f}",
                        f"{r['persist_sofa']:.3f}", f"{r['persist_macro']:.3f}", f"{r['delta_sofa']:+.3f}", f"{r['delta_macro']:+.3f}"])

    print(f"\n保存: {OUTDIR}/frozen_mimic_trajectory.json + .csv", flush=True)


if __name__ == "__main__":
    main()
