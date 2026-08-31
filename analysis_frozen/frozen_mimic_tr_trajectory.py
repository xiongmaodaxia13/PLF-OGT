#!/usr/bin/env python
"""MIMIC-IV Transformer trajectory comparator.

补齐审稿人最关心的跨队列缺口:
  GMUICU: changed-state中 Transformer < PLF < persistence (in MAE)
  MIMIC: 是否也成立?

3-seed Transformer TCR ensemble organ_future → 6h changed-state分层.
与 MIMIC PLF trajectory (已有) + persistence 三方比较.

输出: results_mimic/frozen_mimic_tr_trajectory.json
"""
from __future__ import annotations
import os, sys, json
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
from v6.models.std_transformer import StdTransformer

REPO = Path(r"F:/MIMIC3_1/V12")
OUTDIR = Path(r"F:/MIMIC3_1/V13/results_mimic")
SEEDS = [42, 52, 62]; H6 = 5


def move(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v) for k, v in batch.items()}


def main():
    configure_cuda(); dev = DEVICE
    print("=" * 70, flush=True)
    print("MIMIC-IV Transformer trajectory (3-seed TCR)", flush=True)
    print("=" * 70, flush=True)
    ds = MIMICDataset(split="test")
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"MIMIC test: {len(ds)}\n", flush=True)

    # 标签
    all_organ, all_mask = [], []
    for b in loader:
        all_organ.append(b["organ"].numpy()); all_mask.append(b["organ_mask"].numpy())
    organ = np.concatenate(all_organ); omask = np.concatenate(all_mask)
    o_now = organ[:, 0, :]; m_now = omask[:, 0, :]

    # Transformer 3-seed TCR
    print("=== Transformer TCR 3-seed ===", flush=True)
    ens = []
    for seed in SEEDS:
        ck = torch.load(REPO / f"runs/baselines/transformer_tcr_s{seed}/best.pt", map_location="cpu", weights_only=False)
        model = StdTransformer(prior_dim=14, mode="TCR")
        model.load_state_dict(ck.get("model_state_dict", ck))
        model.to(dev).eval()
        ps = []
        with torch.inference_mode():
            for b in loader:
                b = move(b, dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    out = model(b)
                ps.append(out["organ_future"][:, :12, :].float().cpu().numpy())
        ens.append(np.concatenate(ps))
        print(f"  TR seed {seed}: done", flush=True)
        del model; torch.cuda.empty_cache()
    pred = np.mean(ens, axis=0)  # (N,12,6)

    # 6h changed-state
    print("\n=== 6h changed-state ===", flush=True)
    o_6h = organ[:, H6 + 1, :]; m_6h = omask[:, H6 + 1, :]
    valid = (m_now * m_6h).sum(axis=1) > 0
    delta_sofa = ((m_now * m_6h) * (o_6h - o_now)).sum(axis=1)

    p6 = pred[valid, H6, :]; t6 = o_6h[valid]; m6 = m_6h[valid]
    subsets = [
        ("all", valid),
        ("unchanged", valid & (delta_sofa == 0)),
        ("changed_ge1", valid & (np.abs(delta_sofa) >= 1)),
        ("worsened_ge2", valid & (delta_sofa >= 2)),
    ]
    results = {}
    for label, mask in subsets:
        n = int(mask.sum())
        if n < 10: continue
        p = pred[mask, H6, :]; t = o_6h[mask]; m = m_6h[mask]
        tr_sofa = float(np.abs((p * m).sum(axis=1) - (t * m).sum(axis=1)).mean())
        persist_sofa = float(np.abs((o_now[mask] * m).sum(axis=1) - (t * m).sum(axis=1)).mean())
        results[label] = {"n": n, "transformer_sofa": tr_sofa, "persistence_sofa": persist_sofa}
        print(f"  {label:<16} n={n:<8} TR={tr_sofa:.3f}  persist={persist_sofa:.3f}", flush=True)

    # 也算多时距
    print("\n=== 多时距 ===", flush=True)
    multi = []
    for h, hi in [(1,0),(3,2),(6,5),(12,11)]:
        o_h = organ[:, hi+1, :]; m_h = omask[:, hi+1, :]
        v = (m_now * m_h).sum(axis=1) > 0
        if v.sum() == 0: continue
        tr = float(np.abs((pred[v, hi, :] * m_h[v]).sum(axis=1) - (o_h[v] * m_h[v]).sum(axis=1)).mean())
        per = float(np.abs((o_now[v] * m_h[v]).sum(axis=1) - (o_h[v] * m_h[v]).sum(axis=1)).mean())
        multi.append({"horizon": h, "tr_sofa": tr, "persist_sofa": per})
        print(f"  {h}h: TR={tr:.3f} persist={per:.3f}", flush=True)

    out = {"changed_state_6h": results, "multi_horizon": multi}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUTDIR / "frozen_mimic_tr_trajectory.json", "w"), indent=2, ensure_ascii=False, default=float)
    print(f"\n保存: {OUTDIR / 'frozen_mimic_tr_trajectory.json'}", flush=True)


if __name__ == "__main__":
    main()
