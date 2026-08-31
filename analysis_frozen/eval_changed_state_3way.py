#!/usr/bin/env python
"""Changed-state 三方分析 (single source of truth): PLF vs Transformer vs persistence.

一次推理 PLF(TCR 3-seed) + Transformer(TCR 3-seed), 然后分三层算 changed-state MAE:
  - 全部锚点
  - ΔSOFA=0 (不变)
  - |ΔSOFA|≥1 (变化)
  - ΔSOFA≥2 (恶化)

输出同时作为正文 Table 2 / S3 的最终数值来源.
输出: results/v4/changed_state_3way_frozen.json
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
from v6.data.v4_dataset import PLFOGTV4Dataset
from v6.models.plf_ogt_v4 import PLFOGTV4Model
from v6.models.std_transformer import StdTransformer
from v6.models.v4_axes import load_v4_proxy_contract

REPO = Path(r"F:/MIMIC3_1/V12")
OUT = Path(r"F:/MIMIC3_1/V13/results/v4/changed_state_3way_frozen.json")
SEEDS = [42, 52, 62]; H6 = 5


def move(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v) for k, v in batch.items()}


def run_plf(loader, dev, spec):
    ens = []
    for seed in SEEDS:
        ck = torch.load(REPO / f"runs/v4/full_s5_s{seed}/best.pt", map_location="cpu", weights_only=False)
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
                ps.append(out["organ_future"][:, H6, :].float().cpu().numpy())
        ens.append(np.concatenate(ps))
        print(f"  PLF seed {seed}: done", flush=True)
        del model; torch.cuda.empty_cache()
    return np.mean(ens, axis=0)


def run_tr(loader, dev):
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
                ps.append(out["organ_future"][:, H6, :].float().cpu().numpy())
        ens.append(np.concatenate(ps))
        print(f"  TR seed {seed}: done", flush=True)
        del model; torch.cuda.empty_cache()
    return np.mean(ens, axis=0)


def sofa_mae(pred, true, mask_valid):
    """逐锚点 SOFA 总分绝对误差. pred/true: (N,6), 返回 (N,) 数组."""
    return np.abs((pred * mask_valid).sum(axis=1) - (true * mask_valid).sum(axis=1))


def main():
    configure_cuda(); dev = DEVICE
    print("=" * 70, flush=True)
    print("Changed-state 三方分析 (frozen single source of truth)", flush=True)
    print("=" * 70, flush=True)
    spec = load_v4_proxy_contract(REPO / "configs/v4/v4_proxy_contract.json")
    ds = PLFOGTV4Dataset("test")
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"test: {len(ds)}\n", flush=True)

    # 标签
    all_organ, all_mask = [], []
    for b in loader:
        all_organ.append(b["organ"].numpy()); all_mask.append(b["organ_mask"].numpy())
    organ = np.concatenate(all_organ); omask = np.concatenate(all_mask)
    o_now = organ[:, 0, :]; m_now = omask[:, 0, :]
    o_6h = organ[:, H6 + 1, :]; m_6h = omask[:, H6 + 1, :]
    valid = (m_now * m_6h).sum(axis=1) > 0
    delta_sofa = ((m_now * m_6h) * (o_6h - o_now)).sum(axis=1)  # 有符号 ΔSOFA

    # persistence pred = anchor
    persist_pred = o_now

    # PLF TCR
    print("=== PLF-OGT TCR ===", flush=True)
    plf_pred = run_plf(loader, dev, spec)

    # Transformer TCR
    print("\n=== Transformer TCR ===", flush=True)
    tr_pred = run_tr(loader, dev)

    # 三方逐锚点 SOFA MAE
    mask6 = m_6h  # (N,6) 6h 器官有效掩码
    persist_mae = sofa_mae(persist_pred, o_6h, mask6)
    plf_mae = sofa_mae(plf_pred, o_6h, mask6)
    tr_mae = sofa_mae(tr_pred, o_6h, mask6)

    # 分层
    print(f"\n{'='*80}", flush=True)
    print(f"{'子集':<24}{'n':<8}{'persist':<12}{'Transformer':<14}{'PLF-OGT':<12}{'PLF−persist':<12}{'PLF−TR':<10}")
    print("-" * 80)
    results = {}
    subsets = [
        ("all", "全部锚点", valid),
        ("unchanged", "ΔSOFA=0 (不变)", valid & (delta_sofa == 0)),
        ("changed_ge1", "|ΔSOFA|≥1", valid & (np.abs(delta_sofa) >= 1)),
        ("worsened_ge2", "ΔSOFA≥2 (恶化)", valid & (delta_sofa >= 2)),
    ]
    for key, label, mask in subsets:
        n = int(mask.sum())
        if n < 10:
            continue
        p = persist_mae[mask].mean()
        t = tr_mae[mask].mean()
        l = plf_mae[mask].mean()
        results[key] = {"n": n, "pct": n / valid.sum() * 100, "persistence": p, "transformer": t, "plf_ogt": l,
                        "delta_plf_persist": l - p, "delta_plf_tr": l - t}
        print(f"{label:<24}{n:<8}{p:<12.4f}{t:<14.4f}{l:<12.4f}{l-p:<+12.4f}{l-t:<+10.4f}", flush=True)

    print("=" * 80, flush=True)

    # 全局 MAE (single source of truth for Table 2)
    results["_frozen_table2"] = {
        "plf_sofa_mae_6h": float(plf_mae[valid].mean()),
        "tr_sofa_mae_6h": float(tr_mae[valid].mean()),
        "persist_sofa_mae_6h": float(persist_mae[valid].mean()),
        "valid_anchors": int(valid.sum()),
    }
    print(f"\nFrozen Table 2 (6h SOFA MAE, valid={valid.sum()}):")
    print(f"  persistence: {results['_frozen_table2']['persist_sofa_mae_6h']:.4f}")
    print(f"  Transformer: {results['_frozen_table2']['tr_sofa_mae_6h']:.4f}")
    print(f"  PLF-OGT: {results['_frozen_table2']['plf_sofa_mae_6h']:.4f}")

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=lambda o: float(o) if hasattr(o,"item") else str(o)), encoding="utf-8")
    print(f"\n保存: {OUT}", flush=True)


if __name__ == "__main__":
    main()
