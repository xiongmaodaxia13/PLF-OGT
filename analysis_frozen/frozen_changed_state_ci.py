#!/usr/bin/env python
"""Changed-state paired bootstrap CI: PLF/TR/persistence × GMUICU/MIMIC.

两队列 × 三方模型的 changed-state paired bootstrap CI.
用于回答审稿人: "跨队列排序反转是否 robust to sampling uncertainty?"

输出: results/v4/frozen_changed_state_ci.json (GMUICU)
      results_mimic/frozen_changed_state_ci.json (MIMIC)
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, r"F:/MIMIC3_1/V12")
sys.stdout.reconfigure(encoding="utf-8")

from v6.config import DEVICE, configure_cuda
from v6.data.dataset import collate
from v6.models.plf_ogt_v4 import PLFOGTV4Model
from v6.models.std_transformer import StdTransformer
from v6.models.v4_axes import load_v4_proxy_contract

REPO = Path(r"F:/MIMIC3_1/V12")
OUTDIR_G = Path(r"F:/MIMIC3_1/V13/results/v4")
OUTDIR_M = Path(r"F:/MIMIC3_1/V13/results_mimic")
SEEDS = [42, 52, 62]; H6 = 5; N_BOOT = 2000; BOOT_SEED = 42


def move(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v) for k, v in batch.items()}


def run_plf(loader, dev, spec, repo, seeds):
    ens = []
    for seed in seeds:
        ck = torch.load(repo / f"runs/v4/full_s5_s{seed}/best.pt", map_location="cpu", weights_only=False)
        m = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4, n_residual=6,
                          event_layers=2, concept_layers=1, residual_layers=1,
                          transition_layers=2, dropout=0.0, transition_mode="modulation",
                          n_horizons=12, r_encoder_type="slot_attention", r_n_iters=3, spec=spec)
        m.load_state_dict(ck["model_state_dict"], strict=False); m.to(dev).eval()
        ps = []
        with torch.inference_mode():
            for b in loader:
                b = move(b, dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    out = m(b, stage="conditioned", future_treatment_mode="actual")
                ps.append(out["organ_future"][:, H6, :].float().cpu().numpy())
        ens.append(np.concatenate(ps))
        print(f"    PLF seed {seed}: done", flush=True)
        del m; torch.cuda.empty_cache()
    return np.mean(ens, axis=0)


def run_plf_mimic(loader, dev, spec, repo, seeds):
    ens = []
    for seed in seeds:
        ck = torch.load(repo / f"runs_mimic/v4/full_s5_s{seed}/best.pt", map_location="cpu", weights_only=False)
        m = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4, n_residual=6,
                          event_layers=2, concept_layers=1, residual_layers=1,
                          transition_layers=2, dropout=0.0, transition_mode="modulation",
                          n_horizons=12, r_encoder_type="slot_attention", r_n_iters=3, spec=spec)
        m.load_state_dict(ck["model_state_dict"], strict=False); m.to(dev).eval()
        ps = []
        with torch.inference_mode():
            for b in loader:
                b = move(b, dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    out = m(b, stage="conditioned", future_treatment_mode="actual")
                ps.append(out["organ_future"][:, H6, :].float().cpu().numpy())
        ens.append(np.concatenate(ps))
        print(f"    MIMIC PLF seed {seed}: done", flush=True)
        del m; torch.cuda.empty_cache()
    return np.mean(ens, axis=0)


def run_tr(loader, dev, repo, seeds, mimic=False):
    ens = []
    prefix = "runs/baselines/"  # GMUICU TR checkpoints (MIMIC uses same TR model)
    for seed in seeds:
        ck_path = repo / f"{prefix}transformer_tcr_s{seed}/best.pt"
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        m = StdTransformer(prior_dim=14, mode="TCR")
        m.load_state_dict(ck.get("model_state_dict", ck)); m.to(dev).eval()
        ps = []
        with torch.inference_mode():
            for b in loader:
                b = move(b, dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    out = m(b)
                ps.append(out["organ_future"][:, H6, :].float().cpu().numpy())
        ens.append(np.concatenate(ps))
        print(f"    TR seed {seed}: done", flush=True)
        del m; torch.cuda.empty_cache()
    return np.mean(ens, axis=0)


def sofa_mae_per_anchor(pred, true, mask):
    return np.abs((pred * mask).sum(axis=1) - (true * mask).sum(axis=1))


def paired_bootstrap(mae_a, mae_b, stays, n_boot, seed):
    """配对 bootstrap: mean(a-b) 的 CI. 正值 = a 更差."""
    idx_map = {s: np.where(stays == s)[0] for s in np.unique(stays)}
    n_st = len(idx_map)
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(list(idx_map.keys()), size=n_st, replace=True)
        idx = np.concatenate([idx_map[s] for s in sampled])
        diffs.append((mae_a[idx] - mae_b[idx]).mean())
    a = 0.025
    return float(np.mean(diffs)), float(np.percentile(diffs, a*100)), float(np.percentile(diffs, (1-a)*100))


def process_cohort(name, loader, plf_pred, tr_pred, organ, omask, outdir):
    o_now = organ[:, 0, :]; m_now = omask[:, 0, :]
    o_6h = organ[:, H6+1, :]; m_6h = omask[:, H6+1, :]
    stays = None
    # 收集 stays
    stays_list = []
    for b in loader:
        stays_list.append(b["stay_id"].numpy())
    stays = np.concatenate(stays_list)

    valid = (m_now * m_6h).sum(axis=1) > 0
    delta_sofa = ((m_now * m_6h) * (o_6h - o_now)).sum(axis=1)

    subsets = {"all": valid, "changed_ge1": valid & (np.abs(delta_sofa) >= 1)}
    results = {}

    for sub_name, mask in subsets.items():
        idx = np.where(mask)[0]
        if len(idx) < 50: continue

        mae_plf = sofa_mae_per_anchor(plf_pred[idx], o_6h[idx], m_6h[idx])
        mae_tr = sofa_mae_per_anchor(tr_pred[idx], o_6h[idx], m_6h[idx])
        mae_per = sofa_mae_per_anchor(o_now[idx], o_6h[idx], m_6h[idx])
        st = stays[idx]

        sub = {"n": len(idx),
               "plf_mean": float(mae_plf.mean()), "tr_mean": float(mae_tr.mean()), "persist_mean": float(mae_per.mean())}

        # 三对 paired CI: PLF-TR, PLF-persist, TR-persist
        for a_name, a_mae, b_name, b_mae in [
            ("plf_minus_tr", mae_plf, "tr", mae_tr),
            ("plf_minus_persist", mae_plf, "persist", mae_per),
            ("tr_minus_persist", mae_tr, "persist", mae_per),
        ]:
            pt, lo, hi = paired_bootstrap(a_mae, b_mae, st, N_BOOT, BOOT_SEED)
            sub[a_name] = {"point": pt, "ci_lo": lo, "ci_hi": hi}
        results[sub_name] = sub
        print(f"\n  {name} {sub_name} (n={len(idx)}):", flush=True)
        print(f"    PLF={sub['plf_mean']:.3f}  TR={sub['tr_mean']:.3f}  persist={sub['persist_mean']:.3f}", flush=True)
        for pair in ["plf_minus_tr", "plf_minus_persist", "tr_minus_persist"]:
            r = sub[pair]
            sig = "不跨0" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else "跨0"
            print(f"    {pair}: {r['point']:+.3f} ({r['ci_lo']:+.3f},{r['ci_hi']:+.3f}) {sig}", flush=True)

    outdir.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(outdir / "frozen_changed_state_ci.json", "w"), indent=2, ensure_ascii=False, default=float)
    return results


def main():
    configure_cuda(); dev = DEVICE
    print("=" * 70, flush=True)
    print("Changed-state paired bootstrap CI (GMUICU + MIMIC)", flush=True)
    print("=" * 70, flush=True)

    spec = load_v4_proxy_contract(REPO / "configs/v4/v4_proxy_contract.json")

    # === GMUICU ===
    print("\n=== GMUICU ===", flush=True)
    from v6.data.v4_dataset import PLFOGTV4Dataset
    ds_g = PLFOGTV4Dataset("test")
    loader_g = DataLoader(ds_g, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    organ_g, omask_g = [], []
    for b in loader_g:
        organ_g.append(b["organ"].numpy()); omask_g.append(b["organ_mask"].numpy())
    organ_g = np.concatenate(organ_g); omask_g = np.concatenate(omask_g)

    print("  PLF...", flush=True)
    plf_g = run_plf(loader_g, dev, spec, REPO, SEEDS)
    print("  TR...", flush=True)
    tr_g = run_tr(loader_g, dev, REPO, SEEDS)
    if not os.path.exists(str(OUTDIR_G / "frozen_changed_state_ci.json")):
        process_cohort("GMUICU", loader_g, plf_g, tr_g, organ_g, omask_g, OUTDIR_G)
    else:
        print("GMUICU CI 已存在, 跳过", flush=True)

    # === MIMIC ===
    print("\n=== MIMIC ===", flush=True)
    from v6.data.mimic_dataset import MIMICDataset
    ds_m = MIMICDataset(split="test")
    loader_m = DataLoader(ds_m, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    organ_m, omask_m = [], []
    for b in loader_m:
        organ_m.append(b["organ"].numpy()); omask_m.append(b["organ_mask"].numpy())
    organ_m = np.concatenate(organ_m); omask_m = np.concatenate(omask_m)

    print("  PLF...", flush=True)
    plf_m = run_plf_mimic(loader_m, dev, spec, REPO, SEEDS)
    print("  TR...", flush=True)
    tr_m = run_tr(loader_m, dev, REPO, SEEDS, mimic=True)
    process_cohort("MIMIC", loader_m, plf_m, tr_m, organ_m, omask_m, OUTDIR_M)

    print(f"\n{'='*70}\n全部完成", flush=True)


if __name__ == "__main__":
    main()
