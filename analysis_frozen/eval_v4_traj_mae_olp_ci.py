#!/usr/bin/env python
"""V4 OLP 连续器官轨迹 MAE + OLP→TCR 差值 CI —— 第二档 (#5 补全).

补全表2 的 OLP 行 CI, 并算 OLP→TCR 的轨迹 MAE 差值 CI (让 #5 局限段"其一"可放宽).
- OLP 3-seed 推理 organ_future (mode=zero), 缓存.
- 复刻 eval_v4_organ_mae.py 的 compute_mae 全局加权口径.
- bootstrap 与 traj_mae_ci.json (TCR) 共用同一患者重采样序列 (BOOT_SEED), 保证配对.

输出: results/v4/traj_mae_olp_ci.json  (OLP MAE CI + OLP→TCR 差值 CI)
用法: python scripts/eval_v4_traj_mae_olp_ci.py
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # 1=RTX 4090 D
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from v6.config import DEVICE, RESULTS_DIR, configure_cuda
from v6.data.dataset import collate
from v6.data.v4_dataset import PLFOGTV4Dataset
from v6.models.plf_ogt_v4 import PLFOGTV4Model
from v6.models.v4_axes import load_v4_proxy_contract

REPO = Path(__file__).resolve().parents[1]
HORIZONS = [1, 3, 6, 12]
SEEDS = [42, 52, 62]
N_BOOT = 2000
BOOT_SEED = 42
OUT_DIR = RESULTS_DIR / "v4"


def move(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v)
            for k, v in batch.items()}


def run_olp_inference(loader, dev, spec):
    """OLP (zero) 3-seed ensemble organ_future + 标签."""
    ens = []
    organ_lab = organ_mask = stays = None
    for seed in SEEDS:
        ckpt = REPO / f"runs/v4/full_s5_s{seed}/best.pt"
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4, n_residual=6,
                              event_layers=2, concept_layers=1, residual_layers=1,
                              transition_layers=2, dropout=0.0, transition_mode="modulation",
                              n_horizons=12, r_encoder_type="slot_attention", r_n_iters=3, spec=spec)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        model.to(dev).eval()
        sl = []; ol = []; ml = []; stl = []
        with torch.inference_mode():
            for batch in loader:
                batch = move(batch, dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    out = model(batch, stage="conditioned", future_treatment_mode="zero")
                sl.append(out["organ_future"].float().cpu().numpy())
                ol.append(batch["organ"].cpu().numpy()); ml.append(batch["organ_mask"].cpu().numpy())
                stl.append(batch["stay_id"].cpu().numpy())
        ens.append(np.concatenate(sl))
        if organ_lab is None:
            organ_lab = np.concatenate(ol); organ_mask = np.concatenate(ml); stays = np.concatenate(stl)
        print(f"    OLP seed {seed}: done", flush=True)
        del model; torch.cuda.empty_cache()
    return np.mean(ens, axis=0), organ_lab, organ_mask, stays


def compute_mae(pred, target, mask):
    diff = np.abs(pred - target) * mask
    return diff.sum() / max(mask.sum(), 1)


def metrics_at_h(pred, target, mask_all, hi):
    omh = mask_all[:, hi, :]
    pred_sum = (pred[:, hi, :] * omh).sum(axis=-1)
    target_sum = (target[:, hi, :] * omh).sum(axis=-1)
    n_valid = omh.sum(axis=-1)
    mask_total = (n_valid > 0).astype(float)
    sofa_mae = compute_mae(pred_sum, target_sum, mask_total)
    organ_maes = [compute_mae(pred[:, hi, o], target[:, hi, o], mask_all[:, hi, o]) for o in range(6)]
    return float(sofa_mae), float(np.mean(organ_maes))


def main():
    configure_cuda(); dev = DEVICE
    print("=" * 60, flush=True)
    print("第二档 #5(补): OLP 轨迹 MAE CI + OLP→TCR 差值 CI", flush=True)
    print("=" * 60, flush=True)

    spec = load_v4_proxy_contract(REPO / "configs/v4/v4_proxy_contract.json")
    ds = PLFOGTV4Dataset("test")
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"test: {len(ds)} samples", flush=True)

    # OLP 推理 (缓存)
    cache = OUT_DIR / "traj_mae_olp_cache.npz"
    if cache.exists():
        print(f"\n加载 OLP 缓存: {cache}", flush=True)
        c = np.load(cache)
        pred_olp, organ_lab, organ_mask, stays = c["pred"], c["organ_lab"], c["organ_mask"], c["stays"]
    else:
        print("\n=== OLP (zero) 3-seed 推理 ===", flush=True)
        pred_olp, organ_lab, organ_mask, stays = run_olp_inference(loader, dev, spec)
        np.savez(cache, pred=pred_olp, organ_lab=organ_lab, organ_mask=organ_mask, stays=stays)
        print(f"缓存已存: {cache}", flush=True)

    # TCR pred (复用 traj_mae_ci_cache)
    tcr_cache = OUT_DIR / "traj_mae_ci_cache.npz"
    pred_tcr = np.load(tcr_cache)["pred"]

    target = organ_lab[:, 1:13, :]; mask_all = organ_mask[:, 1:13, :]

    # 点估计核对 (与 organ_trajectory_mae.json OLP 比)
    orig = json.load(open(REPO / "results/v4/organ_trajectory_mae.json"))["OLP"]
    print("\nOLP 点估计核对:", flush=True)
    for h in HORIZONS:
        s, mm = metrics_at_h(pred_olp, target, mask_all, h - 1)
        print(f"  {h:>2}h: 总分 {s:.4f}(原{orig[f'{h}h']['sofa_total_mae']:.4f})  macro {mm:.4f}(原{orig[f'{h}h']['macro_mae']:.4f})", flush=True)

    # cluster bootstrap (与 TCR 脚本同 BOOT_SEED/同重采样, 保证配对)
    idx_map = {s: np.where(stays == s)[0] for s in np.unique(stays)}
    n_st = len(idx_map)
    rng = np.random.RandomState(BOOT_SEED)
    boot_idxs = [np.concatenate([idx_map[s] for s in rng.choice(list(idx_map.keys()), size=n_st, replace=True)])
                 for _ in range(N_BOOT)]
    print(f"\nbootstrap (n_boot={N_BOOT}, n_clusters={n_st})...", flush=True)

    results = {}
    for h in HORIZONS:
        hi = h - 1
        sofa_o_b, mac_o_b, sofa_d_b, mac_d_b = [], [], [], []
        for idx in boot_idxs:
            so, mo = metrics_at_h(pred_olp[idx], target[idx], mask_all[idx], hi)
            st_, mt_ = metrics_at_h(pred_tcr[idx], target[idx], mask_all[idx], hi)
            sofa_o_b.append(so); mac_o_b.append(mo)
            sofa_d_b.append(st_ - so); mac_d_b.append(mt_ - mo)
        a = 0.025
        so_pe, mo_pe = metrics_at_h(pred_olp, target, mask_all, hi)
        st_pe, mt_pe = metrics_at_h(pred_tcr, target, mask_all, hi)
        results[f"{h}h"] = {
            "olp_sofa_mae": so_pe, "olp_sofa_ci": [float(np.percentile(sofa_o_b, a * 100)), float(np.percentile(sofa_o_b, (1 - a) * 100))],
            "olp_macro_mae": mo_pe, "olp_macro_ci": [float(np.percentile(mac_o_b, a * 100)), float(np.percentile(mac_o_b, (1 - a) * 100))],
            "delta_sofa": float(st_pe - so_pe), "delta_sofa_ci": [float(np.percentile(sofa_d_b, a * 100)), float(np.percentile(sofa_d_b, (1 - a) * 100))],
            "delta_macro": float(mt_pe - mo_pe), "delta_macro_ci": [float(np.percentile(mac_d_b, a * 100)), float(np.percentile(mac_d_b, (1 - a) * 100))],
        }
        r = results[f"{h}h"]
        print(f"  {h:>2}h: OLP总分 {so_pe:.4f}({r['olp_sofa_ci'][0]:.4f}-{r['olp_sofa_ci'][1]:.4f})  "
              f"Δ总分(TCR-OLP) {r['delta_sofa']:+.4f}({r['delta_sofa_ci'][0]:+.4f},{r['delta_sofa_ci'][1]:+.4f})", flush=True)

    out = OUT_DIR / "traj_mae_olp_ci.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n保存: {out}", flush=True)


if __name__ == "__main__":
    main()
