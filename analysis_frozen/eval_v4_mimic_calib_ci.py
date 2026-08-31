#!/usr/bin/env python
"""V4 MIMIC-IV 3-seed 校准 + bootstrap CI —— 第二档 (#12).

与 eval_v4_mimic_interp.py 同口径, 但:
  1. 用 3-seed ensemble (而非单 seed 42)
  2. 对 ECE/Brier/slope/intercept 做患者层面聚类 bootstrap CI

输出: results_mimic/v4_calib_ci.json

用法: python scripts/eval_v4_mimic_calib_ci.py
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

from v6.config import DEVICE, configure_cuda
from v6.data.dataset import collate
from v6.data.mimic_dataset import MIMICDataset
from v6.models.plf_ogt_v4 import PLFOGTV4Model
from v6.models.v4_axes import load_v4_proxy_contract
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[1]
H6 = 5
SEEDS = [42, 52, 62]
N_BOOT = 2000
BOOT_SEED = 42
RESULTS_MIMIC = REPO / "results_mimic"
RESULTS_MIMIC.mkdir(parents=True, exist_ok=True)


def move(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v)
            for k, v in batch.items()}


def get_logodds_ensemble(loaders, dev, spec):
    """3-seed ensemble 的 worsen log-odds. 返回 (logodds, y, stays) per loader."""
    # loaders: list of (name, loader); 对同一组样本逐 seed 推理取均值
    raise NotImplementedError  # 见 main, 结构化处理


def softmax_worsen_logodds(logits):
    """logits (N,3) → worsen vs nonworsen 的 log-odds."""
    worsen = logits[:, 0]
    nonworsen = np.logaddexp.reduce(logits[:, 1:], axis=1)
    return worsen - nonworsen


def main():
    configure_cuda(); dev = DEVICE
    print("=" * 60, flush=True)
    print("第二档 #12: MIMIC-IV 3-seed 校准 + bootstrap CI", flush=True)
    print("=" * 60, flush=True)

    spec = load_v4_proxy_contract(REPO / "configs/v4/v4_proxy_contract.json")

    def build_model_load(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4, n_residual=6,
                              event_layers=2, concept_layers=1, residual_layers=1,
                              transition_layers=2, dropout=0.0, transition_mode="modulation",
                              n_horizons=12, r_encoder_type="slot_attention", r_n_iters=3, spec=spec)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        model.to(dev).eval()
        return model

    # 加载 val 和 test
    val_ds = MIMICDataset(split="validation")
    test_ds = MIMICDataset(split="test")
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"val: {len(val_ds)}  test: {len(test_ds)}", flush=True)

    def collect_per_seed(loader, seed):
        """单 seed 推理, 返回 (logodds (N,), y (N,), stays (N,))."""
        model = build_model_load(REPO / f"runs_mimic/v4/full_s5_s{seed}/best.pt")
        all_lo, all_y, all_st = [], [], []
        with torch.inference_mode():
            for batch in loader:
                batch = move(batch, dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    out = model(batch, stage="conditioned", future_treatment_mode="actual")
                logits = out["class_logits"][:, H6, :].float().cpu().numpy()
                all_lo.append(softmax_worsen_logodds(logits))
                organ = batch["organ"].cpu(); organ_mask = batch["organ_mask"].cpu()
                stay = batch["stay_id"].cpu().numpy() if "stay_id" in batch else np.arange(logits.shape[0])
                for j in range(logits.shape[0]):
                    o_now = organ[j, 0].numpy(); m_now = organ_mask[j, 0].numpy()
                    o_h = organ[j, H6 + 1].numpy(); m_h = organ_mask[j, H6 + 1].numpy()
                    valid = m_now * m_h
                    delta = np.sum(valid * (o_h - o_now))
                    all_y.append(1.0 if delta >= 2 else 0.0)
                all_st.append(stay)
        del model; torch.cuda.empty_cache()
        return np.concatenate(all_lo), np.array(all_y), np.concatenate(all_st)

    # val (用于温度拟合): 3-seed 平均 logodds
    print("\n=== val 3-seed ensemble ===", flush=True)
    val_los = []
    for s in SEEDS:
        lo, y, st = collect_per_seed(val_loader, s)
        val_los.append(lo)
        print(f"  val seed {s}: n={len(y)} prev={y.mean():.4f}", flush=True)
    val_lo = np.mean(val_los, axis=0)
    val_y, val_st = y, st  # 标签/stay 与 seed 无关 (同一批样本)

    # test: 3-seed 平均 logodds
    print("\n=== test 3-seed ensemble ===", flush=True)
    test_los = []
    for s in SEEDS:
        lo, y, st = collect_per_seed(test_loader, s)
        test_los.append(lo)
        print(f"  test seed {s}: n={len(y)} prev={y.mean():.4f}", flush=True)
    test_lo = np.mean(test_los, axis=0)
    test_y, test_st = y, st

    # 温度拟合 (val)
    def nll(T):
        p = np.clip(1 / (1 + np.exp(-val_lo / T)), 1e-8, 1 - 1e-8)
        return -np.mean(val_y * np.log(p) + (1 - val_y) * np.log(1 - p))
    T = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded").x
    print(f"\n温度 T={T:.4f}", flush=True)

    def ece(probs, labels, n_bins=10):
        edges = np.linspace(0, 1, n_bins + 1); s = 0.0; n = len(probs)
        for i in range(n_bins):
            m = (probs >= edges[i]) & (probs <= (edges[i + 1] if i == n_bins - 1 else edges[i + 1]))
            if m.sum() == 0: continue
            s += m.sum() / n * abs(probs[m].mean() - labels[m].mean())
        return s

    def calib_metrics(lo, y, ts):
        probs = 1 / (1 + np.exp(-lo / ts))
        lr = LogisticRegression(C=1e10, solver="lbfgs").fit(lo.reshape(-1, 1), y)
        brier = float(np.mean((probs - y) ** 2))
        return {"ECE": float(ece(probs, y)), "Brier": brier,
                "BSS": 1 - brier / (y.mean() * (1 - y.mean())),
                "slope": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0])}

    # 点估计 (缩放后)
    point = calib_metrics(test_lo, test_y, T)
    print("\n点估计 (after T):", {k: round(v, 4) for k, v in point.items()}, flush=True)

    # cluster bootstrap CI (用缩放后温度)
    idx_map = {s: np.where(test_st == s)[0] for s in np.unique(test_st)}
    n_st = len(idx_map)
    rng = np.random.RandomState(BOOT_SEED)
    boot = {k: [] for k in ["ECE", "Brier", "BSS", "slope", "intercept"]}
    print(f"\nbootstrap CI (n_boot={N_BOOT}, n_clusters={n_st})...", flush=True)
    for b in range(N_BOOT):
        sampled = rng.choice(list(idx_map.keys()), size=n_st, replace=True)
        idx = np.concatenate([idx_map[s] for s in sampled])
        m = calib_metrics(test_lo[idx], test_y[idx], T)
        for k in boot: boot[k].append(m[k])
        if (b + 1) % 500 == 0: print(f"  {b + 1}/{N_BOOT}", flush=True)

    a = 0.025
    ci = {k: {"lo": float(np.percentile(boot[k], a * 100)),
              "hi": float(np.percentile(boot[k], (1 - a) * 100))} for k in boot}

    results = {"T": float(T), "prevalence": float(test_y.mean()),
               "n_test": int(len(test_y)), "n_clusters": n_st, "n_boot": N_BOOT,
               "after": point, "ci": ci}
    out = RESULTS_MIMIC / "v4_calib_ci.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n保存: {out}", flush=True)
    print("\n=== 结果汇总 ===", flush=True)
    for k in ["ECE", "Brier", "slope", "intercept"]:
        print(f"  {k}: {point[k]:.4f} ({ci[k]['lo']:.4f}-{ci[k]['hi']:.4f})", flush=True)


if __name__ == "__main__":
    main()
