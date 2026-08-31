#!/usr/bin/env python
"""MIMIC-IV Transformer trajectory — RETRAINED 版 (修改意见4 问题5).

加载 runs_mimic/baselines/transformer_tcr_s{seed} (MIMIC 从头训练),
输出 changed-state 分层 + 多时距 + PLF/persistence 三方 paired cluster bootstrap CI.
推理结果缓存于 mimic_tr_retrained_preds.npz (崩溃可恢复).

输出: results_mimic/frozen_mimic_tr_retrained.json
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
from v6.models.plf_ogt_v4 import PLFOGTV4Model
from v6.models.v4_axes import load_v4_proxy_contract

REPO = Path(r"F:/MIMIC3_1/V12")
OUTDIR = Path(r"F:/MIMIC3_1/V13/results_mimic")
CACHE = OUTDIR / "mimic_tr_retrained_preds.npz"
SEEDS = [42, 52, 62]; H6 = 5
N_BOOT = 2000; BOOT_SEED = 42


def move(b, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v) for k, v in b.items()}


def main():
    configure_cuda(); dev = DEVICE
    print("=" * 70, flush=True)
    print("MIMIC-IV Transformer trajectory (RETRAINED 3-seed TCR)", flush=True)
    print("=" * 70, flush=True)

    if CACHE.exists():
        c = np.load(CACHE, allow_pickle=True)
        plf_pred, tr_pred = c["plf_pred"], c["tr_pred"]
        organ, omask, stays = c["organ"], c["omask"], c["stays"]
        print("缓存命中, 跳过推理", flush=True)
    else:
        ds = MIMICDataset(split="test")
        loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
        print(f"MIMIC test: {len(ds)}\n", flush=True)

        all_organ, all_mask, all_stays = [], [], []
        for b in loader:
            all_organ.append(b["organ"].numpy()); all_mask.append(b["organ_mask"].numpy())
            all_stays.append(b["stay_id"].numpy())
        organ = np.concatenate(all_organ); omask = np.concatenate(all_mask)
        stays = np.concatenate(all_stays)

        spec = load_v4_proxy_contract(REPO / "configs/v4/v4_proxy_contract.json")

        print("=== PLF 3-seed TCR ===", flush=True)
        plf_ens = []
        for seed in SEEDS:
            ck = torch.load(REPO / f"runs_mimic/v4/full_s5_s{seed}/best.pt", map_location="cpu", weights_only=False)
            m = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4, n_residual=6,
                              event_layers=2, concept_layers=1, residual_layers=1,
                              transition_layers=2, dropout=0.0, transition_mode="modulation",
                              n_horizons=12, r_encoder_type="slot_attention", r_n_iters=3, spec=spec)
            m.load_state_dict(ck["model_state_dict"], strict=False)
            m.to(dev).eval()
            ps = []
            with torch.inference_mode():
                for b in loader:
                    b = move(b, dev)
                    with torch.autocast(dev.type, dtype=torch.bfloat16):
                        out = m(b, stage="conditioned", future_treatment_mode="actual")
                    ps.append(out["organ_future"][:, :12, :].float().cpu().numpy())
            plf_ens.append(np.concatenate(ps))
            print(f"  PLF seed {seed}: done", flush=True)
            del m; torch.cuda.empty_cache()
        plf_pred = np.mean(plf_ens, axis=0)

        print("=== TR (retrained) 3-seed TCR ===", flush=True)
        tr_ens = []
        for seed in SEEDS:
            ck = torch.load(REPO / f"runs_mimic/baselines/transformer_tcr_s{seed}/best.pt",
                            map_location="cpu", weights_only=False)
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
            tr_ens.append(np.concatenate(ps))
            print(f"  TR seed {seed}: done", flush=True)
            del model; torch.cuda.empty_cache()
        tr_pred = np.mean(tr_ens, axis=0)

        np.savez_compressed(CACHE, plf_pred=plf_pred, tr_pred=tr_pred,
                            organ=organ, omask=omask, stays=stays)
        print(f"预测缓存: {CACHE}", flush=True)

    o_now = organ[:, 0, :]; m_now = omask[:, 0, :]

    # ── 6h changed-state 三方 ──
    print("\n=== 6h changed-state (PLF / TR-retrained / persistence) ===", flush=True)
    o_6h = organ[:, H6 + 1, :]; m_6h = omask[:, H6 + 1, :]
    valid = (m_now * m_6h).sum(axis=1) > 0
    delta_sofa = ((m_now * m_6h) * (o_6h - o_now)).sum(axis=1)
    persist_pred = np.broadcast_to(o_now[:, None, :], organ.shape[:1] + (12, 6))

    def sofa_err(pred, mask):
        p = pred[mask, H6, :]; t = o_6h[mask]; m = m_6h[mask]
        return np.abs((p * m).sum(axis=1) - (t * m).sum(axis=1))

    subsets = [
        ("all", valid),
        ("unchanged", valid & (delta_sofa == 0)),
        ("changed_ge1", valid & (np.abs(delta_sofa) >= 1)),
        ("worsened_ge2", valid & (delta_sofa >= 2)),
    ]
    results = {}
    err_store = {}
    for label, mask in subsets:
        n = int(mask.sum())
        if n < 10:
            continue
        e_plf = sofa_err(plf_pred, mask)
        e_tr = sofa_err(tr_pred, mask)
        e_pe = sofa_err(persist_pred, mask)
        results[label] = {"n": n,
                          "plf_sofa": float(e_plf.mean()), "tr_sofa": float(e_tr.mean()),
                          "persist_sofa": float(e_pe.mean())}
        err_store[label] = (e_plf, e_tr, e_pe, stays[mask])
        print(f"  {label:<16} n={n:<8} PLF={e_plf.mean():.3f}  TR={e_tr.mean():.3f}  "
              f"persist={e_pe.mean():.3f}", flush=True)

    # ── paired cluster bootstrap CI (6h) ──
    print(f"\n=== Paired cluster bootstrap (n_boot={N_BOOT}) ===", flush=True)
    ci_out = {}
    for label in ["all", "unchanged", "changed_ge1"]:
        if label not in err_store:
            continue
        e_plf, e_tr, e_pe, st = err_store[label]
        uniq = np.unique(st)
        idx_map = {s: np.where(st == s)[0] for s in uniq}
        rng = np.random.RandomState(BOOT_SEED)
        d_pt, d_pp, d_tp = [], [], []
        for _ in range(N_BOOT):
            sampled = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([idx_map[s] for s in sampled])
            d_pt.append(e_plf[idx].mean() - e_tr[idx].mean())
            d_pp.append(e_plf[idx].mean() - e_pe[idx].mean())
            d_tp.append(e_tr[idx].mean() - e_pe[idx].mean())
        ci_out[label] = {
            "plf_minus_tr": {"point": float(e_plf.mean() - e_tr.mean()),
                             "ci": [float(np.percentile(d_pt, 2.5)), float(np.percentile(d_pt, 97.5))]},
            "plf_minus_persist": {"point": float(e_plf.mean() - e_pe.mean()),
                                  "ci": [float(np.percentile(d_pp, 2.5)), float(np.percentile(d_pp, 97.5))]},
            "tr_minus_persist": {"point": float(e_tr.mean() - e_pe.mean()),
                                 "ci": [float(np.percentile(d_tp, 2.5)), float(np.percentile(d_tp, 97.5))]},
        }
        c = ci_out[label]
        print(f"  [{label}] PLF-TR={c['plf_minus_tr']['point']:+.3f} "
              f"({c['plf_minus_tr']['ci'][0]:+.3f},{c['plf_minus_tr']['ci'][1]:+.3f})  "
              f"PLF-persist={c['plf_minus_persist']['point']:+.3f} "
              f"({c['plf_minus_persist']['ci'][0]:+.3f},{c['plf_minus_persist']['ci'][1]:+.3f})  "
              f"TR-persist={c['tr_minus_persist']['point']:+.3f} "
              f"({c['tr_minus_persist']['ci'][0]:+.3f},{c['tr_minus_persist']['ci'][1]:+.3f})", flush=True)

    # ── 多时距 ──
    print("\n=== 多时距 ===", flush=True)
    multi = []
    for h, hi in [(1, 0), (3, 2), (6, 5), (12, 11)]:
        o_h = organ[:, hi + 1, :]; m_h = omask[:, hi + 1, :]
        v = (m_now * m_h).sum(axis=1) > 0
        if v.sum() == 0:
            continue
        plf = float(np.abs((plf_pred[v, hi, :] * m_h[v]).sum(axis=1) - (o_h[v] * m_h[v]).sum(axis=1)).mean())
        tr = float(np.abs((tr_pred[v, hi, :] * m_h[v]).sum(axis=1) - (o_h[v] * m_h[v]).sum(axis=1)).mean())
        per = float(np.abs((o_now[v] * m_h[v]).sum(axis=1) - (o_h[v] * m_h[v]).sum(axis=1)).mean())
        multi.append({"horizon": h, "plf_sofa": plf, "tr_sofa": tr, "persist_sofa": per})
        print(f"  {h}h: PLF={plf:.3f} TR={tr:.3f} persist={per:.3f}", flush=True)

    out = {"changed_state_6h": results, "bootstrap_ci": ci_out, "multi_horizon": multi,
           "note": "TR = MIMIC 从头训练 (runs_mimic/baselines/, 3-seed sel≈0.171), "
                   "替代此前错误的 GMUICU zero-shot. 3-seed ensemble; "
                   "paired stay-cluster bootstrap n_boot=2000."}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUTDIR / "frozen_mimic_tr_retrained.json", "w"), indent=2, ensure_ascii=False, default=float)
    print(f"\n保存: {OUTDIR / 'frozen_mimic_tr_retrained.json'}", flush=True)


if __name__ == "__main__":
    main()
