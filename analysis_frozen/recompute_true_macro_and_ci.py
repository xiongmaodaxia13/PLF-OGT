#!/usr/bin/env python
"""Recompute TRUE macro-MAE and discrimination CIs from the frozen source of truth.

TRUE macro = mean over 6 organs of per-organ pooled MAE (global num/den per organ).
Delta = TCR - care-off. CIs: ICU-stay cluster bootstrap, n_boot=2000, seed 42, percentile.
Output: results/v4/recompute_true_macro_and_ci.json
"""
import os
import sys
import json
from pathlib import Path

import faulthandler
faulthandler.enable()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Pre-load pyarrow BEFORE torch/CUDA: importing pyarrow.dataset after torch
# native libs are loaded can crash with an access violation on this machine.
import pyarrow  # noqa: E402,F401
import pyarrow.dataset  # noqa: E402,F401
import pyarrow.parquet  # noqa: E402,F401

sys.path.insert(0, r"F:/MIMIC3_1/V12")

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

sys.stdout.reconfigure(encoding="utf-8")

from v6.config import DEVICE, configure_cuda
from v6.data.dataset import collate
from v6.data.v4_dataset import PLFOGTV4Dataset
from v6.models.plf_ogt_v4 import PLFOGTV4Model
from v6.models.v4_axes import load_v4_proxy_contract

REPO = Path(r"F:/MIMIC3_1/V12")
OUTPATH = Path(r"F:/MIMIC3_1/V13/results/v4/recompute_true_macro_and_ci.json")
SEEDS = [42, 52, 62]
HORIZONS = [(1, 0), (3, 2), (6, 5), (12, 11)]
N_BOOT = 2000
BOOT_SEED = 42


def move(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v) for k, v in batch.items()}


def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def build_plf(spec, seed, dev):
    ck = torch.load(REPO / f"runs/v4/full_s5_s{seed}/best.pt", map_location="cpu", weights_only=False)
    m = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4, n_residual=6,
                      event_layers=2, concept_layers=1, residual_layers=1,
                      transition_layers=2, dropout=0.0, transition_mode="modulation",
                      n_horizons=12, r_encoder_type="slot_attention", r_n_iters=3, spec=spec)
    m.load_state_dict(ck["model_state_dict"], strict=False)
    m.to(dev).eval()
    return m


def run_plf_mode(loader, dev, spec, mode):
    ens_logits = []
    ens_organ = []
    for seed in SEEDS:
        print(f"  build seed {seed} ({mode})", flush=True)
        m = build_plf(spec, seed, dev)
        ls = []
        os_ = []
        with torch.inference_mode():
            for bi, b in enumerate(loader):
                b = move(b, dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    out = m(b, stage="conditioned", future_treatment_mode=mode)
                ls.append(out["class_logits"].float().cpu().numpy())
                os_.append(out["organ_future"][:, :12, :].float().cpu().numpy())
        ens_logits.append(np.concatenate(ls))
        ens_organ.append(np.concatenate(os_))
        print(f"  seed {seed} done", flush=True)
        del m
        torch.cuda.empty_cache()
    return np.mean(ens_logits, axis=0), np.mean(ens_organ, axis=0)


def pooled_organ_mae(pred, true, mask):
    err = np.abs(pred - true) * mask
    num = err.sum(axis=0)
    den = mask.sum(axis=0)
    return np.where(den > 0, num / np.maximum(den, 1.0), np.nan)


def sofa_mae_per_anchor(pred, true, mask):
    """Per-anchor absolute error of summed SOFA over valid organs."""
    return np.abs((pred * mask).sum(axis=1) - (true * mask).sum(axis=1))


def percentile_ci(a):
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))


def main():
    configure_cuda()
    dev = DEVICE
    print("device:", dev, flush=True)
    spec = load_v4_proxy_contract(REPO / "configs/v4/v4_proxy_contract.json")
    ds = PLFOGTV4Dataset("test")
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    print("loader ready", flush=True)

    all_organ, all_mask, all_delta, all_cmask, all_stays = [], [], [], [], []
    for bi, b in enumerate(loader):
        all_organ.append(b["organ"].numpy())
        all_mask.append(b["organ_mask"].numpy())
        all_delta.append(b["delta_sofa"].numpy())
        all_cmask.append(b["class_mask"].numpy())
        all_stays.append(b["stay_id"].numpy())
    organ = np.concatenate(all_organ)
    omask = np.concatenate(all_mask)
    delta = np.concatenate(all_delta)
    cmask = np.concatenate(all_cmask)
    stays = np.concatenate(all_stays)
    o_now = organ[:, 0, :]
    m_now = omask[:, 0, :]
    print("labels ready", organ.shape, flush=True)

    print("=== inference TCR ===", flush=True)
    plf_tcr_logits, plf_tcr_organ = run_plf_mode(loader, dev, spec, "actual")
    print("=== inference care-off ===", flush=True)
    plf_co_logits, plf_co_organ = run_plf_mode(loader, dev, spec, "zero")
    print("inference done", plf_tcr_logits.shape, plf_tcr_organ.shape, flush=True)

    macro_rows = []
    for h, hi in HORIZONS:
        o_h = organ[:, hi + 1, :]
        m_h = omask[:, hi + 1, :]
        v_h = (m_now * m_h).sum(axis=1) > 0
        n_v = int(v_h.sum())
        pred_t = plf_tcr_organ[:, hi, :][v_h]
        pred_c = plf_co_organ[:, hi, :][v_h]
        true_h = o_h[v_h]
        mask_h = m_h[v_h]
        sv_traj = stays[v_h]
        idx_traj = {s: np.where(sv_traj == s)[0] for s in np.unique(sv_traj)}
        n_st = len(idx_traj)
        om_t = pooled_organ_mae(pred_t, true_h, mask_h)
        om_c = pooled_organ_mae(pred_c, true_h, mask_h)
        macro_t = float(np.nanmean(om_t))
        macro_c = float(np.nanmean(om_c))
        sofa_t = sofa_mae_per_anchor(pred_t, true_h, mask_h)
        sofa_c = sofa_mae_per_anchor(pred_c, true_h, mask_h)
        sofa_t_pe = float(sofa_t.mean())
        sofa_c_pe = float(sofa_c.mean())
        rng = np.random.RandomState(BOOT_SEED)
        b_t, b_c, b_d = [], [], []
        bs_t, bs_c, bs_d = [], [], []
        for _ in range(N_BOOT):
            s_sel = rng.choice(list(idx_traj.keys()), size=n_st, replace=True)
            ix = np.concatenate([idx_traj[s] for s in s_sel])
            ot = pooled_organ_mae(pred_t[ix], true_h[ix], mask_h[ix])
            oc = pooled_organ_mae(pred_c[ix], true_h[ix], mask_h[ix])
            mt = float(np.nanmean(ot))
            mc = float(np.nanmean(oc))
            b_t.append(mt)
            b_c.append(mc)
            b_d.append(mt - mc)
            st = float(sofa_t[ix].mean())
            sc = float(sofa_c[ix].mean())
            bs_t.append(st)
            bs_c.append(sc)
            bs_d.append(st - sc)
        macro_rows.append({
            "horizon": h,
            "valid_n": n_v,
            "n_clusters": n_st,
            "organ_tcr": [float(x) for x in om_t],
            "organ_co": [float(x) for x in om_c],
            "sofa_tcr": sofa_t_pe,
            "sofa_tcr_ci": percentile_ci(bs_t),
            "sofa_co": sofa_c_pe,
            "sofa_co_ci": percentile_ci(bs_c),
            "delta_sofa": sofa_t_pe - sofa_c_pe,
            "delta_sofa_ci": percentile_ci(bs_d),
            "macro_tcr": macro_t,
            "macro_tcr_ci": percentile_ci(b_t),
            "macro_co": macro_c,
            "macro_co_ci": percentile_ci(b_c),
            "delta_macro": macro_t - macro_c,
            "delta_macro_ci": percentile_ci(b_d),
        })
        print(f"{h}h: n={n_v} cl={n_st} TRUE macro TCR={macro_t:.4f} co={macro_c:.4f} "
              f"d={macro_t - macro_c:+.4f} (CI {percentile_ci(b_d)[0]:+.4f}~{percentile_ci(b_d)[1]:+.4f})", flush=True)
        print(f"    sofa: TCR={sofa_t_pe:.4f} ({percentile_ci(bs_t)[0]:.4f}~{percentile_ci(bs_t)[1]:.4f}) "
              f"co={sofa_c_pe:.4f} ({percentile_ci(bs_c)[0]:.4f}~{percentile_ci(bs_c)[1]:.4f}) "
              f"d={sofa_t_pe - sofa_c_pe:+.4f} ({percentile_ci(bs_d)[0]:+.4f}~{percentile_ci(bs_d)[1]:+.4f})", flush=True)
        print(f"    organs TCR={[f'{x:.3f}' for x in om_t]} co={[f'{x:.3f}' for x in om_c]}", flush=True)

    disc_rows = []
    for h, hi in HORIZONS:
        mc = cmask[:, hi] > 0
        y = (delta[:, hi] >= 2).astype(float)
        p_t = softmax_np(plf_tcr_logits[:, hi, :])[:, 0]
        p_c = softmax_np(plf_co_logits[:, hi, :])[:, 0]
        yv, ptv, pcv = y[mc], p_t[mc], p_c[mc]
        sv = stays[mc]
        idx_map = {s: np.where(sv == s)[0] for s in np.unique(sv)}
        n_st = len(idx_map)
        a_t = float(roc_auc_score(yv, ptv))
        ap_t = float(average_precision_score(yv, ptv))
        a_c = float(roc_auc_score(yv, pcv))
        ap_c = float(average_precision_score(yv, pcv))
        rng = np.random.RandomState(BOOT_SEED)
        ba_t, bap_t, ba_c, bap_c, bd_a, bd_ap = [], [], [], [], [], []
        for _ in range(N_BOOT):
            s_sel = rng.choice(list(idx_map.keys()), size=n_st, replace=True)
            ix = np.concatenate([idx_map[s] for s in s_sel])
            yy = yv[ix]
            if len(set(yy)) < 2:
                continue
            try:
                ra = roc_auc_score(yy, ptv[ix])
                rac = roc_auc_score(yy, pcv[ix])
                rp = average_precision_score(yy, ptv[ix])
                rpc = average_precision_score(yy, pcv[ix])
                ba_t.append(ra)
                ba_c.append(rac)
                bap_t.append(rp)
                bap_c.append(rpc)
                bd_a.append(ra - rac)
                bd_ap.append(rp - rpc)
            except Exception:
                continue
        disc_rows.append({
            "horizon": h,
            "n_valid": int(mc.sum()),
            "n_clusters": n_st,
            "prev": float(yv.mean()),
            "tcr_auroc": a_t,
            "tcr_auroc_ci": percentile_ci(ba_t),
            "tcr_auprc": ap_t,
            "tcr_auprc_ci": percentile_ci(bap_t),
            "co_auroc": a_c,
            "co_auroc_ci": percentile_ci(ba_c),
            "co_auprc": ap_c,
            "co_auprc_ci": percentile_ci(bap_c),
            "delta_auroc": a_t - a_c,
            "delta_auroc_ci": percentile_ci(bd_a),
            "delta_auprc": ap_t - ap_c,
            "delta_auprc_ci": percentile_ci(bd_ap),
        })
        print(f"{h}h: n={int(mc.sum())} cl={n_st} prev={yv.mean() * 100:.2f}%", flush=True)
        print(f"    TCR AUROC {a_t:.4f} ({percentile_ci(ba_t)[0]:.4f}~{percentile_ci(ba_t)[1]:.4f}) "
              f"AUPRC {ap_t:.4f} ({percentile_ci(bap_t)[0]:.4f}~{percentile_ci(bap_t)[1]:.4f})", flush=True)
        print(f"    co  AUROC {a_c:.4f} ({percentile_ci(ba_c)[0]:.4f}~{percentile_ci(ba_c)[1]:.4f}) "
              f"AUPRC {ap_c:.4f} ({percentile_ci(bap_c)[0]:.4f}~{percentile_ci(bap_c)[1]:.4f})", flush=True)
        print(f"    delta AUROC {a_t - a_c:+.4f} ({percentile_ci(bd_a)[0]:+.4f}~{percentile_ci(bd_a)[1]:+.4f}) "
              f"AUPRC {ap_t - ap_c:+.4f} ({percentile_ci(bd_ap)[0]:+.4f}~{percentile_ci(bd_ap)[1]:+.4f})", flush=True)

    out = {
        "note": "TRUE macro = mean of 6 per-organ pooled MAE; delta = TCR - care-off; "
                "cluster bootstrap n_boot=2000 seed=42 percentile 2.5/97.5",
        "macro": macro_rows,
        "discrimination": disc_rows,
    }
    OUTPATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SAVED", OUTPATH, flush=True)


if __name__ == "__main__":
    main()
