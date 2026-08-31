#!/usr/bin/env python
"""V4 S/R 治疗通路 2×2 实验 — 3-seed 验证 (φ_R > φ_S 稳定性).

对每个 seed 单独跑 2×2, 再算 ensemble 的 2×2 + Shapley.
报告 φ_R/φ_S 的跨 seed 均值 ± std.

输出: results/v4/sr_pathway_2x2_3seed.json
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, r"F:/MIMIC3_1/V12")
sys.stdout.reconfigure(encoding="utf-8")

from v6.config import DEVICE, RESULTS_DIR, configure_cuda
from v6.data.dataset import collate
from v6.data.v4_dataset import PLFOGTV4Dataset
from v6.models.plf_ogt_v4 import PLFOGTV4Model
from v6.models.v4_axes import load_v4_proxy_contract
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

REPO = Path(r"F:/MIMIC3_1/V12")
H6 = 5; SEEDS = [42, 52, 62]


def move(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v)
            for k, v in batch.items()}

def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def get_metrics_single_seed(model, loader, dev, block_s, block_r):
    """单个 seed 的 2×2 条件指标."""
    model.eval()
    all_p, all_y = [], []
    organ_maes = {o: [] for o in range(6)}

    with torch.inference_mode():
        for batch in loader:
            batch = move(batch, dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                out = model(batch, stage="conditioned", future_treatment_mode="actual",
                            block_s_treat=block_s, block_r_treat=block_r)
            logits = out["class_logits"][:, H6, :].float().cpu().numpy()
            p = softmax_np(logits)[:, 0]
            delta6 = batch["delta_sofa"][:, H6].cpu().numpy()
            mask6 = batch["class_mask"][:, H6].cpu().numpy()
            valid = mask6 > 0
            y = (delta6 >= 2).astype(float)
            all_p.append(p[valid]); all_y.append(y[valid])

            organ = batch["organ"].cpu().numpy()
            organ_mask = batch["organ_mask"].cpu().numpy()
            pred_o = out["organ_future"][:, H6, :].float().cpu().numpy()
            o_6h = organ[:, H6+1]; mk = organ_mask[:, H6+1]
            for o in range(6):
                m = mk[:, o] > 0
                if m.sum() > 0:
                    organ_maes[o].append(float(np.abs(pred_o[m, o] - o_6h[m, o]).mean()))

    p_arr = np.concatenate(all_p); y_arr = np.concatenate(all_y)
    organ_vals = [float(np.mean(organ_maes[o])) if organ_maes[o] else float("nan") for o in range(6)]
    return {
        "auprc": float(average_precision_score(y_arr, p_arr)),
        "auroc": float(roc_auc_score(y_arr, p_arr)),
        "brier": float(brier_score_loss(y_arr, p_arr)),
        "macro_mae": float(np.mean(organ_vals)),
    }


def calc_shapley(results_4):
    """从 SR/S/R/N 四个条件算 Shapley."""
    U_brier = {k: -results_4[k]["brier"] for k in ["SR", "S", "R", "N"]}
    U_mae = {k: -results_4[k]["macro_mae"] for k in ["SR", "S", "R", "N"]}

    phi_s_b = 0.5 * ((U_brier["S"] - U_brier["N"]) + (U_brier["SR"] - U_brier["R"]))
    phi_r_b = 0.5 * ((U_brier["R"] - U_brier["N"]) + (U_brier["SR"] - U_brier["S"]))
    i_sr_b = U_brier["SR"] - U_brier["S"] - U_brier["R"] + U_brier["N"]

    phi_s_m = 0.5 * ((U_mae["S"] - U_mae["N"]) + (U_mae["SR"] - U_mae["R"]))
    phi_r_m = 0.5 * ((U_mae["R"] - U_mae["N"]) + (U_mae["SR"] - U_mae["S"]))
    i_sr_m = U_mae["SR"] - U_mae["S"] - U_mae["R"] + U_mae["N"]

    return {
        "brier": {"phi_S": phi_s_b, "phi_R": phi_r_b, "I_sr": i_sr_b,
                   "ratio": phi_r_b / max(abs(phi_s_b), 1e-8)},
        "mae": {"phi_S": phi_s_m, "phi_R": phi_r_m, "I_sr": i_sr_m,
                 "ratio": phi_r_m / max(abs(phi_s_m), 1e-8)},
    }


def main():
    configure_cuda(); dev = DEVICE
    print("="*70, flush=True)
    print("V4 S/R 通路 2×2 — 3-seed 验证 (φ_R>φ_S 稳定性)", flush=True)
    print("="*70, flush=True)

    spec = load_v4_proxy_contract(REPO / "configs/v4/v4_proxy_contract.json")
    ds = PLFOGTV4Dataset("test")
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"test: {len(ds)} samples\n", flush=True)

    CONDITIONS = [("SR", False, False), ("S", False, True), ("R", True, False), ("N", True, True)]

    per_seed = {}
    all_seed_logits = {name: [] for name, _, _ in CONDITIONS}

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===", flush=True)
        ck = torch.load(REPO / f"runs/v4/full_s5_s{seed}/best.pt", map_location="cpu", weights_only=False)
        model = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4, n_residual=6,
                              event_layers=2, concept_layers=1, residual_layers=1,
                              transition_layers=2, dropout=0.0, transition_mode="modulation",
                              n_horizons=12, r_encoder_type="slot_attention", r_n_iters=3, spec=spec)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        model.to(dev).eval()

        seed_results = {}
        for name, bs, br in CONDITIONS:
            r = get_metrics_single_seed(model, loader, dev, bs, br)
            seed_results[name] = r
            print(f"  {name}: AUPRC={r['auprc']:.4f} Brier={r['brier']:.4f} MAE={r['macro_mae']:.4f}", flush=True)

        shapley = calc_shapley(seed_results)
        print(f"  φ_R/φ_S (Brier) = {shapley['brier']['ratio']:.2f}x", flush=True)
        print(f"  φ_R/φ_S (MAE)   = {shapley['mae']['ratio']:.2f}x", flush=True)

        per_seed[f"seed_{seed}"] = {"conditions": seed_results, "shapley": shapley}
        del model; torch.cuda.empty_cache()

    # 跨 seed 汇总
    ratios_brier = [per_seed[f"seed_{s}"]["shapley"]["brier"]["ratio"] for s in SEEDS]
    ratios_mae = [per_seed[f"seed_{s}"]["shapley"]["mae"]["ratio"] for s in SEEDS]
    phi_r_brier = [per_seed[f"seed_{s}"]["shapley"]["brier"]["phi_R"] for s in SEEDS]
    phi_s_brier = [per_seed[f"seed_{s}"]["shapley"]["brier"]["phi_S"] for s in SEEDS]
    phi_r_mae = [per_seed[f"seed_{s}"]["shapley"]["mae"]["phi_R"] for s in SEEDS]
    phi_s_mae = [per_seed[f"seed_{s}"]["shapley"]["mae"]["phi_S"] for s in SEEDS]

    print(f"\n{'='*60}", flush=True)
    print(f"{'seed':>6} | {'φ_S(Brier)':>10} {'φ_R(Brier)':>10} {'ratio':>7} | "
          f"{'φ_S(MAE)':>10} {'φ_R(MAE)':>10} {'ratio':>7}", flush=True)
    print("-"*60, flush=True)
    for i, s in enumerate(SEEDS):
        print(f"{s:>6} | {phi_s_brier[i]:>10.5f} {phi_r_brier[i]:>10.5f} "
              f"{ratios_brier[i]:>7.2f}x | {phi_s_mae[i]:>10.5f} {phi_r_mae[i]:>10.5f} "
              f"{ratios_mae[i]:>7.2f}x", flush=True)
    print("-"*60, flush=True)
    print(f"{'mean':>6} | {np.mean(phi_s_brier):>10.5f} {np.mean(phi_r_brier):>10.5f} "
          f"{np.mean(ratios_brier):>7.2f}x | {np.mean(phi_s_mae):>10.5f} {np.mean(phi_r_mae):>10.5f} "
          f"{np.mean(ratios_mae):>7.2f}x", flush=True)
    print(f"{'std':>6} | {np.std(phi_s_brier):>10.5f} {np.std(phi_r_brier):>10.5f} "
          f"{np.std(ratios_brier):>7.2f}x | {np.std(phi_s_mae):>10.5f} {np.std(phi_r_mae):>10.5f} "
          f"{np.std(ratios_mae):>7.2f}x", flush=True)
    print("="*60, flush=True)

    # φ_R > φ_S 的跨 seed 一致性
    r_gt_s_brier = sum(1 for s in SEEDS if phi_r_brier[SEEDS.index(s)] > phi_s_brier[SEEDS.index(s)])
    r_gt_s_mae = sum(1 for s in SEEDS if phi_r_mae[SEEDS.index(s)] > phi_s_mae[SEEDS.index(s)])
    print(f"\nφ_R > φ_S 一致性: Brier {r_gt_s_brier}/3 seeds, MAE {r_gt_s_mae}/3 seeds", flush=True)

    # 保存
    out_data = {
        "per_seed": per_seed,
        "summary": {
            "brier": {
                "phi_S_mean": float(np.mean(phi_s_brier)), "phi_S_std": float(np.std(phi_s_brier)),
                "phi_R_mean": float(np.mean(phi_r_brier)), "phi_R_std": float(np.std(phi_r_brier)),
                "ratio_mean": float(np.mean(ratios_brier)), "ratio_std": float(np.std(ratios_brier)),
                "r_gt_s_seeds": r_gt_s_brier,
            },
            "mae": {
                "phi_S_mean": float(np.mean(phi_s_mae)), "phi_S_std": float(np.std(phi_s_mae)),
                "phi_R_mean": float(np.mean(phi_r_mae)), "phi_R_std": float(np.std(phi_r_mae)),
                "ratio_mean": float(np.mean(ratios_mae)), "ratio_std": float(np.std(ratios_mae)),
                "r_gt_s_seeds": r_gt_s_mae,
            },
        },
    }

    out = Path(r"F:/MIMIC3_1/V13/results/v4/sr_pathway_2x2_3seed.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"\n保存: {out}", flush=True)

if __name__ == "__main__":
    main()
