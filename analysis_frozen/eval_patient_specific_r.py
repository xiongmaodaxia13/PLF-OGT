#!/usr/bin/env python
"""Patient-specific R negative controls (frozen, 3-seed).

四种条件评价 R 的患者特异性:
  - matched: 正确患者的 R (基线)
  - shuffled: 随机换别的患者的 R (跨患者置换)
  - population_mean: 全局平均 R 替换
  - query_only: 只保留 query context, 清零 R content (identity-like)

方法: encode_branches 拿到正确 R, 然后在外部替换 branch.residual, 再 rollout.
3-seed ensemble, 全测试集, 6h AUPRC + organ MAE.

输出: results/v4/frozen_patient_specific_r.json
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, r"F:/MIMIC3_1/V12")
sys.stdout.reconfigure(encoding="utf-8")

from v6.config import DEVICE, configure_cuda
from v6.data.dataset import collate
from v6.data.v4_dataset import PLFOGTV4Dataset
from v6.models.plf_ogt_v4 import PLFOGTV4Model, V4BranchState
from v6.models.v4_axes import load_v4_proxy_contract
import copy

REPO = Path(r"F:/MIMIC3_1/V12")
OUTDIR = Path(r"F:/MIMIC3_1/V13/results/v4")
SEEDS = [42, 52, 62]; H6 = 5


def move(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v) for k, v in batch.items()}


def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def build_model(spec, seed, dev):
    ck = torch.load(REPO / f"runs/v4/full_s5_s{seed}/best.pt", map_location="cpu", weights_only=False)
    m = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4, n_residual=6,
                      event_layers=2, concept_layers=1, residual_layers=1,
                      transition_layers=2, dropout=0.0, transition_mode="modulation",
                      n_horizons=12, r_encoder_type="slot_attention", r_n_iters=3, spec=spec)
    m.load_state_dict(ck["model_state_dict"], strict=False)
    m.to(dev).eval()
    return m


def run_condition(model, loader, dev, condition, rng_seed=42):
    """跑一个条件. condition: 'matched'/'shuffled'/'mean'/'query_only'.
    返回 worsen prob (N,), organ_future (N,6), labels.
    """
    model.eval()
    all_p, all_y = [], []
    organ_preds = []
    organ_labels = []
    organ_masks = []
    rng = torch.Generator(device="cpu").manual_seed(rng_seed)  # CPU generator for perm indices

    with torch.inference_mode():
        for batch in loader:
            batch = move(batch, dev)
            # 1. 编码正确 branch
            branch = model.encode_branches(batch)

            if condition == "matched":
                pass  # 用原始 branch
            elif condition == "shuffled":
                # 跨患者置换 R: 对 batch 内随机重排 residual
                B = branch.residual.shape[0]
                perm = torch.randperm(B, generator=rng).to(branch.residual.device)
                branch.residual = branch.residual[perm]
            elif condition == "mean":
                # 全局平均 R: 用 batch 内的均值替换
                branch.residual = branch.residual.mean(dim=0, keepdim=True).expand_as(branch.residual)
            elif condition == "query_only":
                # 清零 R content 但保留 slot 结构 (用零向量)
                branch.residual = torch.zeros_like(branch.residual)

            # 2. rollout (用替换后的 branch)
            out = model.rollout_from_state(
                batch, branch, stage="conditioned", future_treatment_mode="actual")

            logits = out["class_logits"][:, H6, :].float().cpu().numpy()
            all_p.append(softmax_np(logits)[:, 0])
            organ_preds.append(out["organ_future"][:, H6, :].float().cpu().numpy())

            organ = batch["organ"].cpu().numpy()
            omask = batch["organ_mask"].cpu().numpy()
            organ_labels.append(organ[:, H6 + 1])
            organ_masks.append(omask[:, H6 + 1])
            o_now = organ[:, 0, :]; m_now = omask[:, 0, :]
            o_6h = organ[:, H6 + 1, :]; m_6h = omask[:, H6 + 1, :]
            valid = (m_now * m_6h).sum(axis=1) > 0
            delta = ((m_now * m_6h) * (o_6h - o_now)).sum(axis=1)
            all_y.append((delta >= 2).astype(float))

    p = np.concatenate(all_p); y = np.concatenate(all_y)
    op = np.concatenate(organ_preds); ol = np.concatenate(organ_labels); om = np.concatenate(organ_masks)
    return p, y, op, ol, om


def main():
    configure_cuda(); dev = DEVICE
    print("=" * 70, flush=True)
    print("Patient-specific R negative controls (frozen 3-seed)", flush=True)
    print("=" * 70, flush=True)
    spec = load_v4_proxy_contract(REPO / "configs/v4/v4_proxy_contract.json")
    ds = PLFOGTV4Dataset("test")
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"test: {len(ds)}\n", flush=True)

    CONDITIONS = ["matched", "shuffled", "mean", "query_only"]
    # 对每个 seed 跑所有条件, 然后 ensemble
    all_results = {cond: {"auprc_list": [], "auroc_list": [], "macro_mae_list": []} for cond in CONDITIONS}

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===", flush=True)
        model = build_model(spec, seed, dev)
        for cond in CONDITIONS:
            p, y, op, ol, om = run_condition(model, loader, dev, cond, rng_seed=seed)
            mask = np.ones(len(y), dtype=bool)  # 全样本
            if y[mask].sum() > 5 and len(set(y[mask])) > 1:
                auc = float(roc_auc_score(y[mask], p[mask]))
                ap = float(average_precision_score(y[mask], p[mask]))
            else:
                auc = ap = float("nan")
            # macro MAE
            organ_maes = []
            for o in range(6):
                m = om[:, o] > 0
                if m.sum() > 0:
                    organ_maes.append(float(np.abs(op[m, o] - ol[m, o]).mean()))
            macro = float(np.mean(organ_maes)) if organ_maes else float("nan")
            all_results[cond]["auprc_list"].append(ap)
            all_results[cond]["auroc_list"].append(auc)
            all_results[cond]["macro_mae_list"].append(macro)
            print(f"  {cond:<14} AUPRC={ap:.4f} AUROC={auc:.4f} macroMAE={macro:.4f}", flush=True)
        del model; torch.cuda.empty_cache()

    # 汇总 (3-seed 均值)
    print(f"\n{'='*70}", flush=True)
    print(f"{'条件':<14}{'AUPRC':<12}{'AUROC':<12}{'macroMAE':<12}{'ΔAUPRC':<12}{'ΔMAE':<10}")
    print("-" * 70)
    final = {}
    matched_auprc = np.mean(all_results["matched"]["auprc_list"])
    matched_mae = np.mean(all_results["matched"]["macro_mae_list"])
    for cond in CONDITIONS:
        ap = float(np.mean(all_results[cond]["auprc_list"]))
        auc = float(np.mean(all_results[cond]["auroc_list"]))
        mae = float(np.mean(all_results[cond]["macro_mae_list"]))
        d_ap = ap - matched_auprc
        d_mae = mae - matched_mae
        std_ap = float(np.std(all_results[cond]["auprc_list"]))
        final[cond] = {"auprc": ap, "auprc_std": std_ap, "auroc": auc,
                       "macro_mae": mae, "delta_auprc": d_ap, "delta_mae": d_mae}
        print(f"{cond:<14}{ap:<12.4f}{auc:<12.4f}{mae:<12.4f}{d_ap:<+12.4f}{d_mae:<+10.4f}", flush=True)
    print("=" * 70, flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    json.dump(final, open(OUTDIR / "frozen_patient_specific_r.json", "w"), indent=2, ensure_ascii=False, default=float)
    print(f"\n保存: {OUTDIR / 'frozen_patient_specific_r.json'}", flush=True)


if __name__ == "__main__":
    main()
