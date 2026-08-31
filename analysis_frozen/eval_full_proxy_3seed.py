#!/usr/bin/env python
"""完整模型 (有 anchor) 3-seed proxy recovery MAE — 作为 no_anchor 的对照."""
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
from v6.models.v4_axes import load_v4_proxy_contract

REPO = Path(r"F:/MIMIC3_1/V12")
OUTDIR = Path(r"F:/MIMIC3_1/V13/results/v4")
SEEDS = [42, 52, 62]


def move(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v) for k, v in batch.items()}


def main():
    configure_cuda(); dev = DEVICE
    print("="*60, flush=True)
    print("FULL model (with anchor) 3-seed proxy recovery", flush=True)
    print("="*60, flush=True)
    spec = load_v4_proxy_contract(REPO / "configs/v4/v4_proxy_contract.json")
    ds = PLFOGTV4Dataset("test")
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"test: {len(ds)}\n", flush=True)

    per_seed = {}
    for seed in SEEDS:
        ckpt = REPO / f"runs/v4/full_s5_s{seed}/best.pt"
        if not ckpt.exists():
            print(f"  seed {seed}: checkpoint missing {ckpt}", flush=True)
            continue
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        # 完整模型: proxy_bias_init=2.0 (默认), alpha_semantic=1.0
        model = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4, n_residual=6,
                              event_layers=2, concept_layers=1, residual_layers=1,
                              transition_layers=2, dropout=0.0, transition_mode="modulation",
                              n_horizons=12, r_encoder_type="slot_attention", r_n_iters=3, spec=spec)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        model.to(dev).eval()

        all_diff = []
        with torch.inference_mode():
            for batch in loader:
                batch = move(batch, dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    out = model(batch, stage="conditioned", future_treatment_mode="actual")
                cm = out.get("concept_mean")
                tgt = out.get("weak_concept_target")
                mask = out.get("weak_concept_mask")
                if cm is None or tgt is None or mask is None:
                    continue
                cm = cm.float().cpu().numpy()
                tgt = tgt.cpu().numpy()
                mask = mask.cpu().numpy()
                diff = np.abs(cm - tgt) * mask
                valid = mask.sum(axis=0) > 0
                for vi in range(min(cm.shape[-1], 33)):
                    if valid[vi]:
                        all_diff.append(float(diff[:, vi].sum() / mask[:, vi].sum()))

        overall_mae = float(np.mean(all_diff)) if all_diff else float("nan")
        per_seed[f"seed_{seed}"] = {"proxy_mae": overall_mae, "n_proxies": len(all_diff)}
        print(f"  seed {seed}: proxy_mae={overall_mae:.4f} (n_proxies={len(all_diff)})", flush=True)
        del model; torch.cuda.empty_cache()

    maes = [per_seed[f"seed_{s}"]["proxy_mae"] for s in SEEDS if f"seed_{s}" in per_seed]
    summary = {"mean": float(np.mean(maes)), "std": float(np.std(maes)), "per_seed": per_seed}
    print(f"\n3-seed mean={summary['mean']:.4f} ± {summary['std']:.4f}", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(OUTDIR / "full_proxy_3seed.json", "w"), indent=2, ensure_ascii=False, default=float)
    print(f"保存: {OUTDIR / 'full_proxy_3seed.json'}", flush=True)


if __name__ == "__main__":
    main()
