#!/usr/bin/env python
"""TCR trajectory baselines: persistence and standard Transformer.

All three estimators are evaluated on the same anchor--future organ mask:

* Persistence: y_hat(tau+h, organ) = observed SOFA(tau, organ)
* Transformer-TCR: 3-seed ensemble with actual post-anchor treatment actions
* PLF-OGT-TCR: established 3-seed ensemble trajectory cache

The shared mask requires the organ score to be valid at both the anchor and
the evaluated horizon.  Confidence intervals and model differences use a
paired ICU-stay cluster bootstrap and recompute global MAE numerators and
denominators in every replicate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

V13_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = V13_ROOT.parent
CODE_ROOT = WORKSPACE / "V12"
sys.path.insert(0, str(CODE_ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from v6.config import DEVICE, configure_cuda  # noqa: E402
from v6.data.dataset import collate  # noqa: E402
from v6.data.v4_dataset import PLFOGTV4Dataset  # noqa: E402
from v6.models.std_transformer import StdTransformer  # noqa: E402


SEEDS = (42, 52, 62)
HORIZONS = (1, 3, 6, 12)
ORGAN_NAMES = (
    "respiratory",
    "cardiovascular",
    "renal",
    "coagulation",
    "hepatic",
    "cns",
)
OUT_DIR = V13_ROOT / "results" / "v4"
PLF_CACHE = OUT_DIR / "traj_mae_ci_cache.npz"
TRANSFORMER_CACHE = OUT_DIR / "transformertcr_trajectory_cache.npz"
RESULT_PATH = OUT_DIR / "trajectory_baselines_tcr.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=42)
    parser.add_argument("--force-transformer", action="store_true")
    return parser.parse_args()


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: (value.to(device, non_blocking=True) if hasattr(value, "to") else value)
        for key, value in batch.items()
    }


def run_transformer_ensemble(
    loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prediction_sum: np.ndarray | None = None
    organ_lab: np.ndarray | None = None
    organ_mask: np.ndarray | None = None
    stays: np.ndarray | None = None

    for seed_index, seed in enumerate(SEEDS):
        checkpoint = CODE_ROOT / f"runs/baselines/transformer_tcr_s{seed}/best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = StdTransformer(prior_dim=14, mode="TCR")
        model.load_state_dict(state.get("model_state_dict", state), strict=True)
        model.to(device).eval()

        seed_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        mask_parts: list[np.ndarray] = []
        stay_parts: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in loader:
                batch = move_batch(batch, device)
                output = model(batch)
                seed_parts.append(output["organ_future"].float().cpu().numpy())
                if seed_index == 0:
                    label_parts.append(batch["organ"].float().cpu().numpy())
                    mask_parts.append(batch["organ_mask"].float().cpu().numpy())
                    stay_parts.append(batch["stay_id"].cpu().numpy())

        seed_prediction = np.concatenate(seed_parts, axis=0)
        if prediction_sum is None:
            prediction_sum = seed_prediction.astype(np.float64)
            organ_lab = np.concatenate(label_parts, axis=0)
            organ_mask = np.concatenate(mask_parts, axis=0)
            stays = np.concatenate(stay_parts, axis=0)
        else:
            prediction_sum += seed_prediction
        print(f"  Transformer-TCR seed {seed}: done", flush=True)
        del model, seed_parts, seed_prediction
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert prediction_sum is not None
    assert organ_lab is not None and organ_mask is not None and stays is not None
    return (
        (prediction_sum / len(SEEDS)).astype(np.float32),
        organ_lab,
        organ_mask,
        stays,
    )


def save_cache(
    path: Path,
    pred: np.ndarray,
    organ_lab: np.ndarray,
    organ_mask: np.ndarray,
    stays: np.ndarray,
) -> None:
    np.savez(
        path,
        pred=pred,
        organ_lab=organ_lab,
        organ_mask=organ_mask,
        stays=stays,
    )


def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as cache:
        return cache["pred"], cache["organ_lab"], cache["organ_mask"], cache["stays"]


def assert_aligned(
    reference: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    comparison: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    ref_pred, ref_lab, ref_mask, ref_stays = reference
    cmp_pred, cmp_lab, cmp_mask, cmp_stays = comparison
    if ref_pred.shape != cmp_pred.shape:
        raise ValueError(f"Prediction shape mismatch: {ref_pred.shape} vs {cmp_pred.shape}")
    if not np.array_equal(ref_lab, cmp_lab):
        raise ValueError("Organ labels are not aligned")
    if not np.array_equal(ref_mask, cmp_mask):
        raise ValueError("Organ masks are not aligned")
    if not np.array_equal(ref_stays, cmp_stays):
        raise ValueError("ICU-stay order is not aligned")


def cluster_components(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    stay_inverse: np.ndarray,
    n_clusters: int,
) -> dict[str, np.ndarray]:
    organ_num_anchor = np.abs(prediction - target) * mask
    total_valid_anchor = (mask.sum(axis=1) > 0).astype(np.float64)
    prediction_total = (prediction * mask).sum(axis=1)
    target_total = (target * mask).sum(axis=1)
    total_num_anchor = np.abs(prediction_total - target_total) * total_valid_anchor

    organ_num = np.zeros((n_clusters, 6), dtype=np.float64)
    organ_den = np.zeros((n_clusters, 6), dtype=np.float64)
    total_num = np.zeros(n_clusters, dtype=np.float64)
    total_den = np.zeros(n_clusters, dtype=np.float64)
    np.add.at(organ_num, stay_inverse, organ_num_anchor)
    np.add.at(organ_den, stay_inverse, mask)
    np.add.at(total_num, stay_inverse, total_num_anchor)
    np.add.at(total_den, stay_inverse, total_valid_anchor)
    return {
        "organ_num": organ_num,
        "organ_den": organ_den,
        "total_num": total_num,
        "total_den": total_den,
    }


def evaluate_components(
    components: dict[str, np.ndarray], bootstrap_counts: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bootstrap_counts is None:
        organ_num = components["organ_num"].sum(axis=0, keepdims=True)
        organ_den = components["organ_den"].sum(axis=0, keepdims=True)
        total_num = np.asarray([components["total_num"].sum()])
        total_den = np.asarray([components["total_den"].sum()])
    else:
        organ_num = bootstrap_counts @ components["organ_num"]
        organ_den = bootstrap_counts @ components["organ_den"]
        total_num = bootstrap_counts @ components["total_num"]
        total_den = bootstrap_counts @ components["total_den"]
    organ_mae = organ_num / np.maximum(organ_den, 1.0)
    total_mae = total_num / np.maximum(total_den, 1.0)
    macro_mae = organ_mae.mean(axis=1)
    return total_mae, organ_mae, macro_mae


def scalar_summary(estimate: float, boot: np.ndarray) -> dict[str, object]:
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"estimate": float(estimate), "ci95": [float(low), float(high)]}


def model_summary(
    point: tuple[np.ndarray, np.ndarray, np.ndarray],
    boot: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, object]:
    point_total, point_organ, point_macro = point
    boot_total, boot_organ, boot_macro = boot
    return {
        "sofa_total_mae": scalar_summary(float(point_total[0]), boot_total),
        "macro_mae": scalar_summary(float(point_macro[0]), boot_macro),
        "organ_mae": {
            name: scalar_summary(float(point_organ[0, index]), boot_organ[:, index])
            for index, name in enumerate(ORGAN_NAMES)
        },
    }


def difference_summary(
    left_point: tuple[np.ndarray, np.ndarray, np.ndarray],
    right_point: tuple[np.ndarray, np.ndarray, np.ndarray],
    left_boot: tuple[np.ndarray, np.ndarray, np.ndarray],
    right_boot: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, object]:
    point_total = left_point[0] - right_point[0]
    point_organ = left_point[1] - right_point[1]
    point_macro = left_point[2] - right_point[2]
    boot_total = left_boot[0] - right_boot[0]
    boot_organ = left_boot[1] - right_boot[1]
    boot_macro = left_boot[2] - right_boot[2]
    return model_summary(
        (point_total, point_organ, point_macro),
        (boot_total, boot_organ, boot_macro),
    )


def skill_summary(
    model_point: tuple[np.ndarray, np.ndarray, np.ndarray],
    persistence_point: tuple[np.ndarray, np.ndarray, np.ndarray],
    model_boot: tuple[np.ndarray, np.ndarray, np.ndarray],
    persistence_boot: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, object]:
    total_point = 1.0 - model_point[0] / persistence_point[0]
    macro_point = 1.0 - model_point[2] / persistence_point[2]
    total_boot = 1.0 - model_boot[0] / persistence_boot[0]
    macro_boot = 1.0 - model_boot[2] / persistence_boot[2]
    return {
        "sofa_total_mae_skill": scalar_summary(float(total_point[0]), total_boot),
        "macro_mae_skill": scalar_summary(float(macro_point[0]), macro_boot),
        "interpretation": "positive values improve on persistence; negative values are worse",
    }


def main() -> None:
    args = parse_args()
    configure_cuda()
    device = DEVICE
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}", flush=True)

    if not PLF_CACHE.exists():
        raise FileNotFoundError(f"Established PLF trajectory cache not found: {PLF_CACHE}")
    plf_data = load_cache(PLF_CACHE)

    if TRANSFORMER_CACHE.exists() and not args.force_transformer:
        print(f"loading Transformer trajectory cache: {TRANSFORMER_CACHE}", flush=True)
        transformer_data = load_cache(TRANSFORMER_CACHE)
    else:
        dataset = PLFOGTV4Dataset("test")
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
        print(f"running Transformer-TCR trajectory ensemble: {len(dataset)} anchors", flush=True)
        transformer_data = run_transformer_ensemble(loader, device)
        save_cache(TRANSFORMER_CACHE, *transformer_data)
        print(f"saved Transformer cache: {TRANSFORMER_CACHE}", flush=True)

    assert_aligned(plf_data, transformer_data)
    pred_plf, organ_lab, organ_mask, stays = plf_data
    pred_transformer = transformer_data[0]
    unique_stays, stay_inverse = np.unique(stays, return_inverse=True)
    n_clusters = unique_stays.size

    rng = np.random.RandomState(args.boot_seed)
    bootstrap_counts = np.empty((args.n_boot, n_clusters), dtype=np.int16)
    for index in range(args.n_boot):
        sample = rng.randint(0, n_clusters, size=n_clusters)
        bootstrap_counts[index] = np.bincount(sample, minlength=n_clusters)

    results: dict[str, object] = {
        "analysis": {
            "task": "TCR absolute six-organ SOFA trajectory reconstruction",
            "models": {
                "persistence": "future organ SOFA equals organ SOFA at anchor",
                "transformer_tcr": "standard recurrent Transformer with actual post-anchor treatment sequence",
                "plf_ogt_tcr": "PLF-OGT with actual post-anchor treatment sequence",
            },
            "ensemble_seeds": list(SEEDS),
            "evaluation_mask": "organ valid at both anchor and evaluated horizon",
            "bootstrap": {
                "unit": "ICU stay",
                "paired": True,
                "n_boot": args.n_boot,
                "seed": args.boot_seed,
                "interval": "percentile 95% CI",
                "metric_recalculation": "global numerator/denominator within replicate",
            },
            "difference_direction": "PLF-OGT minus comparator; negative MAE favours PLF-OGT",
            "n_anchors": int(len(stays)),
            "n_clusters": int(n_clusters),
            "organ_order": list(ORGAN_NAMES),
        }
    }

    print(
        f"paired bootstrap: {args.n_boot} replicates, {n_clusters} ICU stays",
        flush=True,
    )
    for horizon in HORIZONS:
        target = organ_lab[:, horizon, :]
        common_mask = organ_mask[:, 0, :] * organ_mask[:, horizon, :]
        predictions = {
            "persistence": organ_lab[:, 0, :],
            "transformer_tcr": pred_transformer[:, horizon - 1, :],
            "plf_ogt_tcr": pred_plf[:, horizon - 1, :],
        }
        components = {
            name: cluster_components(
                prediction, target, common_mask, stay_inverse, n_clusters
            )
            for name, prediction in predictions.items()
        }
        point = {name: evaluate_components(value) for name, value in components.items()}
        boot = {
            name: evaluate_components(value, bootstrap_counts)
            for name, value in components.items()
        }

        horizon_result = {
            name: model_summary(point[name], boot[name]) for name in predictions
        }
        horizon_result["delta_plf_minus_persistence"] = difference_summary(
            point["plf_ogt_tcr"],
            point["persistence"],
            boot["plf_ogt_tcr"],
            boot["persistence"],
        )
        horizon_result["delta_plf_minus_transformer"] = difference_summary(
            point["plf_ogt_tcr"],
            point["transformer_tcr"],
            boot["plf_ogt_tcr"],
            boot["transformer_tcr"],
        )
        horizon_result["plf_skill_vs_persistence"] = skill_summary(
            point["plf_ogt_tcr"],
            point["persistence"],
            boot["plf_ogt_tcr"],
            boot["persistence"],
        )
        horizon_result["transformer_skill_vs_persistence"] = skill_summary(
            point["transformer_tcr"],
            point["persistence"],
            boot["transformer_tcr"],
            boot["persistence"],
        )
        horizon_result["valid_counts_by_organ"] = {
            name: int(common_mask[:, index].sum())
            for index, name in enumerate(ORGAN_NAMES)
        }
        horizon_result["valid_anchors_total_sofa"] = int(
            (common_mask.sum(axis=1) > 0).sum()
        )
        results[f"{horizon}h"] = horizon_result

        p_macro = point["persistence"][2][0]
        t_macro = point["transformer_tcr"][2][0]
        v_macro = point["plf_ogt_tcr"][2][0]
        d_p = horizon_result["delta_plf_minus_persistence"]["macro_mae"]
        d_t = horizon_result["delta_plf_minus_transformer"]["macro_mae"]
        print(
            f"  {horizon:>2}h macro-MAE | persistence={p_macro:.4f}, "
            f"Transformer={t_macro:.4f}, PLF={v_macro:.4f} | "
            f"PLF-persistence={d_p['estimate']:+.4f} "
            f"[{d_p['ci95'][0]:+.4f}, {d_p['ci95'][1]:+.4f}] | "
            f"PLF-Transformer={d_t['estimate']:+.4f} "
            f"[{d_t['ci95'][0]:+.4f}, {d_t['ci95'][1]:+.4f}]",
            flush=True,
        )

    RESULT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
