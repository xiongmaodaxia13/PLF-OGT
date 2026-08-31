#!/usr/bin/env python
"""PLF-OGT vs Transformer 推理速度 benchmark —— 第三档 (#7).

陆老师 #7: 当结果弱于 Transformer 时, 用 runtime 优势补偿.
本脚本在同一 GPU(4090)上, 相同 batch/输入, 测两模型:
  - 单 batch 推理时间 (ms, warmup 后多次取中位数)
  - 峰值显存 (MB)
  - 吞吐 (samples/s)
  - 参数量

输出: results/v4/runtime_benchmark.json
用法: python scripts/benchmark_runtime.py
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # 4090
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from v6.config import DEVICE, RESULTS_DIR, configure_cuda
from v6.data.dataset import collate
from v6.data.v4_dataset import PLFOGTV4Dataset
from v6.models.plf_ogt_v4 import PLFOGTV4Model
from v6.models.std_transformer import StdTransformer
from v6.models.v4_axes import load_v4_proxy_contract

REPO = Path(__file__).resolve().parents[1]
BATCH = 512
N_TIMED = 30  # 计时次数


def move(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v)
            for k, v in batch.items()}


def bench(model, loader, dev, n_timed, forward_kw):
    """返回 ms/batch(中位数), 峰值显存MB, 吞吐samples/s."""
    model.to(dev).eval()
    # warmup
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            if i >= 3: break
            batch = move(batch, dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                _ = model(batch, **forward_kw)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(dev)

    times = []
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            if i >= n_timed: break
            batch = move(batch, dev)
            torch.cuda.synchronize(dev); t0 = time.perf_counter()
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                _ = model(batch, **forward_kw)
            torch.cuda.synchronize(dev); t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    peak_mem = torch.cuda.max_memory_allocated(dev) / 1024 / 1024
    ms = float(np.median(times))
    return ms, float(peak_mem), BATCH / (ms / 1000)


def main():
    configure_cuda(); dev = DEVICE
    print("=" * 60, flush=True)
    print(f"第三档 #7: 推理速度 benchmark ({torch.cuda.get_device_name(dev)})", flush=True)
    print("=" * 60, flush=True)

    spec = load_v4_proxy_contract(REPO / "configs/v4/v4_proxy_contract.json")
    ds = PLFOGTV4Dataset("test")
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, collate_fn=collate, num_workers=0)
    print(f"test: {len(ds)}, batch={BATCH}", flush=True)

    results = {}

    # PLF-OGT (TCR)
    print("\n[PLF-OGT TCR]...", flush=True)
    ck = torch.load(REPO / "runs/v4/full_s5_s42/best.pt", map_location="cpu", weights_only=False)
    plf = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4, n_residual=6,
                        event_layers=2, concept_layers=1, residual_layers=1,
                        transition_layers=2, dropout=0.0, transition_mode="modulation",
                        n_horizons=12, r_encoder_type="slot_attention", r_n_iters=3, spec=spec)
    plf.load_state_dict(ck["model_state_dict"], strict=False)
    plf_params = sum(p.numel() for p in plf.parameters())
    ms_plf, mem_plf, thr_plf = bench(plf, loader, dev, N_TIMED, {"stage": "conditioned", "future_treatment_mode": "actual"})
    results["PLF-OGT"] = {"params": plf_params, "ms_per_batch": ms_plf, "peak_mem_MB": mem_plf, "throughput_sps": thr_plf}
    print(f"  {ms_plf:.2f} ms/batch | {mem_plf:.0f} MB | {thr_plf:.0f} sps | {plf_params:,} params", flush=True)
    del plf; torch.cuda.empty_cache()

    # Transformer (TCR)
    print("\n[Transformer TCR]...", flush=True)
    sd = torch.load(REPO / "runs/baselines/transformer_tcr_s42/best.pt", map_location=dev, weights_only=False)
    tr = StdTransformer(prior_dim=14, mode="TCR")
    tr.load_state_dict(sd.get("model_state_dict", sd))
    tr_params = sum(p.numel() for p in tr.parameters())
    ms_tr, mem_tr, thr_tr = bench(tr, loader, dev, N_TIMED, {})
    results["Transformer"] = {"params": tr_params, "ms_per_batch": ms_tr, "peak_mem_MB": mem_tr, "throughput_sps": thr_tr}
    print(f"  {ms_tr:.2f} ms/batch | {mem_tr:.0f} MB | {thr_tr:.0f} sps | {tr_params:,} params", flush=True)
    del tr; torch.cuda.empty_cache()

    # 对比
    speed_ratio = ms_plf / ms_tr
    results["comparison"] = {
        "PLF_vs_TR_speed": f"{speed_ratio:.2f}x (PLF {'慢' if speed_ratio > 1 else '快'})",
        "PLF_vs_TR_params": f"{plf_params / tr_params:.2f}x",
        "PLF_vs_TR_mem": f"{mem_plf / mem_tr:.2f}x",
        "gpu": torch.cuda.get_device_name(dev),
        "batch_size": BATCH,
        "n_timed": N_TIMED,
    }
    print("\n" + "=" * 60, flush=True)
    print(f"对比: PLF/Transformer 速度 = {speed_ratio:.2f}x ({'PLF更慢' if speed_ratio > 1 else 'PLF更快'})", flush=True)
    print(f"      PLF/Transformer 参数 = {plf_params / tr_params:.2f}x", flush=True)
    print(f"      PLF/Transformer 显存 = {mem_plf / mem_tr:.2f}x", flush=True)
    print("=" * 60, flush=True)

    out = RESULTS_DIR / "v4" / "runtime_benchmark.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n保存: {out}", flush=True)


if __name__ == "__main__":
    main()
