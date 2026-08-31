#!/usr/bin/env python
"""Transformer 轨迹 MAE 评价 (与 PLF-OGT 同口径).

跑 Transformer TCR + OLP 的 organ_future, 用同一 compute_mae 口径算
1/3/6/12h 的 sofa_total_mae / organ_mae / macro_mae.

输出: results/v4/transformer_traj_mae.json
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("CUDA_VISIBLE_DEVICES","1")
sys.path.insert(0, r"F:/MIMIC3_1/V12")
sys.stdout.reconfigure(encoding="utf-8")

from v6.config import DEVICE, configure_cuda
from v6.data.dataset import collate
from v6.data.v4_dataset import PLFOGTV4Dataset
from v6.models.std_transformer import StdTransformer
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path(r"F:/MIMIC3_1/V12")
OUT = Path(r"F:/MIMIC3_1/V13/results/v4/transformer_traj_mae.json")
SEEDS=[42,52,62]; HORIZONS=[1,3,6,12]


def move(batch, dev):
    return {k:(v.to(dev,non_blocking=True) if hasattr(v,"to") else v) for k,v in batch.items()}


def compute_mae(pred, target, mask):
    """全局口径: diff.sum()/mask.sum()."""
    diff=np.abs(pred-target)*mask
    return diff.sum()/max(mask.sum(),1)


def run_mode(loader, dev, mode):
    """mode='actual'(TCR) / 'zero'(OLP). 3-seed ensemble organ_future."""
    ens_preds=[]
    organ_lab_all=[]; organ_mask_all=[]
    for seed in SEEDS:
        ckpt=REPO/f"runs/baselines/transformer_{'tcr' if mode=='actual' else 'olp'}_s{seed}/best.pt"
        if not ckpt.exists():
            # OLP 在 runs_v2_backup
            ckpt=REPO/f"runs_v2_backup/baselines/transformer_{'tcr' if mode=='actual' else 'olp'}_s{seed}/best.pt"
        sd=torch.load(ckpt,map_location="cpu",weights_only=False)
        model=StdTransformer(prior_dim=14, mode=("TCR" if mode=="actual" else "OLP"))
        model.load_state_dict(sd.get("model_state_dict",sd))
        model.to(dev).eval()
        ps=[]
        with torch.inference_mode():
            for batch in loader:
                batch=move(batch,dev)
                with torch.autocast(dev.type,dtype=torch.bfloat16):
                    out=model(batch)
                ps.append(out["organ_future"][:, :12, :].float().cpu().numpy())  # (B,12,6)
                if seed==SEEDS[0]:  # 只在第一个seed累积标签(全量)
                    organ_lab_all.append(batch["organ"].cpu().numpy())
                    organ_mask_all.append(batch["organ_mask"].cpu().numpy())
        ens_preds.append(np.concatenate(ps))
        print(f"    TR {mode} seed {seed}: done",flush=True)
        del model; torch.cuda.empty_cache()
    organ_lab=np.concatenate(organ_lab_all); organ_mask=np.concatenate(organ_mask_all)
    return np.mean(ens_preds,axis=0), organ_lab, organ_mask


def main():
    configure_cuda(); dev=DEVICE
    print("="*60,flush=True)
    print("Transformer 轨迹 MAE (3-seed, 同 PLF 口径)",flush=True)
    print("="*60,flush=True)
    ds=PLFOGTV4Dataset("test")
    loader=DataLoader(ds,batch_size=512,shuffle=False,collate_fn=collate,num_workers=0)
    print(f"test: {len(ds)}\n",flush=True)

    results={}
    for mode_name,mode in [("TCR","actual"),("OLP","zero")]:
        print(f"\n=== Transformer {mode_name} ===",flush=True)
        pred, organ_lab, organ_mask = run_mode(loader, dev, mode)
        tgt=organ_lab[:,1:13,:]   # (N,12,6)
        mk=organ_mask[:,1:13,:]
        mr={}
        for h in HORIZONS:
            hi=h-1
            omh=mk[:,hi,:]
            # sofa_total
            pred_sum=(pred[:,hi,:]*omh).sum(axis=-1)
            true_sum=(tgt[:,hi,:]*omh).sum(axis=-1)
            nval=omh.sum(axis=-1)
            mtot=(nval>0).astype(float)
            sofa_mae=compute_mae(pred_sum,true_sum,mtot)
            # 分量
            organ_maes=[compute_mae(pred[:,hi,o],tgt[:,hi,o],mk[:,hi,o]) for o in range(6)]
            mr[f"{h}h"]={"sofa_total_mae":float(sofa_mae),"organ_mae":[float(x) for x in organ_maes],
                         "macro_mae":float(np.mean(organ_maes))}
            print(f"  {h:>2}h: 总分{sofa_mae:.4f} macro{np.mean(organ_maes):.4f}",flush=True)
        results[mode_name]=mr

    OUT.write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(f"\n保存: {OUT}",flush=True)


if __name__=="__main__":
    main()
