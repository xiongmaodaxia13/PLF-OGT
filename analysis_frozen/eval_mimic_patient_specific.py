#!/usr/bin/env python
"""MIMIC-IV patient-specificity (4条件×3seed, stay-level shuffle)."""
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
from v6.models.plf_ogt_v4 import PLFOGTV4Model
from v6.models.v4_axes import load_v4_proxy_contract
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path(r"F:/MIMIC3_1/V12")
OUTDIR = Path(r"F:/MIMIC3_1/V13/results_mimic")
SEEDS = [42, 52, 62]; H6 = 5

def move(batch, dev):
    return {k:(v.to(dev,non_blocking=True) if hasattr(v,"to") else v) for k,v in batch.items()}

def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True)); return e / e.sum(axis=axis, keepdims=True)

def build_model(spec, seed, dev):
    ck = torch.load(REPO / f"runs_mimic/v4/full_s5_s{seed}/best.pt", map_location="cpu", weights_only=False)
    m = PLFOGTV4Model(prior_dim=14, d_model=128, n_heads=4, n_residual=6,
                      event_layers=2, concept_layers=1, residual_layers=1,
                      transition_layers=2, dropout=0.0, transition_mode="modulation",
                      n_horizons=12, r_encoder_type="slot_attention", r_n_iters=3, spec=spec)
    m.load_state_dict(ck["model_state_dict"], strict=False); m.to(dev).eval()
    return m

def encode_all_R(model, loader, dev):
    all_R=[]; all_stays=[]
    with torch.inference_mode():
        for b in loader:
            b = move(b, dev)
            branch = model.encode_branches(b)
            all_R.append(branch.residual.cpu()); all_stays.append(b["stay_id"].cpu())
    return torch.cat(all_R,dim=0), torch.cat(all_stays,dim=0)

def stay_level_shuffle_R(R_all, stays, rng):
    unique_stays = torch.unique(stays); n = len(unique_stays)
    perm = torch.randperm(n, generator=rng)
    identity = torch.arange(n)
    while (perm == identity).any():
        perm = torch.randperm(n, generator=rng)
    stay_map = {unique_stays[i].item(): unique_stays[perm[i]].item() for i in range(n)}
    stay_repr = {}
    for s in unique_stays:
        mask = stays == s; stay_repr[s.item()] = R_all[mask].mean(dim=0)
    R_shuffled = torch.zeros_like(R_all)
    for i in range(len(stays)):
        R_shuffled[i] = stay_repr[stay_map[stays[i].item()]]
    return R_shuffled

def rollout_with_R(model, loader, dev, R_override):
    all_p=[]; all_y=[]
    with torch.inference_mode():
        idx=0
        for b in loader:
            b=move(b,dev); B=b["organ"].shape[0]
            branch=model.encode_branches(b)
            if R_override is not None:
                branch.residual=R_override[idx:idx+B].to(dev)
            out=model.rollout_from_state(b, branch, stage="conditioned", future_treatment_mode="actual")
            logits=out["class_logits"][:,H6,:].float().cpu().numpy()
            all_p.append(softmax_np(logits)[:,0])
            organ=b["organ"].cpu().numpy(); omask=b["organ_mask"].cpu().numpy()
            o_now=organ[:,0,:]; m_now=omask[:,0,:]; o_6h=organ[:,H6+1,:]; m_6h=omask[:,H6+1,:]
            delta=((m_now*m_6h)*(o_6h-o_now)).sum(axis=1)
            all_y.append((delta>=2).astype(float))
            idx+=B
    return np.concatenate(all_p), np.concatenate(all_y)

def main():
    configure_cuda(); dev=DEVICE
    print("="*60,flush=True)
    print("MIMIC-IV patient-specificity (4条件×3seed)",flush=True)
    print("="*60,flush=True)
    spec=load_v4_proxy_contract(REPO/"configs/v4/v4_proxy_contract.json")
    ds=MIMICDataset(split="test")
    loader=DataLoader(ds,batch_size=512,shuffle=False,collate_fn=collate,num_workers=0)
    print(f"MIMIC test: {len(ds)}\n",flush=True)

    CONDITIONS=["matched","shuffled","mean","query_only"]
    all_results={c:{"auprc_list":[],"auroc_list":[],"macro_mae_list":[]} for c in CONDITIONS}

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===",flush=True)
        model=build_model(spec,seed,dev)
        print("  编码全部R...",flush=True)
        R_all,stays=encode_all_R(model,loader,dev)

        rng=torch.Generator(device="cpu").manual_seed(seed*100)
        for cond in CONDITIONS:
            if cond=="matched": R_override=None
            elif cond=="shuffled": R_override=stay_level_shuffle_R(R_all,stays,rng)
            elif cond=="mean": R_override=R_all.mean(dim=0,keepdim=True).expand_as(R_all)
            elif cond=="query_only": R_override=torch.zeros_like(R_all)

            p,y=rollout_with_R(model,loader,dev,R_override)
            mask=np.ones(len(y),dtype=bool)
            if y[mask].sum()>5 and len(set(y[mask]))>1:
                ap=float(average_precision_score(y[mask],p[mask]))
                auc=float(roc_auc_score(y[mask],p[mask]))
            else:
                ap=auc=float("nan")
            all_results[cond]["auprc_list"].append(ap)
            all_results[cond]["auroc_list"].append(auc)
            print(f"  {cond:<14} AUPRC={ap:.4f} AUROC={auc:.4f}",flush=True)
        del model; torch.cuda.empty_cache()

    # 汇总
    print(f"\n{'='*60}",flush=True)
    print(f"{'条件':<14}{'AUPRC':<12}{'AUROC':<12}")
    print("-"*40)
    final={}
    for cond in CONDITIONS:
        ap=float(np.mean(all_results[cond]["auprc_list"]))
        ap_std=float(np.std(all_results[cond]["auprc_list"]))
        auc=float(np.mean(all_results[cond]["auroc_list"]))
        final[cond]={"auprc_mean":ap,"auprc_std":ap_std,"auroc_mean":auc}
        print(f"{cond:<14}{ap:<12.4f}{auc:<12.4f}",flush=True)

    matched_ap=final["matched"]["auprc_mean"]
    for cond in CONDITIONS:
        final[cond]["delta_auprc"]=final[cond]["auprc_mean"]-matched_ap
    print("="*60,flush=True)

    OUTDIR.mkdir(parents=True,exist_ok=True)
    json.dump(final,open(OUTDIR/"frozen_mimic_patient_specific.json","w"),indent=2,ensure_ascii=False,default=float)
    print(f"\n保存: {OUTDIR/'frozen_mimic_patient_specific.json'}",flush=True)

if __name__=="__main__":
    main()
