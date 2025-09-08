#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced OOD Recalibration (Fusion)
-----------------------------------
Learns a small logistic-regression OOD detector per model using:
  - Energy(T), MSP(T), Top1-Top2 margin, MaxLogit, Entropy(T)
  - Feature L2 norm, and diagonal Mahalanobis distance to class means
Inputs:
  - ID calibration set (uuid, style) from your train distribution (no UNK)
  - Real OOD holdout uuids (UNK images)

Outputs (saved into each model's meta JSON and registry):
  - T (temperature)
  - ood_calibrator: {
        "feature_names": [...],
        "coef": [...],
        "intercept": ...,
        "thr_prob": ...,
        "x_mean": [...], "x_std": [...],
        "feat_mean_per_class": [[...]*D x C],
        "feat_var_diag": [...]*D,
        "eps": 1e-6
    }

Usage Example
-------------
python recalibrate_ood_fusion.py \
  --train_csv /content/train_all.csv --img_dir /content/train_images_all \
  --out_dir /content/out \
  --ood_csv /content/ood.csv --ood_img_dir /content/ood_images \
  --models tf_efficientnet_b4_ns convnextv2_tiny.fcmae_ft_in22k_in1k \
  --num_id_samples 4000 --objective max_f1
"""

import os, json, argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import timm
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

STYLES = [
    'Ink scenery','comic','cyberpunk','futuristic UI','lowpoly',
    'oil painting','pixel','realistic','steampunk','water color'
]
STYLE2IDX = {s:i for i,s in enumerate(STYLES)}

# --------- Dataset ----------

class StyleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform, has_labels=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_labels = has_labels

    def _resolve_path(self, uuid: str):
        for ext in (".png",".jpg",".jpeg",".webp",".bmp"):
            p = os.path.join(self.img_dir, f"{uuid}{ext}")
            if os.path.exists(p): return p
        p = os.path.join(self.img_dir, uuid)
        if os.path.exists(p): return p
        raise FileNotFoundError(f"Image not found for uuid={uuid}")

    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        uuid = str(row['uuid'])
        img = cv2.imread(self._resolve_path(uuid), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image for uuid={uuid}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        if self.has_labels:
            return img, STYLE2IDX[row['style']]
        return img, uuid

# --------- Utils ----------

def build_eval_tf(size: int):
    return A.Compose([
        A.Resize(size, size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])

def build_model(model_name: str, num_classes: int = 10):
    return timm.create_model(model_name, pretrained=False, num_classes=num_classes)

def extract_feats(model, x):
    feats = model.forward_features(x)
    
    if isinstance(feats, (list, tuple)):
        feats = feats[-1]

    
    if feats.ndim == 3:
        
        if feats.shape[1] >= 1:
            feats = feats[:, 0, :]          # (B, C)
        else:
            feats = feats.mean(dim=1)       # (B, C)


    elif feats.ndim == 4:
        feats = torch.nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)  # (B, C)

    
    elif feats.ndim == 2:
        pass

    
    else:
        feats = feats.view(feats.size(0), -1)

    return feats

def energy_from_logits(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    return -T * torch.logsumexp(logits / T, dim=1)

def logits_to_stats(logits: torch.Tensor, T: float):
    scaled = logits / T
    probs = torch.softmax(scaled, dim=1)
    topv, topi = probs.max(dim=1)
    top2v, _ = probs.topk(k=min(2, probs.shape[1]), dim=1)
    margin = (top2v[:,0] - top2v[:,1]) if probs.shape[1] >= 2 else torch.zeros_like(topv)
    max_logit = logits.max(dim=1).values
    entropy = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=1)
    return topv, margin, max_logit, entropy  # MSP, margin, maxlogit, entropy

def diag_mahalanobis(feat: torch.Tensor, mean_per_class: torch.Tensor, var_diag: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inv_var = 1.0 / (var_diag + eps)
    x2 = (feat.unsqueeze(1) - mean_per_class.unsqueeze(0))**2  # [B,C,D]
    d = (x2 * inv_var.unsqueeze(0).unsqueeze(0)).sum(dim=2)     # [B,C]
    return d.min(dim=1).values  # min across classes

@torch.no_grad()
def forward_collect(model, loader, device, T: float):
    model.eval()
    logits_all, feats_all = [], []
    for batch in loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        feats = extract_feats(model, imgs)
        logits = (logits + model(torch.flip(imgs, dims=[3]))) / 2.0
        feats_all.append(feats.detach().cpu())
        logits_all.append(logits.detach().cpu())
    feats_all = torch.cat(feats_all, dim=0)
    logits_all = torch.cat(logits_all, dim=0)
    E = energy_from_logits(logits_all, T=T).cpu()
    msp, margin, maxlogit, entropy = logits_to_stats(logits_all, T=T)
    return feats_all, logits_all, E.cpu(), msp.cpu(), margin.cpu(), maxlogit.cpu(), entropy.cpu()

def fit_temperature(logits: np.ndarray, labels: np.ndarray, init_T: float = 1.0, max_iter: int = 50):
    T = torch.tensor([init_T], requires_grad=True)
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter)
    def _nll():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits_t / T.clamp(min=1e-3), labels_t)
        loss.backward()
        return loss
    opt.step(_nll)
    return float(T.clamp(min=1e-3, max=100.0).item())

def main():
    ap = argparse.ArgumentParser("Advanced OOD recalibration with feature fusion")
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--img_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--ood_csv", type=str, required=True)
    ap.add_argument("--ood_img_dir", type=str, required=True)
    ap.add_argument("--models", type=str, nargs="+", default=None)
    ap.add_argument("--num_id_samples", type=int, default=4000)
    ap.add_argument("--per_class", type=int, default=None)
    ap.add_argument("--objective", type=str, default="max_f1", choices=["max_f1","fpr95","fprx"])
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reg_path = Path(args.out_dir)/"models_registry.json"
    assert reg_path.exists(), f"Missing models_registry.json at {reg_path}"
    registry = json.load(open(reg_path, "r"))

    entries = []
    for it in registry:
        name = it["meta"]["model_name"]
        if args.models is None or name in args.models:
            entries.append(it)
    assert len(entries)>0, "No models selected."

    df_id = pd.read_csv(args.train_csv)
    df_id = df_id[df_id['style'].isin(list(STYLES))].drop_duplicates('uuid', keep='first')
    if args.per_class:
        parts = []
        for s, g in df_id.groupby('style'):
            parts.append(g.sample(n=min(len(g), args.per_class), random_state=42))
        df_id_cal = pd.concat(parts, ignore_index=True)
    else:
        df_id_cal = df_id.sample(n=min(len(df_id), args.num_id_samples), random_state=42)

    df_ood = pd.read_csv(args.ood_csv)
    assert 'uuid' in df_ood.columns, "ood_csv must have column: uuid"

    updated = []

    for item in entries:
        model_name = item["meta"]["model_name"]
        size = item["meta"]["img_size"]
        wpath = item["weight_path"]
        meta_path = Path(wpath).with_suffix("").parent / (Path(wpath).stem + "_meta.json")

        print(f"\n=== Calibrating {model_name}@{size} with feature fusion ===")

        tf_eval = build_eval_tf(size)
        ds_id = StyleDataset(df_id_cal, args.img_dir, tf_eval, has_labels=True)
        ds_ood = StyleDataset(df_ood, args.ood_img_dir, tf_eval, has_labels=False)
        ld_id = DataLoader(ds_id, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)
        ld_ood = DataLoader(ds_ood, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

        model = build_model(model_name, num_classes=10)
        state = torch.load(wpath, map_location='cpu')
        model.load_state_dict(state, strict=True)
        model.to(device).eval()

        # 1) temperature
        print("  • gathering logits on ID for T...")
        logits_list, labels_list = [], []
        for imgs, labels in ld_id:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            logits = (logits + model(torch.flip(imgs, dims=[3]))) / 2.0
            logits_list.append(logits.detach().cpu())
            labels_list.append(labels.detach().cpu())
        logits_id_for_T = torch.cat(logits_list,0).numpy()
        labels_id_for_T = torch.cat(labels_list,0).numpy()
        Topt = fit_temperature(logits_id_for_T, labels_id_for_T, init_T=1.0, max_iter=50)
        print(f"  • Optimal T = {Topt:.4f}")

        # 2) collect features & stats
        print("  • collecting ID features/stats...")
        feats_id, logits_id, E_in, MSP_in, MARG_in, MAXL_in, ENT_in = forward_collect(model, ld_id, device, T=Topt)
        print("  • collecting OOD features/stats...")
        feats_ood, logits_ood, E_out, MSP_out, MARG_out, MAXL_out, ENT_out = forward_collect(model, ld_ood, device, T=Topt)

        # class means (ID) + diag var
        labels_id = []
        for _, labels in DataLoader(ds_id, batch_size=args.batch, shuffle=False, num_workers=2):
            labels_id.append(labels)
        labels_id = torch.cat(labels_id,0)
        C = len(STYLES); D = feats_id.shape[1]
        mean_per_class = torch.zeros(C, D)
        for c in range(C):
            idx = (labels_id==c).nonzero(as_tuple=True)[0]
            mean_per_class[c] = feats_id[idx].mean(dim=0) if len(idx)>0 else feats_id.mean(dim=0)
        var_diag = feats_id.var(dim=0).clamp_min(1e-6)

        maha_in  = diag_mahalanobis(feats_id, mean_per_class, var_diag, eps=1e-6)
        maha_out = diag_mahalanobis(feats_ood, mean_per_class, var_diag, eps=1e-6)

        def to_np(t): return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t
        X_in  = np.column_stack([ to_np(E_in),  to_np(MSP_in),  to_np(MARG_in),  to_np(MAXL_in),  to_np(ENT_in),
                                  to_np(feats_id.norm(dim=1)), to_np(maha_in) ])
        X_out = np.column_stack([ to_np(E_out), to_np(MSP_out), to_np(MARG_out), to_np(MAXL_out), to_np(ENT_out),
                                  to_np(feats_ood.norm(dim=1)), to_np(maha_out) ])
        y_in  = np.zeros(len(X_in), dtype=np.int64)
        y_out = np.ones(len(X_out), dtype=np.int64)
        X = np.vstack([X_in, X_out])
        y = np.concatenate([y_in, y_out])

        x_mean = X.mean(axis=0); x_std = X.std(axis=0) + 1e-6
        Xz = (X - x_mean) / x_std

        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(Xz, y)
        probs = clf.predict_proba(Xz)[:,1]

        if args.objective == "max_f1":
            thr_candidates = np.quantile(probs, np.linspace(0.01, 0.99, 199))
            best_f1, best_thr = -1.0, 0.5
            for thr in thr_candidates:
                pred = (probs >= thr).astype(int)
                f1 = f1_score(y, pred, average='binary')
                if f1 > best_f1:
                    best_f1, best_thr = f1, float(thr)
            thr_prob = best_thr
        elif args.objective in ("fpr95","fprx"):
            pi = probs[:len(X_in)]
            thr_prob = float(np.quantile(pi, 1.0 - args.fpr))
        else:
            thr_prob = 0.5

        meta = json.load(open(meta_path, "r"))
        meta["T"] = float(Topt)
        meta["ood_calibrator"] = {
            "feature_names": ["energy","msp","margin","maxlogit","entropy","feat_norm","maha_diag_min"],
            "coef": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]),
            "thr_prob": float(thr_prob),
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "feat_mean_per_class": mean_per_class.numpy().tolist(),
            "feat_var_diag": var_diag.numpy().tolist(),
            "eps": 1e-6
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        item["meta"]["T"] = float(Topt)
        item["meta"]["ood_calibrator"] = meta["ood_calibrator"]

        print(f"  ✔ Saved calibrator with thr_prob={thr_prob:.3f}")

        del model
        torch.cuda.empty_cache()

    with open(reg_path, "w") as f:
        json.dump(registry, f, indent=2)
    print("\nDONE. Updated models_registry.json with feature-fusion OOD calibrators.")

if __name__ == "__main__":
    main()
