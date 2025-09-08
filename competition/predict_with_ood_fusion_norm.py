
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_with_ood_fusion_norm.py
--------------------------------
- Flip-TTA only (match calibration pass).
- OOD probability is normalized per-model: p_norm = p_ood / thr_prob_model.
- Separate ensemble weights for CLS vs OOD:
    * w_cls = best_f1_macro ** W_EXP_CLS   (can exclude weak models via CLS_EXCLUDE)
    * w_ood = best_f1_macro ** W_EXP_OOD
- Final OOD decision uses a single global scale OOD_THR_NORM (~1.05).
- Optional per-size dumps that use the **normalized** OOD criterion.

Env vars
--------
UNK_TOPP_MAX : float in [0,1], if top-1 prob > this, never UNK (default 1.00 disables)
OOD_THR_NORM : float, normalized OOD threshold (default 1.05; try 1.04–1.08)
W_EXP_CLS    : float, exponent for CLS weights (default 2.0)
W_EXP_OOD    : float, exponent for OOD weights (default 1.0)
CLS_EXCLUDE  : comma/space-separated model names to exclude from CLS (still used for OOD)

Usage
-----
python predict_with_ood_fusion_norm.py \
  --test_csv data/test.csv --img_dir data/test \
  --out_dir out \
  --models tf_efficientnet_b4_ns convnextv2_tiny.fcmae_ft_in22k_in1k \
  --submission out/submission.csv \
  --dump_per_size
"""
from collections import defaultdict
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

torch.set_grad_enabled(False)

# ---------- Environment knobs ----------
UNK_TOPP_MAX  = float(os.getenv("UNK_TOPP_MAX", "1.00"))
OOD_THR_NORM  = float(os.getenv("OOD_THR_NORM", "1.05"))
W_EXP_CLS     = float(os.getenv("W_EXP_CLS",  "2.0"))
W_EXP_OOD     = float(os.getenv("W_EXP_OOD",  "1.0"))
CLS_EXCLUDE   = os.getenv("CLS_EXCLUDE", "").replace(",", " ").split()
CLS_EXCLUDE   = set([s for s in CLS_EXCLUDE if s.strip()])

STYLES = [
    'Ink scenery','comic','cyberpunk','futuristic UI','lowpoly',
    'oil painting','pixel','realistic','steampunk','water color'
]
IDX2STYLE = {i:s for i,s in enumerate(STYLES)}
UNK_NAME = "UNK"

# ---------- Helpers ----------
def base_logits(model, x):
    # single-scale + flip
    return (model(x) + model(torch.flip(x, dims=[3]))) / 2.0

class StyleDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
    def _resolve(self, uuid):
        for ext in (".png",".jpg",".jpeg",".webp",".bmp"):
            p = os.path.join(self.img_dir, f"{uuid}{ext}")
            if os.path.exists(p): return p
        p = os.path.join(self.img_dir, uuid)
        if os.path.exists(p): return p
        raise FileNotFoundError(f"Image not found for uuid={uuid}")
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        uuid = str(self.df.iloc[i]['uuid'])
        img = cv2.imread(self._resolve(uuid), cv2.IMREAD_COLOR)
        if img is None: raise FileNotFoundError(f"Failed read {uuid}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        return img, uuid

def build_eval_tf(size:int):
    return A.Compose([
        A.Resize(size,size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])

def build_model(model_name, num_classes=10):
    return timm.create_model(model_name, pretrained=False, num_classes=num_classes)

def extract_feats(model, x):
    feats = model.forward_features(x)
    
    if isinstance(feats, (list, tuple)):
        feats = feats[-1]
    if feats.ndim == 3:
        # (B, N, C) -> CLS token
        feats = feats[:, 0, :] if feats.shape[1] >= 1 else feats.mean(dim=1)
    elif feats.ndim == 4:
        # (B, C, H, W) -> GAP
        feats = torch.nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)
    elif feats.ndim == 2:
        pass
    else:
        feats = feats.view(feats.size(0), -1)
    return feats

def energy_from_logits(logits: torch.Tensor, T: float = 1.0):
    return -T * torch.logsumexp(logits / max(T, 1e-6), dim=1)

def logits_to_stats(logits: torch.Tensor, T: float):
    scaled = logits / max(T, 1e-6)
    probs = torch.softmax(scaled, dim=1)
    topv, _ = probs.max(dim=1)
    top2v, _ = probs.topk(k=min(2, probs.shape[1]), dim=1)
    margin = (top2v[:,0]-top2v[:,1]) if probs.shape[1]>=2 else torch.zeros_like(topv)
    maxlogit = logits.max(dim=1).values
    entropy = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=1)
    return topv, margin, maxlogit, entropy

def diag_maha(feat: torch.Tensor, mean_per_class: torch.Tensor, var_diag: torch.Tensor, eps: float = 1e-6):
    # feat: (B, C), mean_per_class: (K, C), var_diag: (C,) or (K, C)
    if var_diag.ndim == 1:
        inv_var = 1.0 / (var_diag + eps)              # (C,)
        inv_var = inv_var.unsqueeze(0).unsqueeze(0)   # (1,1,C)
    else:
        inv_var = 1.0 / (var_diag + eps)              # (K, C)
        inv_var = inv_var.unsqueeze(0)                # (1,K,C)
    d = ((feat.unsqueeze(1) - mean_per_class.unsqueeze(0))**2 * inv_var).sum(dim=2)  # (B, K)
    return d.min(dim=1).values                        # (B,)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser("Predict with OOD fusion (normalized)")
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--img_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--models", type=str, nargs="+", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--submission", type=str, required=True)
    ap.add_argument("--dump_per_size", action="store_true",
                    help="Also write per-size CSVs: out/sub_size_{size}.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reg_path = Path(args.out_dir)/"models_registry.json"
    assert reg_path.exists(), f"Missing models_registry.json at {reg_path}"
    registry = json.load(open(reg_path,"r"))

    # filter selected entries that have calibrator
    entries = []
    for it in registry:
        name = it["meta"]["model_name"]
        if args.models is None or name in args.models:
            if "ood_calibrator" in it["meta"]:
                entries.append(it)
    assert len(entries)>0, "No models with ood_calibrator found. Run recalibrate_ood_fusion.py first."

    test_df = pd.read_csv(args.test_csv)
    assert 'uuid' in test_df.columns, "test_csv must contain uuid"

    # group by image size
    by_size = {}
    for it in entries:
        size = it["meta"]["img_size"]
        by_size.setdefault(size, []).append(it)

    # Accumulators across sizes
    acc_logits_sum = {}                 # uuid -> np.array(10,)
    acc_wsum_cls   = defaultdict(float) # uuid -> float
    acc_pood_norm_wsum = defaultdict(float)  # uuid -> float
    acc_wsum_ood   = defaultdict(float) # uuid -> float

    for size, infos in by_size.items():
        tf = build_eval_tf(size)
        ds = StyleDataset(test_df, args.img_dir, tf)
        ld = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=False)

        # Load models of this size
        models = []
        for it in infos:
            name = it["meta"]["model_name"]
            wpath = it["weight_path"]
            meta  = it["meta"]
            model = build_model(name, num_classes=10)
            state = torch.load(wpath, map_location='cpu')
            model.load_state_dict(state, strict=True)
            model.to(device).eval()
            # per-model weights
            base_w = float(meta.get("best_f1_macro", 1.0))
            w_cls = (base_w ** W_EXP_CLS) if (name not in CLS_EXCLUDE) else 0.0
            w_ood = (base_w ** W_EXP_OOD)
            models.append((name, model, meta, w_cls, w_ood))

        # Per-size accumulators for optional dumps
        logits_sum_size_by_uuid = {}
        wsum_size_cls_by_uuid   = defaultdict(float)
        pood_norm_wsum_by_uuid  = defaultdict(float)
        wsum_size_ood_by_uuid   = defaultdict(float)

        for imgs, uuids in tqdm(ld, desc=f"Infer@{size}"):
            with torch.inference_mode():
                imgs = imgs.to(device, non_blocking=True)

                # temp holders within the batch
                logits_wsum_batch = None                     # (B,10) cls weighted
                pnorm_wsum_batch  = None                     # (B,)   ood weighted
                wsum_cls_batch    = 0.0
                wsum_ood_batch    = 0.0

                for name, model, meta, w_cls, w_ood in models:
                    # base pass (flip only)
                    logits_base = base_logits(model, imgs)
                    T = float(meta.get("T", 1.0))

                    # OOD logistic fusion (per-model)
                    feats = extract_feats(model, imgs)
                    cal = meta["ood_calibrator"]
                    mean_per_class = torch.tensor(cal["feat_mean_per_class"], dtype=torch.float32, device=imgs.device)
                    var_diag       = torch.tensor(cal["feat_var_diag"],     dtype=torch.float32, device=imgs.device)
                    E_b   = energy_from_logits(logits_base, T=T)
                    MSP_b, MARG_b, MAXL_b, ENT_b = logits_to_stats(logits_base, T=T)
                    maha  = diag_maha(feats, mean_per_class, var_diag, eps=float(cal.get("eps", 1e-6)))

                    X   = torch.stack([E_b, MSP_b, MARG_b, MAXL_b, ENT_b, feats.norm(dim=1), maha], dim=1).detach().cpu().numpy()
                    Xz  = (X - np.array(cal["x_mean"])) / (np.array(cal["x_std"]) + 1e-6)
                    logit_ood = Xz.dot(np.array(cal["coef"]).reshape(-1,1)).squeeze(1) + cal["intercept"]
                    p_ood     = 1.0 / (1.0 + np.exp(-logit_ood))  # (B,)

                    # normalize by this model's threshold
                    thr_prob  = float(cal["thr_prob"])
                    p_norm    = p_ood / max(thr_prob, 1e-6)       # (B,)

                    # accumulate OOD (weighted by w_ood)
                    pw = (p_norm * w_ood).astype('float64')
                    pnorm_wsum_batch = pw if pnorm_wsum_batch is None else (pnorm_wsum_batch + pw)
                    wsum_ood_batch   += w_ood

                    # accumulate CLS logits (weighted by w_cls)
                    if w_cls > 0.0:
                        lw = (logits_base * w_cls).detach().cpu().numpy()  # (B,10)
                        logits_wsum_batch = lw if logits_wsum_batch is None else (logits_wsum_batch + lw)
                        wsum_cls_batch    += w_cls

                # push to per-size & global accumulators
                for i, u in enumerate(uuids):
                    u = str(u)
                    # CLS
                    if logits_wsum_batch is not None and wsum_cls_batch > 0:
                        if u not in acc_logits_sum:
                            acc_logits_sum[u] = logits_wsum_batch[i].copy()
                        else:
                            acc_logits_sum[u] += logits_wsum_batch[i]
                        acc_wsum_cls[u] += wsum_cls_batch

                        # per-size for dump
                        if u not in logits_sum_size_by_uuid:
                            logits_sum_size_by_uuid[u] = logits_wsum_batch[i].copy()
                        else:
                            logits_sum_size_by_uuid[u] += logits_wsum_batch[i]
                        wsum_size_cls_by_uuid[u] += wsum_cls_batch

                    # OOD
                    if pnorm_wsum_batch is not None and wsum_ood_batch > 0:
                        acc_pood_norm_wsum[u] += float(pnorm_wsum_batch[i])
                        acc_wsum_ood[u]       += wsum_ood_batch

                        # per-size
                        pood_norm_wsum_by_uuid[u] += float(pnorm_wsum_batch[i])
                        wsum_size_ood_by_uuid[u]  += wsum_ood_batch

        # free models for this size
        for _, m, _, _, _ in models:
            del m
        torch.cuda.empty_cache()

        # optional per-size dump (normalized OOD criterion)
        if args.dump_per_size:
            uu, pp = [], []
            for u in test_df['uuid'].astype(str):
                if (u not in wsum_size_cls_by_uuid) or (wsum_size_cls_by_uuid[u] <= 0):
                    # if a uuid got excluded from CLS (e.g., all CLS models excluded), we skip this row
                    # you can also decide to fall back to global CLS later
                    continue

                logits_s = logits_sum_size_by_uuid[u] / max(wsum_size_cls_by_uuid[u], 1e-6)
                p = np.exp(logits_s - logits_s.max()); p /= p.sum()
                topv, top_idx = float(p.max()), int(p.argmax())

                if wsum_size_ood_by_uuid[u] > 0:
                    p_norm_s = pood_norm_wsum_by_uuid[u] / max(wsum_size_ood_by_uuid[u], 1e-6)
                else:
                    # if no OOD contributors for this size, default to 0 (ID)
                    p_norm_s = 0.0
                is_unk = (p_norm_s >= OOD_THR_NORM)
                if topv > UNK_TOPP_MAX: is_unk = False

                uu.append(u)
                pp.append(UNK_NAME if is_unk else IDX2STYLE[top_idx])

            out_ps = Path(args.out_dir) / f"sub_size_{size}.csv"
            pd.DataFrame({"uuid": uu, "style": pp}).to_csv(out_ps, index=False)
            print("Wrote per-size:", out_ps)

    # ---------- Final merge across sizes ----------
    all_uuid, all_preds = [], []
    for u in test_df['uuid'].astype(str).tolist():
        # CLS
        if acc_wsum_cls[u] > 0:
            logits_avg = acc_logits_sum[u] / max(acc_wsum_cls[u], 1e-6)
            p = np.exp(logits_avg - logits_avg.max()); p /= p.sum()
            topv = float(p.max()); top_idx = int(p.argmax())
        else:
            # in extreme case (all CLS excluded), fallback to UNK
            topv, top_idx = 0.0, 0

        # OOD
        if acc_wsum_ood[u] > 0:
            p_norm = acc_pood_norm_wsum[u] / max(acc_wsum_ood[u], 1e-6)
        else:
            p_norm = 0.0
        is_unk = (p_norm >= OOD_THR_NORM)
        if topv > UNK_TOPP_MAX: is_unk = False

        all_uuid.append(u)
        all_preds.append(UNK_NAME if is_unk else IDX2STYLE[top_idx])

    sub = pd.DataFrame({"uuid": all_uuid, "style": all_preds})
    Path(args.submission).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.submission, index=False)
    print("Wrote:", args.submission)

if __name__ == "__main__":
    main()
