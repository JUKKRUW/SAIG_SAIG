
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---------------------------------------
  --opt {adamw,sgd,radam}
  --sched {cosine}
  --warmup_epochs N
  --mixup A, --cutmix A
  --label_smoothing S
  --ema {0,1} --ema_decay D
"""

import os, json, random, argparse
from pathlib import Path
import time
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import timm
from tqdm import tqdm

# ------------------------------
# Labels
# ------------------------------

STYLES = [
    'Ink scenery','comic','cyberpunk','futuristic UI','lowpoly',
    'oil painting','pixel','realistic','steampunk','water color'
]
STYLE2IDX = {s:i for i,s in enumerate(STYLES)}
IDX2STYLE = {i:s for s,i in STYLE2IDX.items()}
UNK_NAME = 'UNK'

# ------------------------------
# Utils
# ------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = (1.0 - self.decay) * p.data + self.decay * self.shadow[name]

    def apply_shadow(self, model):
        self.original = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.original[name] = p.data.clone()
                p.data = self.shadow[name]

    def restore(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.original:
                p.data = self.original[name]
        self.original = {}

def energy_from_logits(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    return -T * torch.logsumexp(logits / T, dim=1)

# ------------------------------
# Dataset
# ------------------------------

class StyleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform, has_labels=True,
                 ext_prefer=('.png','.jpg','.jpeg','.webp','.bmp')):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_labels = has_labels
        self.exts = ext_prefer

    def __len__(self):
        return len(self.df)

    def _path_for_uuid(self, uuid: str):
        for ext in self.exts:
            p = os.path.join(self.img_dir, f"{uuid}{ext}")
            if os.path.exists(p): return p
        p = os.path.join(self.img_dir, uuid)
        if os.path.exists(p): return p
        raise FileNotFoundError(f"Image not found: {uuid} in {self.img_dir}")

    def __getitem__(self, i):
        row = self.df.iloc[i]
        uuid = str(row['uuid'])
        img_path = self._path_for_uuid(uuid)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(image=img)['image']
        if self.has_labels:
            return img, STYLE2IDX[row['style']]
        else:
            return img, uuid

# ------------------------------
# Transforms (Albumentations v2 compatible)
# ------------------------------

def build_transforms(size: int = 384):
    tf_train = A.Compose([
        A.Resize(size, size, interpolation=cv2.INTER_AREA),
        A.Affine(translate_percent={"x":(-0.05,0.05),"y":(-0.05,0.05)},
                 scale=(0.95,1.05), rotate=(-10,10),
                 interpolation=cv2.INTER_LINEAR, p=0.30),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.30),
        A.ImageCompression(quality_range=(80,98), p=0.20),
        A.GaussNoise(std_range=(0.02,0.06), p=0.15),
        A.CoarseDropout(num_holes_range=(1, 8),
                        hole_height_range=(0.03,0.12),
                        hole_width_range=(0.03,0.12),
                        fill=0, p=0.20),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])
    tf_eval = A.Compose([
        A.Resize(size, size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])
    return tf_train, tf_eval

# ------------------------------
# Model
# ------------------------------

def build_model(model_name: str = 'tf_efficientnet_b4_ns', num_classes: int = 10, pretrained=True):
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

# ------------------------------
# Helpers
# ------------------------------

def make_weighted_sampler(labels: np.ndarray):
    classes, counts = np.unique(labels, return_counts=True)
    class_weights = {c: 1.0 / cnt for c, cnt in zip(classes, counts)}
    weights = np.array([class_weights[y] for y in labels], dtype=np.float32)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler

@torch.no_grad()
def evaluate(model, loader, device, tta=True, show_pbar=True, desc="Valid"):
    model.eval()
    all_logits, all_labels = [], []
    it = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True) if show_pbar else loader
    for imgs, labels in it:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        if tta:
            logits = (logits + model(torch.flip(imgs, dims=[3]))) / 2.0
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    preds = all_logits.argmax(1)
    f1_macro = f1_score(all_labels, preds, average='macro')
    report = classification_report(all_labels, preds, target_names=STYLES, digits=4)
    return all_logits, all_labels, f1_macro, report

def temperature_scaling(logits: np.ndarray, labels: np.ndarray, init_T: float = 1.0, max_iter: int = 50):
    T = torch.tensor([init_T], requires_grad=True)
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter)
    def _nll():
        opt.zero_grad()
        loss = F.cross_entropy(logits_t / T.clamp(min=1e-3), labels_t)
        loss.backward()
        return loss
    opt.step(_nll)
    return float(T.clamp(min=1e-3, max=100.0).item())

def calibrate_energy_threshold(logits: np.ndarray, labels: np.ndarray, T: float, seed=42):
    rng = np.random.default_rng(seed)
    x = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    idxs_by_cls = [torch.where(y==c)[0] for c in range(len(STYLES))]
    n_val = len(y)
    pseudo = []
    for _ in range(n_val):
        c1, c2 = rng.choice(len(STYLES), size=2, replace=False)
        i1 = int(rng.choice(idxs_by_cls[c1]))
        i2 = int(rng.choice(idxs_by_cls[c2]))
        lmix = 0.5*x[i1] + 0.5*x[i2]
        pseudo.append(lmix.unsqueeze(0))
    pseudo = torch.cat(pseudo, dim=0)
    with torch.no_grad():
        e_in  = energy_from_logits(x, T=T).cpu().numpy()
        e_out = energy_from_logits(pseudo, T=T).cpu().numpy()
    xs = np.concatenate([e_in, e_out])
    ys = np.concatenate([np.zeros_like(e_in), np.ones_like(e_out)])
    thr_candidates = np.quantile(xs, np.linspace(0.05, 0.95, 91))
    from sklearn.metrics import f1_score
    best_f1, best_thr = -1.0, float(xs.mean())
    for thr in thr_candidates:
        pred = (xs > thr).astype(int)
        f1 = f1_score(ys, pred, average='binary')
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1

# ===== Mixup / CutMix helpers =====

def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()

def apply_mixup(x: torch.Tensor, y_idx: torch.Tensor, num_classes: int, alpha: float):
    if alpha <= 0 or x.size(0) < 2:
        return x, None
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[perm]
    y1 = F.one_hot(y_idx, num_classes=num_classes).float()
    y2 = y1[perm]
    return x_mix, (y1, y2, lam)

def rand_bbox(W, H, lam):
    # CutMix bbox
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def apply_cutmix(x: torch.Tensor, y_idx: torch.Tensor, num_classes: int, alpha: float):
    if alpha <= 0 or x.size(0) < 2:
        return x, None
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    x1, y1, x2, y2 = rand_bbox(x.size(3), x.size(2), lam)
    x_cut = x.clone()
    x_cut[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    # adjust lam to the pixel ratio
    lam_adj = 1.0 - ((x2 - x1) * (y2 - y1) / (x.size(2) * x.size(3)))
    yA = F.one_hot(y_idx, num_classes=num_classes).float()
    yB = yA[perm]
    return x_cut, (yA, yB, lam_adj)

# ------------------------------
# Train a single fold
# ------------------------------

def make_optimizer(params, name: str, lr: float, wd: float):
    n = name.lower()
    if n == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if n == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    if n == 'radam':
        return torch.optim.RAdam(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown --opt {name}")

def make_scheduler(opt, name: str, epochs: int, warmup_epochs: int, lr: float):
    n = name.lower()
    if n == 'cosine':
        if warmup_epochs > 0:
            warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs)
            cos  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warmup_epochs), eta_min=lr*0.01)
            return torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warm, cos], milestones=[warmup_epochs])
        else:
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    raise ValueError(f"Unknown --sched {name}")

def train_one_fold(model_name, fold, train_df, val_df, img_dir, size, epochs, batch, lr, wd,
                   device, out_dir,
                   mixup_alpha=0.0, cutmix_alpha=0.0,
                   label_smoothing=0.05,
                   opt_name='adamw', sched_name='cosine', warmup_epochs=0,
                   ema_enabled=True, ema_decay=0.999):
    tf_train, tf_eval = build_transforms(size=size)
    ds_tr = StyleDataset(train_df, img_dir, tf_train, has_labels=True)
    ds_va = StyleDataset(val_df,   img_dir, tf_eval,  has_labels=True)

    labels_tr = train_df['style'].map(STYLE2IDX).values

    # ✅ ลด workers ให้พอดีกับ Colab
    num_workers = min(2, (os.cpu_count() or 2))

    # ถ้าอยากคุม sampling ให้สมดุลคงเดิมก็ใช้ sampler เช่นเดิม
    sampler = make_weighted_sampler(labels_tr)
    loader_tr = DataLoader(ds_tr, batch_size=batch, sampler=sampler,
                           num_workers=num_workers, pin_memory=(device.type=='cuda'),
                           persistent_workers=(num_workers>0))
    loader_va = DataLoader(ds_va, batch_size=batch*2, shuffle=False,
                           num_workers=num_workers, pin_memory=(device.type=='cuda'),
                           persistent_workers=(num_workers>0))

    model = build_model(model_name, num_classes=len(STYLES), pretrained=True).to(device)
    optimizer = make_optimizer(model.parameters(), opt_name, lr, wd)
    scheduler = make_scheduler(optimizer, sched_name, epochs, warmup_epochs, lr)
    scaler = GradScaler(device.type if device.type!='cpu' else 'cpu', enabled=(device.type=='cuda'))
    ema = EMA(model, decay=ema_decay) if ema_enabled else None

    best_f1 = -1.0
    best_path = Path(out_dir) / f"{model_name}_fold{fold}.pt"
    criterion = nn.CrossEntropyLoss(label_smoothing=(0.0 if (mixup_alpha>0 or cutmix_alpha>0) else label_smoothing))

    epoch_times = []

    def _fmt(sec):
        m, s = divmod(int(sec), 60); h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    for epoch in range(1, epochs+1):
        model.train()
        t0 = time.time()
        pbar = tqdm(loader_tr, desc=f"[{model_name}][Fold {fold}] Epoch {epoch}/{epochs}", leave=False)
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device.type if device.type!='cpu' else 'cpu', enabled=(device.type=='cuda')):
                if mixup_alpha>0 or cutmix_alpha>0:
                    use_mix = (mixup_alpha>0 and cutmix_alpha>0 and torch.rand(1, device=imgs.device).item() < 0.5)
                    if use_mix or (mixup_alpha>0 and cutmix_alpha==0):
                        imgs_aug, mix = apply_mixup(imgs, labels, num_classes=len(STYLES), alpha=mixup_alpha)
                    else:
                        imgs_aug, mix = apply_cutmix(imgs, labels, num_classes=len(STYLES), alpha=cutmix_alpha)
                    if mix is not None:
                        logits = model(imgs_aug)
                        y1, y2, lam = mix
                        loss = lam * soft_cross_entropy(logits, y1) + (1.0 - lam) * soft_cross_entropy(logits, y2)
                    else:
                        logits = model(imgs)
                        loss = criterion(logits, labels)
                else:
                    logits = model(imgs)
                    loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # ✅ Eval “ในลูป” ทุก epoch
        t1 = time.time()
        if ema is not None:
            ema.apply_shadow(model)
        _, _, f1_macro, _ = evaluate(
            model, loader_va, device, tta=True, show_pbar=False,
            desc=f"[{model_name}][Fold {fold}] Valid {epoch}/{epochs}"
        )
        if ema is not None:
            ema.restore(model)
        t2 = time.time()

        # เวลาต่อ epoch + ETA
        train_sec, valid_sec, total_sec = (t1 - t0), (t2 - t1), (t2 - t0)
        epoch_times.append(total_sec)
        avg_sec = np.mean(epoch_times[-3:]) if len(epoch_times) >= 1 else total_sec
        remain = (epochs - epoch) * avg_sec
        print(f"Fold {fold} — Epoch {epoch}: F1_macro={f1_macro:.4f} (best {max(best_f1, f1_macro):.4f}) | "
              f"Train {_fmt(train_sec)} + Val {_fmt(valid_sec)} = {_fmt(total_sec)} | ETA ~ {_fmt(remain)}")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            torch.save(model.state_dict(), best_path)

    # Final eval + calibration on best (เหมือนเดิม)
    model.load_state_dict(torch.load(best_path, map_location='cpu'))
    model.to(device)
    logits_val, labels_val, f1_macro, report = evaluate(model, loader_va, device, tta=True)
    print("Validation report:\n", report)

    T_opt = temperature_scaling(logits_val, labels_val, init_T=1.0, max_iter=50)
    thr_energy, f1_ood = calibrate_energy_threshold(logits_val, labels_val, T=T_opt, seed=42)
    print(f"[Fold {fold}] Calibrated T={T_opt:.3f}, EnergyThr={thr_energy:.4f}, OOD-F1(pseudo)={f1_ood:.4f}")

    meta = {
        "model_name": model_name, "fold": fold, "img_size": size,
        "best_f1_macro": float(best_f1), "T": float(T_opt), "energy_thr": float(thr_energy)
    }
    with open(Path(out_dir)/f"{model_name}_fold{fold}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return best_path, meta

# ------------------------------
# Inference (simple energy-UNK)
# ------------------------------

@torch.no_grad()
def predict_ensemble(models_info, test_df, img_dir, device, batch=64, tta=True,
                     ms_scales=(1.0,), unk_guard=0.98, thr_shift=0.0, unk_margin=0.15):
    # group by input size
    by_size = {}
    for wpath, meta in models_info:
        by_size.setdefault(meta['img_size'], []).append((wpath, meta))

    preds, uuids = [], []
    num_workers = 4 if os.name != 'nt' else 4
    ms_scales = tuple(float(s) for s in ms_scales)

    def logits_with_tta(model, imgs):
        outs = []
        for sc in ms_scales:
            if sc == 1.0:
                x = imgs
            else:
                H, W = imgs.shape[2], imgs.shape[3]
                x = F.interpolate(imgs, size=(int(round(H*sc)), int(round(W*sc))),
                                  mode='bilinear', align_corners=False)
            o = model(x)
            if tta:
                o = (o + model(torch.flip(x, dims=[3]))) / 2.0
            outs.append(o)
        return sum(outs) / len(outs)

    for size, infos in by_size.items():
        tf_eval = A.Compose([
            A.Resize(size, size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2()
        ])
        ds = StyleDataset(test_df, img_dir, tf_eval, has_labels=False)
        ld = DataLoader(ds, batch_size=batch, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type=='cuda'))

        models, weights, Ts, Es = [], [], [], []
        for wpath, meta in infos:
            m = build_model(meta['model_name'], num_classes=len(STYLES), pretrained=False)
            state = torch.load(wpath, map_location='cpu')
            m.load_state_dict(state, strict=True)
            m.to(device).eval()
            models.append(m)
            w = float(max(1e-6, meta.get('best_f1_macro', 1.0)))
            weights.append(w)
            Ts.append(float(meta['T']))
            Es.append(float(meta['energy_thr']))

        w_sum = float(sum(weights))
    
        E_thr = (sum(w*e for w,e in zip(weights, Es)) / max(w_sum, 1e-6)) + float(thr_shift)

        for imgs, uuid in tqdm(ld, desc=f"Infer@{size}"):
            imgs = imgs.to(device, non_blocking=True)

            
            wlogits_cal = None
            
            E_wsum = None

            for m, w, Ti in zip(models, weights, Ts):
                l = logits_with_tta(m, imgs)            
                
                lc = l / max(Ti, 1e-6)
                wlogits_cal = (lc * w) if wlogits_cal is None else (wlogits_cal + lc * w)

                
                Ei = energy_from_logits(l, T=Ti)        
                Ei = Ei * w
                E_wsum = Ei if E_wsum is None else (E_wsum + Ei)

            logits_avg_cal = wlogits_cal / max(w_sum, 1e-6)   
            E_avg = E_wsum / max(w_sum, 1e-6)                  

            probs = F.softmax(logits_avg_cal, dim=1)          
            top_p, top_idx = probs.max(dim=1)
            # margin guard
            top2 = torch.topk(probs, k=2, dim=1).values
            margin = (top2[:,0] - top2[:,1])

            unk_mask = (E_avg > E_thr) & (top_p <= unk_guard) & (margin <= unk_margin)

            for i in range(len(uuid)):
                preds.append(UNK_NAME if bool(unk_mask[i].item()) else IDX2STYLE[int(top_idx[i].item())])
            uuids.extend(list(uuid))

        for m in models:
            del m
        torch.cuda.empty_cache()

    return uuids, preds

# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Open-Set Art Style — Pro (CLI-Extended)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # TRAIN
    p_tr = sub.add_parser("train", help="Train with KFold and save models + calibration")
    p_tr.add_argument("--train_csv", type=str, required=True)
    p_tr.add_argument("--img_dir", type=str, required=True)
    p_tr.add_argument("--out_dir", type=str, required=True)
    p_tr.add_argument("--models", type=str, nargs="+", default=["tf_efficientnet_b4_ns","convnextv2_tiny.fcmae_ft_in22k_in1k"])
    p_tr.add_argument("--sizes", type=int, nargs="+", default=[380, 224])
    p_tr.add_argument("--folds", type=int, default=5)
    p_tr.add_argument("--epochs", type=int, default=12)
    p_tr.add_argument("--batch", type=int, default=32)
    p_tr.add_argument("--lr", type=float, default=3e-4)
    p_tr.add_argument("--wd", type=float, default=0.02)
    p_tr.add_argument("--seed", type=int, default=42)
    # New training knobs
    p_tr.add_argument("--opt", type=str, default="adamw", choices=["adamw","sgd","radam"])
    p_tr.add_argument("--sched", type=str, default="cosine", choices=["cosine"])
    p_tr.add_argument("--warmup_epochs", type=int, default=0)
    p_tr.add_argument("--mixup", type=float, default=0.0, help="MixUp alpha (0=disable)")
    p_tr.add_argument("--cutmix", type=float, default=0.0, help="CutMix alpha (0=disable)")
    p_tr.add_argument("--label_smoothing", type=float, default=0.05)
    p_tr.add_argument("--ema", type=int, default=1, help="Use EMA (1=yes,0=no)")
    p_tr.add_argument("--ema_decay", type=float, default=0.999)

    # PREDICT
    p_pr = sub.add_parser("predict", help="Predict with ensemble + OOD to UNK, output submission CSV")
    p_pr.add_argument("--test_csv", type=str, required=True)
    p_pr.add_argument("--img_dir", type=str, required=True)
    p_pr.add_argument("--out_dir", type=str, required=True)
    p_pr.add_argument("--models", type=str, nargs="+", default=["tf_efficientnet_b4_ns","convnextv2_tiny.fcmae_ft_in22k_in1k"])
    p_pr.add_argument("--submission", type=str, required=True)
    p_pr.add_argument("--batch", type=int, default=64)
    p_pr.add_argument("--seed", type=int, default=42)
    p_pr.add_argument("--unk_guard", type=float, default=0.98)
    p_pr.add_argument("--ms_scales", type=str, default="1.0")   # เช่น "1.0,1.12"
    p_pr.add_argument("--thr_shift", type=float, default=0.00)
    p_pr.add_argument("--unk_margin", type=float, default=0.15)


    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.cmd == "train":
        df = pd.read_csv(args.train_csv)
        assert set(['uuid','style']).issubset(df.columns), "train CSV must have columns: uuid, style"
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

        all_model_records = []
        # zip_longest-like: if one model with multiple sizes, pair by position; else if lengths differ, reuse last size
        from itertools import zip_longest
        last_size = args.sizes[-1]
        for i, model_name in enumerate(args.models):
            size = args.sizes[i] if i < len(args.sizes) else last_size
            print(f"=== Training model: {model_name} @ size {size} ===")
            for fold, (tr_idx, va_idx) in enumerate(skf.split(df['uuid'], df['style']), start=1):
                train_df = df.iloc[tr_idx].reset_index(drop=True)
                val_df   = df.iloc[va_idx].reset_index(drop=True)
                wpath, meta = train_one_fold(
                    model_name=model_name,
                    fold=fold,
                    train_df=train_df,
                    val_df=val_df,
                    img_dir=args.img_dir,
                    size=size,
                    epochs=args.epochs,
                    batch=args.batch,
                    lr=args.lr,
                    wd=args.wd,
                    device=device,
                    out_dir=args.out_dir,
                    mixup_alpha=args.mixup,
                    cutmix_alpha=args.cutmix,
                    label_smoothing=args.label_smoothing,
                    opt_name=args.opt, sched_name=args.sched, warmup_epochs=args.warmup_epochs,
                    ema_enabled=bool(args.ema), ema_decay=args.ema_decay
                )
                all_model_records.append((str(wpath), meta))

        with open(Path(args.out_dir)/"models_registry.json", "w") as f:
            json.dump([{"weight_path":w, "meta":m} for w,m in all_model_records], f, indent=2)
        print("Saved model registry:", Path(args.out_dir)/"models_registry.json")

    elif args.cmd == "predict":
        test_df = pd.read_csv(args.test_csv)
        assert 'uuid' in test_df.columns, "test CSV must have column: uuid"

        reg_path = Path(args.out_dir) / "models_registry.json"
        assert reg_path.exists(), f"Model registry not found: {reg_path}. Run train first."
        registry = json.load(open(reg_path, "r"))
        reg_dir = reg_path.parent

        def _resolve_weight_path(wp: str) -> Path:
            p = Path(wp)

            
            if p.is_absolute() and p.exists():
                return p.resolve()

           
            candidates = [
                reg_dir / p,            
                p,                      
                reg_dir.parent / p,     
            ]

            
            if p.suffix and not p.exists():
                candidates += list(reg_dir.rglob(p.name))

            for c in candidates:
                if c.exists():
                    return c.resolve()

            
            tried = "\n  - " + "\n  - ".join(str(c) for c in candidates)
            raise FileNotFoundError(f"Weight file not found for entry '{wp}'. Tried:{tried}")

        models_info = []
        for item in registry:
            meta = item["meta"]
            if meta["model_name"] not in args.models:
                continue
            wpath = _resolve_weight_path(item["weight_path"])
            models_info.append((str(wpath), meta))

        assert len(models_info) > 0, "No matching models found in registry for --models"

        uuids, preds = predict_ensemble(
    models_info, test_df, args.img_dir, device,
    batch=args.batch, tta=True,
    ms_scales=tuple(s.strip() for s in args.ms_scales.split(",")),
    unk_guard=args.unk_guard,
    thr_shift=args.thr_shift,
    unk_margin=args.unk_margin
)


        sub = pd.DataFrame({"uuid": uuids, "style": preds})
        Path(args.submission).parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(args.submission, index=False)
        print("Wrote submission:", args.submission)

if __name__ == "__main__":
    main()
