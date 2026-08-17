"""
run_experiments.py

Runs the full experiment suite and saves everything to results.json.

WHAT IT RUNS
  A) Classical (Gabor + Fourier + LBP) at 25%, 50%, 100% of training data
  B) CNN from scratch, 3 seeds, at 100% data  -> mean +/- std
  C) CNN from scratch at 25%, 50% data        -> data-efficiency curve

KEY SPEEDUP
  Classical features are extracted ONCE and cached to features_cache.npz.
  After that every classical experiment is just an SVM fit (seconds, not minutes).

RESUMABLE
  Every result is written to results.json as soon as it finishes. If the script
  crashes or you close the terminal, just run it again: it skips whatever is
  already done.

USAGE
  python run_experiments.py              # run everything still outstanding
  python run_experiments.py --list       # show what's done and what's pending
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

from eval_utils import set_seed
from Tex_class_classical import build_gabor_bank, extract_features

RESULTS_FILE = "results.json"
CACHE_FILE = "features_cache.npz"
PARTITION = 1
SEEDS = (1, 2, 3)
FRACTIONS = (0.25, 0.50, 1.00)
CNN_EPOCHS = 60


# =========================================================
# results bookkeeping
# =========================================================
def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_result(key, value):
    r = load_results()
    r[key] = value
    with open(RESULTS_FILE, "w") as f:
        json.dump(r, f, indent=2)
    print(f"  [saved] {key}")


def subset_indices(n, fraction, seed=0):
    """Deterministic random subset of indices, so every model sees the same subset."""
    rng = np.random.default_rng(seed)
    return rng.permutation(n)[: int(round(n * fraction))]


# 
# A) CLASSICAL
# 
def build_feature_cache():
    """Extract classical features once for train and test; save to disk."""
    if os.path.exists(CACHE_FILE):
        print("Feature cache found, loading...")
        d = np.load(CACHE_FILE)
        return d["Xtr"], d["ytr"], d["Xte"], d["yte"]

    print("Extracting classical features (one time only, ~20 min)...")
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    tr = torchvision.datasets.DTD(root="./data", split="train", partition=PARTITION,
                                  transform=tf, download=True)
    te = torchvision.datasets.DTD(root="./data", split="test", partition=PARTITION,
                                  transform=tf, download=True)
    bank = build_gabor_bank()

    def run(ds, name):
        X, y = [], []
        t0 = time.time()
        for i in range(len(ds)):
            img, lab = ds[i]
            X.append(extract_features(img, bank))
            y.append(lab)
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                print(f"  {name}: {i+1}/{len(ds)}  ({el/60:.1f} min elapsed)")
        return np.array(X), np.array(y)

    Xtr, ytr = run(tr, "train")
    Xte, yte = run(te, "test")
    np.savez_compressed(CACHE_FILE, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    print(f"Cached features to {CACHE_FILE}  (train {Xtr.shape}, test {Xte.shape})")
    return Xtr, ytr, Xte, yte


def run_classical(fraction, Xtr, ytr, Xte, yte):
    idx = subset_indices(len(Xtr), fraction)
    Xs, ys = Xtr[idx], ytr[idx]

    scaler = StandardScaler().fit(Xs)
    Xs_s, Xte_s = scaler.transform(Xs), scaler.transform(Xte)

    grid = GridSearchCV(
        SVC(kernel="rbf"),
        {"C": [1, 10, 100, 1000], "gamma": ["scale", 0.01, 0.001]},
        cv=3, n_jobs=-1,
    )
    grid.fit(Xs_s, ys)
    pred = grid.predict(Xte_s)
    return {
        "top1": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro")),
        "n_train": int(len(idx)),
        "best_params": {k: str(v) for k, v in grid.best_params_.items()},
    }


# 
# B/C) CNN
# 
class BalancedTextureCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(32); self.relu1 = nn.ReLU(); self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64); self.relu2 = nn.ReLU(); self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128); self.relu3 = nn.ReLU(); self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256); self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.2); self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu4(self.fc1(x)))
        return self.fc2(x)


def cnn_loaders(fraction):
    train_tf = transforms.Compose([
        transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    eval_tf = transforms.Compose([
        transforms.Resize((64, 64)), transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)])

    tr = torchvision.datasets.DTD(root="./data", split="train", partition=PARTITION,
                                  transform=train_tf, download=True)
    va = torchvision.datasets.DTD(root="./data", split="val", partition=PARTITION,
                                  transform=eval_tf, download=True)
    te = torchvision.datasets.DTD(root="./data", split="test", partition=PARTITION,
                                  transform=eval_tf, download=True)

    if fraction < 1.0:
        # SAME indices as the classical subset, so the comparison is fair
        tr = Subset(tr, subset_indices(len(tr), fraction).tolist())

    return (DataLoader(tr, 32, shuffle=True),
            DataLoader(va, 32, shuffle=False),
            DataLoader(te, 32, shuffle=False),
            te.classes)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def run_cnn(seed, fraction, device, save_weights=None):
    set_seed(seed)
    tr, va, te, classes = cnn_loaders(fraction)
    model = BalancedTextureCNN(len(classes)).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

    best_state, best_val, best_epoch = None, -1.0, -1
    t0 = time.time()
    for epoch in range(CNN_EPOCHS):
        model.train()
        run_loss = 0.0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item()

        yv, pv = predict(model, va, device)
        val_acc = float((yv == pv).mean())
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    epoch {epoch+1}/{CNN_EPOCHS}  loss {run_loss/len(tr):.3f}"
                  f"  val {val_acc*100:.1f}%  ({(time.time()-t0)/60:.1f} min)")

    model.load_state_dict(best_state)
    if save_weights:
        torch.save(model.state_dict(), save_weights)

    yt, pt = predict(model, te, device)
    return {
        "top1": float(accuracy_score(yt, pt)),
        "macro_f1": float(f1_score(yt, pt, average="macro")),
        "best_epoch": best_epoch,
        "best_val": best_val,
        "final_train_loss": run_loss / len(tr),
        "minutes": (time.time() - t0) / 60,
    }


# =========================================================
# main
# =========================================================
def planned_jobs():
    jobs = [("classical", f, None) for f in FRACTIONS]
    jobs += [("cnn", 1.00, s) for s in SEEDS]          # seeds at full data
    jobs += [("cnn", f, SEEDS[0]) for f in (0.25, 0.50)]  # efficiency curve
    return jobs


def key_for(kind, frac, seed):
    return f"{kind}_frac{int(frac*100)}" + (f"_seed{seed}" if seed is not None else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show status and exit")
    args = ap.parse_args()

    done = load_results()
    jobs = planned_jobs()

    if args.list:
        print(f"{'job':32s} status")
        for kind, frac, seed in jobs:
            k = key_for(kind, frac, seed)
            print(f"{k:32s} {'DONE' if k in done else 'pending'}")
        return

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count())
        print(f"Using {torch.get_num_threads()} CPU threads")

    # classical features once
    need_classical = any(key_for(k, f, s) not in done for k, f, s in jobs if k == "classical")
    if need_classical:
        Xtr, ytr, Xte, yte = build_feature_cache()

    for kind, frac, seed in jobs:
        k = key_for(kind, frac, seed)
        if k in done:
            print(f"\n== {k}: already done, skipping ==")
            continue

        print(f"\n== {k}: running ==")
        t0 = time.time()
        if kind == "classical":
            res = run_classical(frac, Xtr, ytr, Xte, yte)
        else:
            weights = "cnn_best.pth" if (frac == 1.00 and seed == SEEDS[0]) else None
            res = run_cnn(seed, frac, device, save_weights=weights)
        res["elapsed_min"] = (time.time() - t0) / 60

        print(f"  top-1 {res['top1']*100:.2f}%   macro-F1 {res['macro_f1']*100:.2f}%"
              f"   ({res['elapsed_min']:.1f} min)")
        save_result(k, res)

    print("\n\n================ SUMMARY ================")
    r = load_results()

    print("\nClassical (Gabor + Fourier + LBP):")
    for f in FRACTIONS:
        k = key_for("classical", f, None)
        if k in r:
            print(f"  {int(f*100):3d}% data: top-1 {r[k]['top1']*100:5.2f}%  "
                  f"macro-F1 {r[k]['macro_f1']*100:5.2f}%  (n={r[k]['n_train']})")

    print("\nCNN from scratch, 100% data, across seeds:")
    accs = [r[key_for('cnn', 1.00, s)]["top1"] * 100 for s in SEEDS
            if key_for("cnn", 1.00, s) in r]
    f1s = [r[key_for('cnn', 1.00, s)]["macro_f1"] * 100 for s in SEEDS
           if key_for("cnn", 1.00, s) in r]
    for s in SEEDS:
        k = key_for("cnn", 1.00, s)
        if k in r:
            print(f"  seed {s}: top-1 {r[k]['top1']*100:5.2f}%  "
                  f"macro-F1 {r[k]['macro_f1']*100:5.2f}%  (best epoch {r[k]['best_epoch']})")
    if len(accs) > 1:
        print(f"  --> top-1 {np.mean(accs):.2f} +/- {np.std(accs):.2f}%   "
              f"macro-F1 {np.mean(f1s):.2f} +/- {np.std(f1s):.2f}%")

    print("\nData-efficiency curve (top-1 %):")
    print(f"  {'fraction':>10s} {'classical':>11s} {'CNN':>8s}")
    for f in FRACTIONS:
        ck = key_for("classical", f, None)
        nk = key_for("cnn", f, SEEDS[0])
        c = f"{r[ck]['top1']*100:.2f}" if ck in r else "-"
        n = f"{r[nk]['top1']*100:.2f}" if nk in r else "-"
        print(f"  {int(f*100):9d}% {c:>11s} {n:>8s}")

    print("\nAll results in results.json")


if __name__ == "__main__":
    main()
