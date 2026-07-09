"""Phase 4 — unsupervised anomaly detection & dimensionality reduction.

Two unsupervised fraud detectors, trained WITHOUT labels:
1. PCA reconstruction error — fit PCA on legitimate transactions only, then flag
   transactions whose reconstruction error is high.
2. Autoencoder reconstruction error — a small PyTorch autoencoder trained on
   legit-only; high reconstruction error => anomaly.

Both are scored by PR-AUC against the (held-out) labels, so they can be compared
to the supervised models on the same footing. Fitting on legit-only is the key
leakage-safe move: the detector never sees fraud during training.

Run:  python -m src.anomaly        (autoencoder needs torch)
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score

from src.seed import seed_everything
from src.data import get_data
from src.train import engineer, temporal_split, FEATURES


def _prep(train, test):
    imp = SimpleImputer(strategy="median").fit(train[FEATURES])
    sc = StandardScaler().fit(imp.transform(train[FEATURES]))   # TRAIN-ONLY fit
    return (sc.transform(imp.transform(train[FEATURES])).astype("float32"),
            sc.transform(imp.transform(test[FEATURES])).astype("float32"))


def pca_detector(Xtr_legit, Xte, n_components=3):
    """PCA fit on legit-only; anomaly score = reconstruction error."""
    pca = PCA(n_components=n_components, random_state=42).fit(Xtr_legit)
    recon = pca.inverse_transform(pca.transform(Xte))
    err = ((Xte - recon) ** 2).mean(axis=1)
    return err, pca.explained_variance_ratio_


def autoencoder_detector(Xtr_legit, Xte, epochs=25, bottleneck=2):
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    d = Xtr_legit.shape[1]
    net = nn.Sequential(
        nn.Linear(d, 8), nn.ReLU(), nn.Linear(8, bottleneck), nn.ReLU(),
        nn.Linear(bottleneck, 8), nn.ReLU(), nn.Linear(8, d),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.tensor(Xtr_legit)), batch_size=256, shuffle=True)
    net.train()
    for _ in range(epochs):
        for (xb,) in dl:
            opt.zero_grad(); loss_fn(net(xb), xb).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        recon = net(torch.tensor(Xte)).numpy()
    return ((Xte - recon) ** 2).mean(axis=1)


def main():
    seed_everything(42)
    df = engineer(get_data()[0])
    train, test = temporal_split(df)
    Xtr, Xte = _prep(train, test)
    ytr = train["is_fraud"].to_numpy(); yte = test["is_fraud"].to_numpy()
    Xtr_legit = Xtr[ytr == 0]                     # train on legit only (no labels/fraud seen)

    pca_err, evr = pca_detector(Xtr_legit, Xte)
    ae_err = autoencoder_detector(Xtr_legit, Xte)
    out = {
        "pca_explained_variance": [round(float(v), 3) for v in evr],
        "pca_recon_pr_auc": round(float(average_precision_score(yte, pca_err)), 3),
        "autoencoder_recon_pr_auc": round(float(average_precision_score(yte, ae_err)), 3),
        "baseline_pr_auc": round(float(yte.mean()), 3),
    }
    print(out)
    return out


if __name__ == "__main__":
    main()
