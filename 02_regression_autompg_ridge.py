"""
02 - Regression: Ridge (L2-regularised) Linear Regression on the Auto MPG dataset.

Original source: UCI Machine Learning Repository, dataset 9 (auto+mpg).
The script downloads a clean CSV mirror of the same data and caches it in ./data.

MATHEMATICS IMPLEMENTED HERE
----------------------------
Standardisation (feature transformation, lecture 7-8):
    z_j = (x_j - mu_j) / sigma_j
Needed because the ridge penalty is not scale invariant.

Ridge objective (MSE + L2 regularisation):
    L(w, b) = (1/N) ||y - Xw - b1||^2 + lambda ||w||^2

Gradients:
    dL/dw = -(2/N) X^T (y - Xw - b1) + 2 lambda w
    dL/db = -(2/N) 1^T (y - Xw - b1)

Setting dL/db = 0 gives b = mean(y) - mean(x)^T w, so after centring y and X the
intercept drops out. Setting dL/dw = 0 then gives the closed form

    w_hat = (X^T X + N lambda I)^{-1} X^T y                (centred X, y)

Why ridge:
  * X^T X + N lambda I is always invertible (lambda > 0), even when features are
    collinear - here weight, displacement and #cylinders are strongly correlated.
  * Shrinkage trades a little bias for a large variance reduction.
  * Bayesian reading: ridge is the MAP estimate with a Gaussian prior
    w ~ N(0, tau^2 I) and Gaussian noise, with lambda = sigma^2 / (N tau^2).

lambda is selected by k-fold cross-validation on the training split.

Metrics: MSE, RMSE, MAE, R^2.
"""

import argparse
import json
import os
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
RESDIR = os.path.join(HERE, "results")
DATADIR = os.path.join(HERE, "data")
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_autompg():
    os.makedirs(DATADIR, exist_ok=True)
    path = os.path.join(DATADIR, "auto_mpg.csv")
    if not os.path.exists(path):
        print(f"Downloading Auto MPG data from {URL} ...")
        urllib.request.urlretrieve(URL, path)
    df = pd.read_csv(path)
    n_raw = len(df)
    df = df.dropna().reset_index(drop=True)
    feats = ["cylinders", "displacement", "horsepower", "weight",
             "acceleration", "model_year"]
    X = df[feats].to_numpy(dtype=float)
    # one-hot encoding of the categorical 'origin' (drop first level)
    origin = pd.get_dummies(df["origin"], prefix="origin", drop_first=True)
    X = np.hstack([X, origin.to_numpy(dtype=float)])
    names = feats + list(origin.columns)
    y = df["mpg"].to_numpy(dtype=float)
    return X, y, names, n_raw


# ----------------------------------------------------------------------
# Core mathematics
# ----------------------------------------------------------------------
def standardise(X, mu=None, sd=None):
    if mu is None:
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


def ridge_fit(X, y, lam):
    """w = (X^T X + N lam I)^{-1} X^T y on centred data; b recovered afterwards."""
    N, p = X.shape
    xbar, ybar = X.mean(axis=0), y.mean()
    Xc, yc = X - xbar, y - ybar
    A = Xc.T @ Xc + N * lam * np.eye(p)
    w = np.linalg.solve(A, Xc.T @ yc)
    b = ybar - xbar @ w
    return w, b


def predict(X, w, b):
    return X @ w + b


def mse(y, p):
    return float(np.mean((y - p) ** 2))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def kfold_cv(X, y, lam, k=5, seed=0):
    """k-fold CV estimate of the test MSE for one lambda."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    errs = []
    for i in range(k):
        va = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        Xtr, mu, sd = standardise(X[tr])
        Xva, _, _ = standardise(X[va], mu, sd)
        w, b = ridge_fit(Xtr, y[tr], lam)
        errs.append(mse(y[va], predict(Xva, w, b)))
    return float(np.mean(errs))


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(RESDIR, exist_ok=True)

    X, y, names, n_raw = load_autompg()
    print("=" * 72)
    print("02 | RIDGE REGRESSION  -  Auto MPG dataset")
    print("=" * 72)
    print(f"Rows in raw file        : {n_raw}")
    print(f"Rows after dropping NaN : {X.shape[0]}   (missing horsepower values)")
    print(f"Features p = {X.shape[1]}: {', '.join(names)}")
    print(f"Target: mpg,  mean = {y.mean():.3f}, std = {y.std():.3f}")
    print()

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(y))
    ntr = int(0.8 * len(y))
    tr, te = idx[:ntr], idx[ntr:]
    X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]
    print(f"Train / test split: {len(tr)} / {len(te)}")

    # collinearity diagnostic on standardised features
    Z_tr, mu, sd = standardise(X_tr)
    Z_te, _, _ = standardise(X_te, mu, sd)
    G = Z_tr.T @ Z_tr
    print(f"Condition number of Z^T Z (lambda = 0) : {np.linalg.cond(G):.3e}")
    C = np.corrcoef(Z_tr, rowvar=False)
    print("Strongest feature correlations (|r| > 0.8):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(C[i, j]) > 0.8:
                print(f"   corr({names[i]}, {names[j]}) = {C[i, j]:+.3f}")
    print()

    # --- lambda selection by k-fold CV -----------------------------------
    lams = np.logspace(-6, 2, 41)
    cv = np.array([kfold_cv(X_tr, y_tr, l, k=args.folds, seed=args.seed) for l in lams])
    lam_star = float(lams[np.argmin(cv)])
    print("-" * 72)
    print(f"({args.folds}-fold cross-validation over lambda)")
    print("-" * 72)
    for l, e in zip(lams[::5], cv[::5]):
        print(f"   lambda = {l:10.2e}   CV MSE = {e:8.4f}")
    print(f"Selected lambda* = {lam_star:.4e}   (CV MSE = {cv.min():.4f})")
    print()

    # --- final fit --------------------------------------------------------
    w_ols, b_ols = ridge_fit(Z_tr, y_tr, 0.0)
    w, b = ridge_fit(Z_tr, y_tr, lam_star)
    print("-" * 72)
    print("Coefficients on standardised features (mpg per 1 s.d. of the feature)")
    print("-" * 72)
    print(f"{'feature':>16}{'OLS':>12}{'ridge':>12}")
    for n, a, c in zip(names, w_ols, w):
        print(f"{n:>16}{a:12.4f}{c:12.4f}")
    print(f"{'intercept':>16}{b_ols:12.4f}{b:12.4f}")
    print(f"||w||_2 : OLS = {np.linalg.norm(w_ols):.4f}, ridge = {np.linalg.norm(w):.4f}  (shrinkage)")
    print()

    out = {}
    for tag, (ww, bb) in {"ols": (w_ols, b_ols), "ridge": (w, b)}.items():
        p_tr, p_te = predict(Z_tr, ww, bb), predict(Z_te, ww, bb)
        out[tag] = {
            "train_mse": mse(y_tr, p_tr), "test_mse": mse(y_te, p_te),
            "test_rmse": float(np.sqrt(mse(y_te, p_te))),
            "test_mae": mae(y_te, p_te),
            "train_r2": r2(y_tr, p_tr), "test_r2": r2(y_te, p_te),
        }
    print("-" * 72)
    print("PERFORMANCE")
    print("-" * 72)
    print(f"{'model':<8}{'train MSE':>12}{'test MSE':>12}{'test RMSE':>12}{'test MAE':>11}{'test R^2':>10}")
    for tag in ("ols", "ridge"):
        m = out[tag]
        print(f"{tag:<8}{m['train_mse']:12.4f}{m['test_mse']:12.4f}"
              f"{m['test_rmse']:12.4f}{m['test_mae']:11.4f}{m['test_r2']:10.4f}")
    print()

    # --- plots ------------------------------------------------------------
    path = np.array([ridge_fit(Z_tr, y_tr, l)[0] for l in lams])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogx(lams, cv, "o-", ms=3)
    ax[0].axvline(lam_star, color="crimson", ls="--", lw=1,
                  label=rf"$\lambda^*={lam_star:.2e}$")
    ax[0].set_xlabel(r"$\lambda$")
    ax[0].set_ylabel(f"{args.folds}-fold CV MSE")
    ax[0].set_title("Cross-validated model selection")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    for j, n in enumerate(names):
        ax[1].semilogx(lams, path[:, j], label=n, lw=1.4)
    ax[1].axvline(lam_star, color="crimson", ls="--", lw=1)
    ax[1].set_xlabel(r"$\lambda$")
    ax[1].set_ylabel(r"$w_j(\lambda)$")
    ax[1].set_title("Ridge coefficient paths (shrinkage)")
    ax[1].legend(fontsize=6, ncol=2)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    f1 = os.path.join(FIGDIR, "02_autompg_ridge_path.pdf")
    fig.savefig(f1)
    plt.close(fig)

    p_te = predict(Z_te, w, b)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(y_te, p_te, s=20, alpha=0.75, edgecolor="none")
    lims = [min(y_te.min(), p_te.min()), max(y_te.max(), p_te.max())]
    ax[0].plot(lims, lims, "k--", lw=1)
    ax[0].set_xlabel("true mpg")
    ax[0].set_ylabel("predicted mpg")
    ax[0].set_title(f"Test set, $R^2$ = {out['ridge']['test_r2']:.3f}")
    ax[0].grid(alpha=0.3)
    ax[1].hist(y_te - p_te, bins=20, edgecolor="k", alpha=0.8)
    ax[1].set_xlabel(r"residual $y-\hat{y}$ (mpg)")
    ax[1].set_ylabel("count")
    ax[1].set_title("Residual distribution")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    f2 = os.path.join(FIGDIR, "02_autompg_fit.pdf")
    fig.savefig(f2)
    plt.close(fig)

    with open(os.path.join(RESDIR, "02_autompg.json"), "w") as fh:
        json.dump({"metrics": out, "lambda_star": lam_star,
                   "cv_mse_min": float(cv.min()),
                   "coefficients_ridge": {n: float(v) for n, v in zip(names, w)},
                   "coefficients_ols": {n: float(v) for n, v in zip(names, w_ols)},
                   "cond_ZtZ": float(np.linalg.cond(G))}, fh, indent=2)
    print(f"Figures saved: {f1}")
    print(f"               {f2}")
    print(f"Results saved: {os.path.join(RESDIR, '02_autompg.json')}")


if __name__ == "__main__":
    main()
