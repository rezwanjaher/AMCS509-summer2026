"""
02 - Regression: Ridge Regression (linear regression with an L2 penalty) on the
     Auto MPG dataset (UCI). The script downloads a clean CSV copy of the data
     and caches it in ./data.

----------------------------------------------------------------------
WHY RIDGE HERE
----------------------------------------------------------------------
The predictors of fuel efficiency are strongly related to one another: heavier
cars have bigger engines and more cylinders. When columns of X are nearly
proportional, X^T X is close to singular, the plain least-squares solution
becomes unstable, and individual coefficients can be huge and of opposite sign.
Adding an L2 penalty fixes this.

----------------------------------------------------------------------
STANDARDISATION
----------------------------------------------------------------------
    z_j = (x_j - mean_j) / std_j
computed on the TRAINING data only. Needed because the penalty below depends
on the size of the weights, which depends on the units of the features.

----------------------------------------------------------------------
THE MODEL AND THE LOSS
----------------------------------------------------------------------
    y_hat = w^T z + b

    L(w, b) = (1/N) ||y - Zw - b||^2 + lambda ||w||^2

The first term is the usual mean squared error; the second shrinks the weights
towards zero. lambda = 0 gives ordinary least squares.

----------------------------------------------------------------------
THE CLOSED-FORM SOLUTION
----------------------------------------------------------------------
Differentiating with respect to b and setting it to zero gives
b = mean(y) - mean(z)^T w, so if we CENTRE y and Z first, the intercept
disappears from the problem. Differentiating with respect to w:

    dL/dw = -(2/N) Z^T (y - Zw) + 2 lambda w = 0

    =>   w = (Z^T Z + N lambda I)^{-1} Z^T y

Note the +N*lambda*I: it lifts every diagonal entry, which is exactly what
makes the matrix invertible even when the columns are collinear. The bias b is
recovered afterwards from the means and is never penalised.

----------------------------------------------------------------------
CHOOSING LAMBDA
----------------------------------------------------------------------
lambda cannot be chosen on the training error (larger lambda always increases
it). We split the training data into a smaller training part and a validation
part, fit with several values of lambda, and keep the one with the lowest
validation error.
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


# ======================================================================
# DATA
# ======================================================================
def load_autompg():
    os.makedirs(DATADIR, exist_ok=True)
    path = os.path.join(DATADIR, "auto_mpg.csv")
    if not os.path.exists(path):
        print(f"Downloading Auto MPG data from {URL} ...")
        urllib.request.urlretrieve(URL, path)
    df = pd.read_csv(path)
    n_raw = len(df)
    df = df.dropna().reset_index(drop=True)       # six rows lack horsepower
    features = ["cylinders", "displacement", "horsepower", "weight",
                "acceleration", "model_year"]
    X = df[features].to_numpy(dtype=float)
    # 'origin' is a category (USA / Europe / Japan), so turn it into 0/1 columns
    origin = pd.get_dummies(df["origin"], prefix="origin", drop_first=True)
    X = np.hstack([X, origin.to_numpy(dtype=float)])
    names = features + list(origin.columns)
    y = df["mpg"].to_numpy(dtype=float)
    return X, y, names, n_raw


# ======================================================================
# THE MATHEMATICS
# ======================================================================
def standardise(X, mean=None, std=None):
    """z = (x - mean) / std. Statistics come from the training set."""
    if mean is None:
        mean, std = X.mean(axis=0), X.std(axis=0)
        std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def ridge_fit(Z, y, lam):
    """w = (Z^T Z + N lambda I)^{-1} Z^T y on centred data; then recover b."""
    N, p = Z.shape
    z_mean, y_mean = Z.mean(axis=0), y.mean()
    Zc, yc = Z - z_mean, y - y_mean               # centre, so b drops out
    A = Zc.T @ Zc + N * lam * np.eye(p)
    w = np.linalg.solve(A, Zc.T @ yc)
    b = y_mean - z_mean @ w
    return w, b


def predict(Z, w, b):
    return Z @ w + b


def mse(y, y_hat):
    return float(np.mean((y - y_hat) ** 2))


def mae(y, y_hat):
    return float(np.mean(np.abs(y - y_hat)))


def r2_score(y, y_hat):
    return float(1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2))


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(RESDIR, exist_ok=True)

    X, y, names, n_raw = load_autompg()
    print("=" * 72)
    print("02 | RIDGE REGRESSION  -  Auto MPG dataset")
    print("=" * 72)
    print(f"Rows in the raw file       : {n_raw}")
    print(f"Rows after dropping missing: {X.shape[0]}")
    print(f"Features p = {X.shape[1]}: {', '.join(names)}")
    print(f"Target: mpg,  mean = {y.mean():.2f}, standard deviation = {y.std():.2f}")
    print()

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(y))
    n_test = int(0.2 * len(y))
    n_val = int(0.2 * len(y))
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]
    print(f"Train / validation / test: {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")

    Z_tr, mean, std = standardise(X_tr)
    Z_va, _, _ = standardise(X_va, mean, std)
    Z_te, _, _ = standardise(X_te, mean, std)
    print("Features standardised with the training mean and standard deviation.")
    print()

    # show the collinearity that motivates the penalty
    C = np.corrcoef(Z_tr, rowvar=False)
    print("Strongly correlated feature pairs (|correlation| > 0.8):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(C[i, j]) > 0.8:
                print(f"   {names[i]} and {names[j]} : {C[i, j]:+.3f}")
    print()

    # ---------------- choose lambda on the validation set ------------------
    lambdas = np.logspace(-6, 2, 17)
    val_errors = []
    print("-" * 72)
    print("(a) CHOOSING LAMBDA ON THE VALIDATION SET")
    print("-" * 72)
    print(f"{'lambda':>12}{'validation MSE':>18}")
    for lam in lambdas:
        w, b = ridge_fit(Z_tr, y_tr, lam)
        err = mse(y_va, predict(Z_va, w, b))
        val_errors.append(err)
        print(f"{lam:>12.1e}{err:>18.4f}")
    val_errors = np.array(val_errors)
    best_lambda = float(lambdas[np.argmin(val_errors)])
    print(f"\nBest lambda = {best_lambda:.1e} "
          f"(validation MSE = {val_errors.min():.4f})")
    print()

    # ---------------- final models ----------------------------------------
    w_ols, b_ols = ridge_fit(Z_tr, y_tr, 0.0)          # lambda = 0
    w, b = ridge_fit(Z_tr, y_tr, best_lambda)

    print("-" * 72)
    print("(b) COEFFICIENTS (change in mpg per one standard deviation)")
    print("-" * 72)
    print(f"{'feature':>16}{'OLS':>12}{'ridge':>12}")
    for name, a, c in zip(names, w_ols, w):
        print(f"{name:>16}{a:12.4f}{c:12.4f}")
    print(f"{'intercept':>16}{b_ols:12.4f}{b:12.4f}")
    print(f"\nSize of the weight vector ||w||: "
          f"OLS = {np.linalg.norm(w_ols):.4f}, ridge = {np.linalg.norm(w):.4f}")
    print("The ridge weights are smaller - that is the shrinkage the penalty buys.")
    print()

    results = {}
    for tag, (ww, bb) in {"ols": (w_ols, b_ols), "ridge": (w, b)}.items():
        p_tr, p_te = predict(Z_tr, ww, bb), predict(Z_te, ww, bb)
        results[tag] = {"train_mse": mse(y_tr, p_tr), "test_mse": mse(y_te, p_te),
                        "test_rmse": float(np.sqrt(mse(y_te, p_te))),
                        "test_mae": mae(y_te, p_te),
                        "train_r2": r2_score(y_tr, p_tr),
                        "test_r2": r2_score(y_te, p_te)}
    print("-" * 72)
    print("(c) PERFORMANCE")
    print("-" * 72)
    print(f"{'model':<8}{'train MSE':>12}{'test MSE':>12}"
          f"{'test RMSE':>12}{'test MAE':>11}{'test R^2':>10}")
    for tag in ("ols", "ridge"):
        m = results[tag]
        print(f"{tag:<8}{m['train_mse']:12.4f}{m['test_mse']:12.4f}"
              f"{m['test_rmse']:12.4f}{m['test_mae']:11.4f}{m['test_r2']:10.4f}")
    print()

    # ---------------- plots -------------------------------------------------
    coefficient_paths = np.array([ridge_fit(Z_tr, y_tr, lam)[0] for lam in lambdas])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogx(lambdas, val_errors, "o-", ms=4)
    ax[0].axvline(best_lambda, color="crimson", ls="--", lw=1,
                  label=rf"best $\lambda = {best_lambda:.1e}$")
    ax[0].set_xlabel(r"$\lambda$")
    ax[0].set_ylabel("validation MSE")
    ax[0].set_title("Choosing the penalty strength")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    for j, name in enumerate(names):
        ax[1].semilogx(lambdas, coefficient_paths[:, j], label=name, lw=1.4)
    ax[1].axvline(best_lambda, color="crimson", ls="--", lw=1)
    ax[1].set_xlabel(r"$\lambda$")
    ax[1].set_ylabel(r"$w_j$")
    ax[1].set_title("Weights shrink as $\\lambda$ grows")
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
    ax[0].set_title(f"Test set, $R^2$ = {results['ridge']['test_r2']:.3f}")
    ax[0].grid(alpha=0.3)
    ax[1].hist(y_te - p_te, bins=20, edgecolor="k", alpha=0.8)
    ax[1].set_xlabel(r"residual $y - \hat{y}$ (mpg)")
    ax[1].set_ylabel("count")
    ax[1].set_title("Residuals")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    f2 = os.path.join(FIGDIR, "02_autompg_fit.pdf")
    fig.savefig(f2)
    plt.close(fig)

    with open(os.path.join(RESDIR, "02_autompg.json"), "w") as fh:
        json.dump({"metrics": results, "best_lambda": best_lambda,
                   "validation_mse_min": float(val_errors.min()),
                   "coefficients_ridge": {n: float(v) for n, v in zip(names, w)},
                   "coefficients_ols": {n: float(v) for n, v in zip(names, w_ols)}},
                  fh, indent=2)
    print(f"Figures saved: {f1}")
    print(f"               {f2}")
    print(f"Results saved: {os.path.join(RESDIR, '02_autompg.json')}")


if __name__ == "__main__":
    main()
