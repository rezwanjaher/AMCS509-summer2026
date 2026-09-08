"""
01 - Regression: Linear Regression on the Diabetes dataset (scikit-learn / StatLib).

MATHEMATICS IMPLEMENTED HERE
----------------------------
Model (linear regression):
    yhat_i = w^T x_i + b,      x_i in R^p,  w in R^p,  b in R

Design-matrix form (bias absorbed as a column of ones):
    Xtilde = [1, X],   theta = [b, w],   yhat = Xtilde theta

Loss (Mean Squared Error):
    L(theta) = (1/N) * sum_i (y_i - xtilde_i^T theta)^2
             = (1/N) * ||y - Xtilde theta||^2

Gradient (vector calculus):
    dL/dtheta = -(2/N) * Xtilde^T (y - Xtilde theta)

Stationary point  dL/dtheta = 0  gives the NORMAL EQUATIONS:
    (Xtilde^T Xtilde) theta = Xtilde^T y
    theta_hat = (Xtilde^T Xtilde)^{-1} Xtilde^T y        (closed form / OLS)

Gradient descent (iterative alternative, lecture 7-8):
    theta_{t+1} = theta_t - eta * dL/dtheta

The Hessian of the MSE loss is
    H = (2/N) * Xtilde^T Xtilde,
which is positive semi-definite, so the MSE loss is convex: the stationary
point found above is a global minimum, and gradient descent converges to the
same solution as the normal equations.

Reported metrics:
    MSE = (1/n) sum (y - yhat)^2 ,   MAE = (1/n) sum |y - yhat| ,
    R^2 = 1 - SS_res / SS_tot
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
RESDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ----------------------------------------------------------------------
# Core mathematics
# ----------------------------------------------------------------------
def add_bias(X):
    """Xtilde = [1, X]."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def mse(y, yhat):
    return float(np.mean((y - yhat) ** 2))


def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def ols_closed_form(X, y):
    """theta = (Xt^T Xt)^{-1} Xt^T y, computed with a stable least-squares solve."""
    Xt = add_bias(X)
    theta, *_ = np.linalg.lstsq(Xt, y, rcond=None)
    return theta


def mse_gradient(Xt, y, theta):
    """dL/dtheta = -(2/N) Xt^T (y - Xt theta)."""
    N = Xt.shape[0]
    residual = y - Xt @ theta
    return -(2.0 / N) * (Xt.T @ residual)


def hessian_spectrum(X):
    """Eigenvalues of H = (2/N) Xt^T Xt, the (constant) Hessian of the MSE loss."""
    Xt = add_bias(X)
    H = (2.0 / Xt.shape[0]) * (Xt.T @ Xt)
    return np.linalg.eigvalsh(H)


def gradient_descent(X, y, eta=None, n_iter=200000, tol=1e-14):
    """Batch gradient descent on the MSE loss.

    For a quadratic loss the iteration is stable iff 0 < eta < 2/lambda_max(H);
    we use eta = 1.9 / lambda_max(H) unless the user overrides it.
    """
    Xt = add_bias(X)
    if eta is None:
        eta = 1.9 / hessian_spectrum(X).max()
    theta = np.zeros(Xt.shape[1])
    history = []
    for t in range(n_iter):
        g = mse_gradient(Xt, y, theta)
        theta_new = theta - eta * g
        loss = mse(y, Xt @ theta_new)
        history.append(loss)
        if np.linalg.norm(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new
    return theta, np.array(history), eta


def predict(X, theta):
    return add_bias(X) @ theta


# ----------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", type=float, default=None,
                    help="learning rate (default: 1.9 / lambda_max(H))")
    ap.add_argument("--iters", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(RESDIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    data = load_diabetes()
    X, y = data.data, data.target
    names = list(data.feature_names)

    print("=" * 72)
    print("01 | LINEAR REGRESSION  -  Diabetes dataset")
    print("=" * 72)
    print(f"Samples N = {X.shape[0]},  features p = {X.shape[1]}")
    print(f"Features: {', '.join(names)}")
    print("Target: quantitative measure of disease progression one year after baseline")
    print(f"Target mean = {y.mean():.3f}, std = {y.std():.3f}")
    print()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )
    print(f"Train / test split: {X_tr.shape[0]} / {X_te.shape[0]}")
    print()

    # --- (a) closed form -------------------------------------------------
    theta_cf = ols_closed_form(X_tr, y_tr)
    print("-" * 72)
    print("(a) CLOSED FORM  theta = (X^T X)^{-1} X^T y")
    print("-" * 72)
    Xt = add_bias(X_tr)
    G = Xt.T @ Xt
    print(f"Gram matrix X^T X : shape {G.shape}, condition number = {np.linalg.cond(G):.3e}")
    print(f"Intercept b = {theta_cf[0]:.4f}")
    for n, w in zip(names, theta_cf[1:]):
        print(f"   w[{n:>4s}] = {w:10.4f}")
    # residual gradient at the optimum should be numerically zero
    g_star = mse_gradient(Xt, y_tr, theta_cf)
    print(f"||dL/dtheta|| at the closed-form solution = {np.linalg.norm(g_star):.3e}  (should be ~0)")
    print()

    # --- (b) gradient descent -------------------------------------------
    ev = hessian_spectrum(X_tr)
    theta_gd, hist, eta_used = gradient_descent(X_tr, y_tr, eta=args.eta, n_iter=args.iters)
    print("-" * 72)
    print(f"(b) GRADIENT DESCENT  theta <- theta - eta * dL/dtheta")
    print("-" * 72)
    print(f"Hessian H = (2/N) X^T X : lambda_min = {ev.min():.4e}, lambda_max = {ev.max():.4e}")
    print(f"Condition number kappa  : {ev.max()/ev.min():.3e}  (large => slow convergence)")
    print(f"Step size used eta      : {eta_used:.6f}   (stability bound: eta < 2/lambda_max = {2/ev.max():.6f})")
    print(f"Iterations run           : {len(hist)}")
    print(f"Initial training MSE     : {mse(y_tr, predict(X_tr, np.zeros_like(theta_gd))):.4f}")
    print(f"Final training MSE (GD)  : {hist[-1]:.4f}")
    print(f"Training MSE (closed)    : {mse(y_tr, predict(X_tr, theta_cf)):.4f}")
    print(f"||theta_GD - theta_OLS|| : {np.linalg.norm(theta_gd - theta_cf):.6f}")
    print()

    # --- (c) evaluation --------------------------------------------------
    yhat_te = predict(X_te, theta_cf)
    yhat_tr = predict(X_tr, theta_cf)
    metrics = {
        "train_mse": mse(y_tr, yhat_tr),
        "train_mae": mae(y_tr, yhat_tr),
        "train_r2": r2(y_tr, yhat_tr),
        "test_mse": mse(y_te, yhat_te),
        "test_mae": mae(y_te, yhat_te),
        "test_r2": r2(y_te, yhat_te),
        "test_rmse": float(np.sqrt(mse(y_te, yhat_te))),
        "gd_final_train_mse": float(hist[-1]),
        "gd_iters": int(len(hist)),
        "theta_gap": float(np.linalg.norm(theta_gd - theta_cf)),
        "eta": float(eta_used),
        "hessian_lambda_min": float(ev.min()),
        "hessian_lambda_max": float(ev.max()),
        "condition_number": float(ev.max() / ev.min()),
    }
    print("-" * 72)
    print("(c) PERFORMANCE")
    print("-" * 72)
    print(f"{'':<10}{'MSE':>12}{'MAE':>12}{'R^2':>10}")
    print(f"{'train':<10}{metrics['train_mse']:>12.3f}{metrics['train_mae']:>12.3f}{metrics['train_r2']:>10.4f}")
    print(f"{'test':<10}{metrics['test_mse']:>12.3f}{metrics['test_mae']:>12.3f}{metrics['test_r2']:>10.4f}")
    print(f"test RMSE = {metrics['test_rmse']:.3f}")
    print()

    # --- plots -----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist, lw=1.6)
    ax[0].set_xlabel("iteration $t$")
    ax[0].set_ylabel(r"MSE  $L(\theta_t)$")
    ax[0].set_title(rf"Gradient descent, $\eta={eta_used:.3f}$")
    ax[0].set_yscale("log")
    ax[0].grid(alpha=0.3)
    ax[1].scatter(y_te, yhat_te, s=18, alpha=0.75, edgecolor="none")
    lims = [min(y_te.min(), yhat_te.min()), max(y_te.max(), yhat_te.max())]
    ax[1].plot(lims, lims, "k--", lw=1)
    ax[1].set_xlabel("true $y$")
    ax[1].set_ylabel(r"predicted $\hat{y}$")
    ax[1].set_title(f"Test set, $R^2$ = {metrics['test_r2']:.3f}")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    f1 = os.path.join(FIGDIR, "01_diabetes_fit.pdf")
    fig.savefig(f1)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    res = y_te - yhat_te
    ax[0].scatter(yhat_te, res, s=18, alpha=0.75, edgecolor="none")
    ax[0].axhline(0, color="k", lw=1, ls="--")
    ax[0].set_xlabel(r"predicted $\hat{y}$")
    ax[0].set_ylabel(r"residual $e = y-\hat{y}$")
    ax[0].set_title("Residuals vs fitted")
    ax[0].grid(alpha=0.3)
    order = np.argsort(theta_cf[1:])
    ax[1].barh(np.array(names)[order], theta_cf[1:][order])
    ax[1].set_xlabel(r"weight $w_j$")
    ax[1].set_title("OLS coefficients")
    ax[1].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    f2 = os.path.join(FIGDIR, "01_diabetes_diagnostics.pdf")
    fig.savefig(f2)
    plt.close(fig)

    with open(os.path.join(RESDIR, "01_diabetes.json"), "w") as fh:
        json.dump(
            {
                "metrics": metrics,
                "coefficients": {n: float(w) for n, w in zip(names, theta_cf[1:])},
                "intercept": float(theta_cf[0]),
            },
            fh,
            indent=2,
        )
    print(f"Figures saved: {f1}")
    print(f"               {f2}")
    print(f"Results saved: {os.path.join(RESDIR, '01_diabetes.json')}")


if __name__ == "__main__":
    main()
