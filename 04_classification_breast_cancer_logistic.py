"""
04 - Classification: Binary logistic regression on the Breast Cancer Wisconsin
     dataset (UCI / scikit-learn), trained by (a) gradient descent and
     (b) Newton's method (IRLS).

MATHEMATICS IMPLEMENTED HERE
----------------------------
Model:
    p_i = P(y_i = 1 | x_i) = sigma(w^T x_i + b),   sigma(z) = 1 / (1 + e^{-z})

Key derivative of the sigmoid (Task in lecture 3-4):
    sigma'(z) = sigma(z) (1 - sigma(z))
    Proof: sigma(z) = (1+e^{-z})^{-1};
           sigma'(z) = e^{-z} (1+e^{-z})^{-2}
                     = [1/(1+e^{-z})] * [e^{-z}/(1+e^{-z})]
                     = sigma(z)(1 - sigma(z)).

Loss (binary cross-entropy / log loss) with L2 penalty:
    L(theta) = -(1/N) sum_i [ y_i log p_i + (1-y_i) log(1-p_i) ] + lambda ||w||^2

Gradient (chain rule dL/dp * dp/dz * dz/dw, the sigmoid factor cancels):
    dL/dz_i = (p_i - y_i)/N
    dL/dw   = (1/N) X^T (p - y) + 2 lambda w
    dL/db   = (1/N) 1^T (p - y)

Hessian (second-order information, lecture 7-8):
    H = (1/N) X^T S X + 2 lambda I,   S = diag(p_i (1 - p_i))
S is positive semi-definite, hence H is positive definite for lambda > 0, so the
regularised log loss is strictly convex and has a unique global minimum.

Newton / IRLS update:
    theta <- theta - H^{-1} dL/dtheta
Because it rescales the gradient by the local curvature, Newton converges in a
handful of iterations, while plain gradient descent needs thousands.

Decision rule: yhat = 1 if p >= t (default t = 0.5). The ROC curve sweeps t and
the area under it (AUC) is reported.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
RESDIR = os.path.join(HERE, "results")


# ----------------------------------------------------------------------
# Core mathematics
# ----------------------------------------------------------------------
def sigmoid(z):
    """Stable logistic sigmoid."""
    out = np.empty_like(z, dtype=float)
    pos, neg = z >= 0, z < 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[neg])
    out[neg] = e / (1.0 + e)
    return out


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def bce(theta, Xt, y, lam):
    p = sigmoid(Xt @ theta)
    eps = 1e-12
    L = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    return float(L + lam * np.sum(theta[1:] ** 2))


def gradient(theta, Xt, y, lam):
    N = Xt.shape[0]
    p = sigmoid(Xt @ theta)
    g = Xt.T @ (p - y) / N
    reg = np.zeros_like(theta)
    reg[1:] = 2 * lam * theta[1:]        # intercept is never penalised
    return g + reg


def hessian(theta, Xt, y, lam):
    N = Xt.shape[0]
    p = sigmoid(Xt @ theta)
    s = p * (1 - p)
    H = (Xt * s[:, None]).T @ Xt / N
    R = 2 * lam * np.eye(Xt.shape[1])
    R[0, 0] = 0.0
    return H + R


def fit_gd(Xt, y, lam, eta, n_iter):
    theta = np.zeros(Xt.shape[1])
    hist = [bce(theta, Xt, y, lam)]
    for _ in range(n_iter):
        theta -= eta * gradient(theta, Xt, y, lam)
        hist.append(bce(theta, Xt, y, lam))
    return theta, np.array(hist)


def fit_newton(Xt, y, lam, n_iter=12):
    theta = np.zeros(Xt.shape[1])
    hist = [bce(theta, Xt, y, lam)]
    for _ in range(n_iter):
        g = gradient(theta, Xt, y, lam)
        H = hessian(theta, Xt, y, lam)
        theta -= np.linalg.solve(H, g)
        hist.append(bce(theta, Xt, y, lam))
    return theta, np.array(hist)


def gradient_check(theta, Xt, y, lam, eps=1e-6):
    g = gradient(theta, Xt, y, lam)
    num = np.zeros_like(g)
    for i in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps
        tm[i] -= eps
        num[i] = (bce(tp, Xt, y, lam) - bce(tm, Xt, y, lam)) / (2 * eps)
    return float(np.max(np.abs(g - num)))


def roc_curve(y, scores):
    """Sweep the threshold and return (FPR, TPR) plus the AUC (trapezoid rule)."""
    order = np.argsort(-scores)
    y = y[order]
    P, N = y.sum(), len(y) - y.sum()
    tpr = np.concatenate([[0], np.cumsum(y) / P])
    fpr = np.concatenate([[0], np.cumsum(1 - y) / N])
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def metrics_at(y, p, t=0.5):
    yhat = (p >= t).astype(int)
    tp = int(np.sum((yhat == 1) & (y == 1)))
    tn = int(np.sum((yhat == 0) & (y == 0)))
    fp = int(np.sum((yhat == 1) & (y == 0)))
    fn = int(np.sum((yhat == 0) & (y == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": (tp + tn) / len(y),
            "precision": prec, "recall": rec, "f1": f1,
            "specificity": tn / (tn + fp) if tn + fp else 0.0}


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--newton-iters", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(RESDIR, exist_ok=True)

    data = load_breast_cancer()
    X, y = data.data, data.target.astype(float)
    names = list(data.feature_names)

    print("=" * 72)
    print("04 | LOGISTIC REGRESSION  -  Breast Cancer Wisconsin (diagnostic)")
    print("=" * 72)
    print(f"Samples N = {X.shape[0]}, features p = {X.shape[1]}")
    print(f"Classes: {data.target_names[0]} = 0 ({int((y==0).sum())}), "
          f"{data.target_names[1]} = 1 ({int((y==1).sum())})")
    print()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=args.seed, stratify=y
    )
    mu, sd = X_tr.mean(0), X_tr.std(0)
    Xt_tr = add_bias((X_tr - mu) / sd)
    Xt_te = add_bias((X_te - mu) / sd)
    print(f"Train / test split: {len(y_tr)} / {len(y_te)} (stratified, standardised)")
    print()

    # --- gradient check ---------------------------------------------------
    rng = np.random.default_rng(args.seed)
    th0 = 0.1 * rng.standard_normal(Xt_tr.shape[1])
    err = gradient_check(th0, Xt_tr, y_tr, args.lam)
    print("-" * 72)
    print("(a) CHECKS")
    print("-" * 72)
    print(f"max |analytic grad - finite difference| = {err:.3e}")
    zz = np.array([-2.0, -0.5, 0.0, 0.7, 3.0])
    num_ds = (sigmoid(zz + 1e-6) - sigmoid(zz - 1e-6)) / 2e-6
    ana_ds = sigmoid(zz) * (1 - sigmoid(zz))
    print(f"max |sigma'(z) - sigma(z)(1-sigma(z))|  = {np.max(np.abs(num_ds - ana_ds)):.3e}")
    print()

    # --- training ---------------------------------------------------------
    th_gd, hist_gd = fit_gd(Xt_tr, y_tr, args.lam, args.eta, args.iters)
    th_nt, hist_nt = fit_newton(Xt_tr, y_tr, args.lam, args.newton_iters)
    print("-" * 72)
    print("(b) OPTIMISATION: gradient descent vs Newton (IRLS)")
    print("-" * 72)
    print(f"{'iter':>6}{'BCE (grad. desc.)':>22}{'BCE (Newton)':>18}")
    for k in [0, 1, 2, 3, 5, 8, args.newton_iters]:
        gd = hist_gd[min(k, len(hist_gd) - 1)]
        print(f"{k:>6}{gd:22.8f}{hist_nt[min(k, len(hist_nt)-1)]:18.8f}")
    print(f"{args.iters:>6}{hist_gd[-1]:22.8f}{'-':>18}")
    print(f"\nFinal training BCE : GD = {hist_gd[-1]:.8f} after {args.iters} iterations")
    print(f"                     Newton = {hist_nt[-1]:.8f} after {args.newton_iters} iterations")
    print(f"||theta_GD - theta_Newton|| = {np.linalg.norm(th_gd - th_nt):.6f}")
    H = hessian(th_nt, Xt_tr, y_tr, args.lam)
    ev = np.linalg.eigvalsh(H)
    print(f"Hessian at the optimum: lambda_min = {ev.min():.4e} > 0  => strict convexity, "
          f"kappa = {ev.max()/ev.min():.3e}")
    print()

    theta = th_nt
    p_tr = sigmoid(Xt_tr @ theta)
    p_te = sigmoid(Xt_te @ theta)

    print("Ten largest |weights| (standardised features):")
    order = np.argsort(-np.abs(theta[1:]))[:10]
    for j in order:
        print(f"   {names[j]:>26} : {theta[1+j]:+8.4f}")
    print(f"   {'intercept':>26} : {theta[0]:+8.4f}")
    print()

    # --- evaluation -------------------------------------------------------
    m_tr = metrics_at(y_tr, p_tr)
    m_te = metrics_at(y_te, p_te)
    fpr, tpr, auc = roc_curve(y_te.astype(int), p_te)
    print("-" * 72)
    print("(c) PERFORMANCE  (threshold t = 0.5)")
    print("-" * 72)
    print(f"{'':<8}{'BCE':>12}{'acc':>10}{'prec':>10}{'recall':>10}{'F1':>10}{'spec':>10}")
    print(f"{'train':<8}{bce(theta, Xt_tr, y_tr, 0.0):12.6f}{m_tr['accuracy']:10.4f}"
          f"{m_tr['precision']:10.4f}{m_tr['recall']:10.4f}{m_tr['f1']:10.4f}{m_tr['specificity']:10.4f}")
    print(f"{'test':<8}{bce(theta, Xt_te, y_te, 0.0):12.6f}{m_te['accuracy']:10.4f}"
          f"{m_te['precision']:10.4f}{m_te['recall']:10.4f}{m_te['f1']:10.4f}{m_te['specificity']:10.4f}")
    print(f"\nTest confusion matrix: TP = {m_te['tp']}, FP = {m_te['fp']}, "
          f"FN = {m_te['fn']}, TN = {m_te['tn']}")
    print(f"Test AUC = {auc:.4f}")
    print()

    # --- plots ------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist_gd, label=f"gradient descent ($\\eta$={args.eta})", lw=1.6)
    ax[0].plot(hist_nt, "o-", ms=4, label="Newton / IRLS", lw=1.6)
    ax[0].set_xscale("symlog")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("regularised BCE")
    ax[0].set_title("First order vs second order optimisation")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(fpr, tpr, lw=1.8, label=f"AUC = {auc:.4f}")
    ax[1].plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax[1].set_xlabel("false positive rate")
    ax[1].set_ylabel("true positive rate")
    ax[1].set_title("ROC curve (test set)")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    f1p = os.path.join(FIGDIR, "04_breast_cancer_training_roc.pdf")
    fig.savefig(f1p)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    zs = np.linspace(-8, 8, 400)
    ax[0].plot(zs, sigmoid(zs), lw=1.8, label=r"$\sigma(z)$")
    ax[0].plot(zs, sigmoid(zs) * (1 - sigmoid(zs)), lw=1.8,
               label=r"$\sigma'(z)=\sigma(1-\sigma)$")
    ax[0].axhline(0.5, color="gray", ls=":", lw=1)
    ax[0].set_xlabel("$z$")
    ax[0].set_title("Sigmoid and its derivative")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].hist(p_te[y_te == 0], bins=25, alpha=0.7, label="malignant (y=0)")
    ax[1].hist(p_te[y_te == 1], bins=25, alpha=0.7, label="benign (y=1)")
    ax[1].axvline(0.5, color="k", ls="--", lw=1, label="threshold 0.5")
    ax[1].set_xlabel(r"predicted probability $\hat{p}$")
    ax[1].set_ylabel("count")
    ax[1].set_title("Separation of predicted probabilities")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    f2p = os.path.join(FIGDIR, "04_breast_cancer_sigmoid_probs.pdf")
    fig.savefig(f2p)
    plt.close(fig)

    with open(os.path.join(RESDIR, "04_breast_cancer.json"), "w") as fh:
        json.dump({"train": m_tr, "test": m_te, "auc": auc,
                   "train_bce": bce(theta, Xt_tr, y_tr, 0.0),
                   "test_bce": bce(theta, Xt_te, y_te, 0.0),
                   "gd_final_bce": float(hist_gd[-1]),
                   "newton_final_bce": float(hist_nt[-1]),
                   "newton_iters": args.newton_iters, "gd_iters": args.iters,
                   "grad_check_max_err": err,
                   "hessian_lambda_min": float(ev.min()),
                   "hessian_cond": float(ev.max() / ev.min())}, fh, indent=2)
    print(f"Figures saved: {f1p}")
    print(f"               {f2p}")
    print(f"Results saved: {os.path.join(RESDIR, '04_breast_cancer.json')}")


if __name__ == "__main__":
    main()
