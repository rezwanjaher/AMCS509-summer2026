"""
03 - Classification: Softmax (multinomial logistic) regression on the Iris dataset.

MATHEMATICS IMPLEMENTED HERE
----------------------------
Scores (logits) for C classes:
    z_i = W^T x_i + b,      W in R^{p x C},  b in R^C

Softmax link (lecture 3-4):
    p_ic = exp(z_ic) / sum_k exp(z_ik),      sum_c p_ic = 1,  p_ic in (0,1)

Numerically stable form: subtract max_k z_ik before exponentiating; this leaves
the softmax unchanged because exp(z-m)/sum exp(z-m) = exp(z)/sum exp(z).

Loss (categorical cross-entropy with one-hot targets Y):
    L = -(1/N) sum_i sum_c y_ic log p_ic

Jacobian of the softmax:
    dp_c / dz_k = p_c (delta_ck - p_k)

Chain rule through the softmax and the log gives the remarkably simple gradient
of the cross-entropy with respect to the logits:
    dL/dz_i = (p_i - y_i) / N
and therefore
    dL/dW = (1/N) X^T (P - Y),      dL/db = (1/N) 1^T (P - Y)

Optimisation: batch gradient descent
    W <- W - eta dL/dW,   b <- b - eta dL/db

The cross-entropy of the softmax model is convex in (W, b), so gradient descent
reaches a global optimum. The parameterisation is over-complete (adding the same
vector to every column of W leaves P unchanged), which is why a small L2 term is
included to pin down a unique solution.

Prediction: class = argmax_c p_c. Accuracy, per-class precision/recall and the
confusion matrix are reported.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
RESDIR = os.path.join(HERE, "results")


# ----------------------------------------------------------------------
# Core mathematics
# ----------------------------------------------------------------------
def one_hot(y, C):
    Y = np.zeros((len(y), C))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def softmax(Z):
    """Row-wise, numerically stabilised softmax."""
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def cross_entropy(P, Y, W=None, lam=0.0):
    N = Y.shape[0]
    L = -np.sum(Y * np.log(P + 1e-12)) / N
    if W is not None and lam > 0:
        L += lam * np.sum(W * W)
    return float(L)


def grads(X, Y, W, b, lam=0.0):
    """dL/dW = (1/N) X^T (P - Y) + 2 lam W ;  dL/db = (1/N) 1^T (P - Y)."""
    N = X.shape[0]
    P = softmax(X @ W + b)
    D = (P - Y) / N
    return X.T @ D + 2 * lam * W, D.sum(axis=0), P


def fit(X, Y, eta=0.5, n_iter=4000, lam=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    p, C = X.shape[1], Y.shape[1]
    W = 0.01 * rng.standard_normal((p, C))
    b = np.zeros(C)
    hist = []
    for _ in range(n_iter):
        gW, gb, P = grads(X, Y, W, b, lam)
        W -= eta * gW
        b -= eta * gb
        hist.append(cross_entropy(P, Y, W, lam))
    return W, b, np.array(hist)


def gradient_check(X, Y, W, b, lam, eps=1e-6):
    """Compare the analytic dL/dW with a central finite difference."""
    gW, _, _ = grads(X, Y, W, b, lam)
    num = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp, Wm = W.copy(), W.copy()
            Wp[i, j] += eps
            Wm[i, j] -= eps
            Lp = cross_entropy(softmax(X @ Wp + b), Y, Wp, lam)
            Lm = cross_entropy(softmax(X @ Wm + b), Y, Wm, lam)
            num[i, j] = (Lp - Lm) / (2 * eps)
    return float(np.max(np.abs(gW - num)))


def predict(X, W, b):
    return np.argmax(X @ W + b, axis=1)


def confusion(y, yhat, C):
    M = np.zeros((C, C), dtype=int)
    for t, p in zip(y, yhat):
        M[t, p] += 1
    return M


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(RESDIR, exist_ok=True)

    data = load_iris()
    X, y = data.data, data.target
    names, cls = list(data.feature_names), list(data.target_names)
    C = len(cls)

    print("=" * 72)
    print("03 | SOFTMAX REGRESSION  -  Iris dataset")
    print("=" * 72)
    print(f"Samples N = {X.shape[0]}, features p = {X.shape[1]}, classes C = {C}")
    print(f"Features: {', '.join(names)}")
    print(f"Classes : {', '.join(cls)}  (counts: {np.bincount(y).tolist()})")
    print()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y
    )
    mu, sd = X_tr.mean(0), X_tr.std(0)
    Z_tr, Z_te = (X_tr - mu) / sd, (X_te - mu) / sd
    Y_tr, Y_te = one_hot(y_tr, C), one_hot(y_te, C)
    print(f"Train / test split: {len(y_tr)} / {len(y_te)} (stratified, standardised)")
    print()

    # --- gradient check ---------------------------------------------------
    rng = np.random.default_rng(args.seed)
    W0 = 0.3 * rng.standard_normal((X.shape[1], C))
    b0 = 0.1 * rng.standard_normal(C)
    err = gradient_check(Z_tr, Y_tr, W0, b0, args.lam)
    print("-" * 72)
    print("(a) GRADIENT CHECK  analytic  X^T(P-Y)/N  vs central finite differences")
    print("-" * 72)
    print(f"max |analytic - numerical| = {err:.3e}  (should be ~1e-9 or smaller)")
    # softmax sanity: rows sum to 1
    Pchk = softmax(Z_tr @ W0 + b0)
    print(f"max |row sum of softmax - 1| = {np.max(np.abs(Pchk.sum(1) - 1)):.3e}")
    print()

    # --- training ---------------------------------------------------------
    W, b, hist = fit(Z_tr, Y_tr, eta=args.eta, n_iter=args.iters,
                     lam=args.lam, seed=args.seed)
    print("-" * 72)
    print(f"(b) TRAINING  (eta = {args.eta}, iterations = {args.iters}, lambda = {args.lam})")
    print("-" * 72)
    print(f"Initial cross-entropy      : {hist[0]:.6f}   (log C = {np.log(C):.6f} for uniform guessing)")
    print(f"Final training cross-entropy: {hist[-1]:.6f}")
    P_te = softmax(Z_te @ W + b)
    print(f"Final test cross-entropy    : {cross_entropy(P_te, Y_te):.6f}")
    print()
    print("Learned weight matrix W (standardised features x classes):")
    print(f"{'feature':>22}" + "".join(f"{c:>14}" for c in cls))
    for n, row in zip(names, W):
        print(f"{n:>22}" + "".join(f"{v:14.4f}" for v in row))
    print(f"{'bias':>22}" + "".join(f"{v:14.4f}" for v in b))
    print()

    # --- evaluation -------------------------------------------------------
    yhat_tr, yhat_te = predict(Z_tr, W, b), predict(Z_te, W, b)
    acc_tr = float(np.mean(yhat_tr == y_tr))
    acc_te = float(np.mean(yhat_te == y_te))
    M = confusion(y_te, yhat_te, C)
    print("-" * 72)
    print("(c) PERFORMANCE")
    print("-" * 72)
    print(f"Train accuracy = {acc_tr:.4f}    Test accuracy = {acc_te:.4f}")
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(f"{'':>12}" + "".join(f"{c:>12}" for c in cls))
    for i, c in enumerate(cls):
        print(f"{c:>12}" + "".join(f"{v:12d}" for v in M[i]))
    print("\nPer-class precision / recall / F1 (test):")
    per = {}
    for i, c in enumerate(cls):
        tp = M[i, i]
        prec = tp / M[:, i].sum() if M[:, i].sum() else 0.0
        rec = tp / M[i].sum() if M[i].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"precision": float(prec), "recall": float(rec), "f1": float(f1)}
        print(f"   {c:>12}: precision = {prec:.4f}, recall = {rec:.4f}, F1 = {f1:.4f}")
    print()

    # --- plots ------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist, lw=1.6)
    ax[0].axhline(np.log(C), color="gray", ls=":", lw=1, label=r"$\log C$ (chance)")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("cross-entropy")
    ax[0].set_title(rf"Softmax regression training, $\eta={args.eta}$")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    im = ax[1].imshow(M, cmap="Blues")
    ax[1].set_xticks(range(C), cls, rotation=30)
    ax[1].set_yticks(range(C), cls)
    for i in range(C):
        for j in range(C):
            ax[1].text(j, i, M[i, j], ha="center", va="center",
                       color="white" if M[i, j] > M.max() / 2 else "black")
    ax[1].set_xlabel("predicted")
    ax[1].set_ylabel("true")
    ax[1].set_title(f"Test confusion matrix (acc = {acc_te:.3f})")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    f1p = os.path.join(FIGDIR, "03_iris_training.pdf")
    fig.savefig(f1p)
    plt.close(fig)

    # decision regions using the two petal features only
    j1, j2 = 2, 3
    W2, b2, _ = fit(Z_tr[:, [j1, j2]], Y_tr, eta=args.eta,
                    n_iter=args.iters, lam=args.lam, seed=args.seed)
    gx, gy = np.meshgrid(
        np.linspace(Z_tr[:, j1].min() - 1, Z_tr[:, j1].max() + 1, 300),
        np.linspace(Z_tr[:, j2].min() - 1, Z_tr[:, j2].max() + 1, 300),
    )
    grid = np.c_[gx.ravel(), gy.ravel()]
    zz = predict(grid, W2, b2).reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.contourf(gx, gy, zz, levels=[-0.5, 0.5, 1.5, 2.5], alpha=0.25,
                colors=["tab:blue", "tab:orange", "tab:green"])
    for c in range(C):
        m = y_tr == c
        ax.scatter(Z_tr[m, j1], Z_tr[m, j2], s=22, label=cls[c], edgecolor="k", lw=0.3)
    ax.set_xlabel(f"{names[j1]} (standardised)")
    ax.set_ylabel(f"{names[j2]} (standardised)")
    ax.set_title("Softmax decision regions (linear boundaries)")
    ax.legend()
    fig.tight_layout()
    f2p = os.path.join(FIGDIR, "03_iris_decision_regions.pdf")
    fig.savefig(f2p)
    plt.close(fig)

    with open(os.path.join(RESDIR, "03_iris.json"), "w") as fh:
        json.dump({"train_acc": acc_tr, "test_acc": acc_te,
                   "train_ce": float(hist[-1]),
                   "test_ce": cross_entropy(P_te, Y_te),
                   "grad_check_max_err": err,
                   "confusion": M.tolist(), "per_class": per,
                   "classes": cls}, fh, indent=2)
    print(f"Figures saved: {f1p}")
    print(f"               {f2p}")
    print(f"Results saved: {os.path.join(RESDIR, '03_iris.json')}")


if __name__ == "__main__":
    main()
