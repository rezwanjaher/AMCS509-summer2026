"""
03 - Classification: Softmax Regression (multi-class logistic regression) on the
     Iris dataset, trained with standard gradient descent.

----------------------------------------------------------------------
THE MODEL
----------------------------------------------------------------------
With C classes we keep one weight vector and one bias per class, collected in
a matrix W (p x C) and a vector b (length C):

    z_i = W^T x_i + b                    (C scores for sample i)
    p_ic = softmax(z_i)_c = e^{z_ic} / sum_k e^{z_ik}

The probabilities are positive and sum to 1. The predicted class is the one
with the largest probability.

A useful trick: subtracting the largest score from every score leaves the
softmax unchanged, because the factor cancels in numerator and denominator.
We always do this so that e^{...} can never overflow.

----------------------------------------------------------------------
THE LOSS: CATEGORICAL CROSS-ENTROPY
----------------------------------------------------------------------
Targets are one-hot: Y[i, c] = 1 for the true class of sample i, 0 otherwise.

    L = -(1/N) sum_i sum_c Y[i,c] log(p_ic)  +  lambda ||W||^2

Because only one entry of each row of Y is 1, this is just the average of
-log(probability given to the correct class). If the model predicted every
class equally, the loss would be log C.

----------------------------------------------------------------------
THE GRADIENT
----------------------------------------------------------------------
Differentiating the softmax and the logarithm together, everything cancels and
leaves the same simple result as in binary logistic regression:

    dL/dz = (P - Y)          (predicted probabilities minus one-hot targets)

and then, since z = W^T x + b,

    dL/dW = (1/N) X^T (P - Y) + 2 lambda W
    dL/db = (1/N) column sums of (P - Y)

----------------------------------------------------------------------
TRAINING
----------------------------------------------------------------------
    W <- W - eta * dL/dW
    b <- b - eta * dL/db

A small L2 term is included because the parameters are otherwise not unique:
adding the same vector to every column of W leaves all probabilities unchanged.
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


# ======================================================================
# THE MATHEMATICS
# ======================================================================
def one_hot(y, C):
    """Label 2 with C = 3 becomes the row [0, 0, 1]."""
    Y = np.zeros((len(y), C))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def softmax(Z):
    """Row-wise softmax, with the row maximum subtracted for safety."""
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def predict_proba(X, W, b):
    """Forward pass: scores then probabilities."""
    return softmax(X @ W + b)


def cross_entropy(X, Y, W, b, lam):
    """Categorical cross-entropy plus the L2 penalty."""
    P = predict_proba(X, W, b)
    loss = -np.sum(Y * np.log(P + 1e-12)) / Y.shape[0]
    return float(loss + lam * np.sum(W ** 2))


def gradients(X, Y, W, b, lam):
    """dL/dW = (1/N) X^T (P - Y) + 2*lambda*W ;  dL/db = mean of (P - Y)."""
    N = X.shape[0]
    P = predict_proba(X, W, b)
    error = (P - Y) / N                       # this is dL/dz
    grad_W = X.T @ error + 2 * lam * W
    grad_b = error.sum(axis=0)
    return grad_W, grad_b


def train(X, Y, eta, n_iter, lam):
    """Gradient descent on W and b."""
    p, C = X.shape[1], Y.shape[1]
    W = np.zeros((p, C))
    b = np.zeros(C)
    history = [cross_entropy(X, Y, W, b, lam)]
    for _ in range(n_iter):
        grad_W, grad_b = gradients(X, Y, W, b, lam)
        W = W - eta * grad_W
        b = b - eta * grad_b
        history.append(cross_entropy(X, Y, W, b, lam))
    return W, b, np.array(history)


def predict(X, W, b):
    """Predicted class = the one with the largest score."""
    return np.argmax(X @ W + b, axis=1)


def confusion_matrix(y_true, y_pred, C):
    M = np.zeros((C, C), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[t, p] += 1
    return M


# ======================================================================
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
    names, class_names = list(data.feature_names), list(data.target_names)
    C = len(class_names)

    print("=" * 72)
    print("03 | SOFTMAX REGRESSION  -  Iris dataset")
    print("=" * 72)
    print(f"Samples N = {X.shape[0]}, features p = {X.shape[1]}, classes C = {C}")
    print(f"Features: {', '.join(names)}")
    print(f"Classes : {', '.join(class_names)} "
          f"(counts {np.bincount(y).tolist()})")
    print()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y)
    mean, std = X_tr.mean(axis=0), X_tr.std(axis=0)
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std
    Y_tr, Y_te = one_hot(y_tr, C), one_hot(y_te, C)
    print(f"Train / test split: {len(y_tr)} / {len(y_te)} "
          "(stratified, features standardised)")
    print()

    # ---------------- training --------------------------------------------
    print("-" * 72)
    print(f"(a) TRAINING  (eta = {args.eta}, iterations = {args.iters}, "
          f"lambda = {args.lam})")
    print("-" * 72)
    W, b, history = train(X_tr, Y_tr, args.eta, args.iters, args.lam)
    print(f"{'iteration':>12}{'training loss':>16}")
    for k in [0, 10, 100, 500, 1000, args.iters]:
        print(f"{k:>12}{history[k]:>16.6f}")
    print(f"\nAt iteration 0 all scores are zero, so every class gets probability")
    print(f"1/3 and the loss is log 3 = {np.log(3):.6f}.")
    print(f"Test loss after training: "
          f"{cross_entropy(X_te, Y_te, W, b, 0.0):.6f}")
    print()

    print("Learned weights (standardised features x classes):")
    print(f"{'feature':>22}" + "".join(f"{c:>14}" for c in class_names))
    for name, row in zip(names, W):
        print(f"{name:>22}" + "".join(f"{v:14.4f}" for v in row))
    print(f"{'bias':>22}" + "".join(f"{v:14.4f}" for v in b))
    print()

    # ---------------- evaluation ------------------------------------------
    y_pred_tr = predict(X_tr, W, b)
    y_pred_te = predict(X_te, W, b)
    acc_tr = float(np.mean(y_pred_tr == y_tr))
    acc_te = float(np.mean(y_pred_te == y_te))
    M = confusion_matrix(y_te, y_pred_te, C)

    print("-" * 72)
    print("(b) PERFORMANCE")
    print("-" * 72)
    print(f"Train accuracy = {acc_tr:.4f}    Test accuracy = {acc_te:.4f}")
    print("\nTest confusion matrix (rows = true, columns = predicted):")
    print(f"{'':>12}" + "".join(f"{c:>12}" for c in class_names))
    for i, c in enumerate(class_names):
        print(f"{c:>12}" + "".join(f"{v:12d}" for v in M[i]))
    print("\nPer-class precision / recall / F1:")
    per_class = {}
    for i, c in enumerate(class_names):
        tp = M[i, i]
        precision = tp / M[:, i].sum() if M[:, i].sum() else 0.0
        recall = tp / M[i].sum() if M[i].sum() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = {"precision": float(precision), "recall": float(recall),
                        "f1": float(f1)}
        print(f"   {c:>12}: precision = {precision:.4f}, "
              f"recall = {recall:.4f}, F1 = {f1:.4f}")
    print()

    # ---------------- plots ------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history, lw=1.6)
    ax[0].axhline(np.log(C), color="gray", ls=":", lw=1,
                  label=r"$\log C$ (equal probabilities)")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("cross-entropy")
    ax[0].set_title(rf"Training loss, $\eta = {args.eta}$")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    im = ax[1].imshow(M, cmap="Blues")
    ax[1].set_xticks(range(C), class_names, rotation=30)
    ax[1].set_yticks(range(C), class_names)
    for i in range(C):
        for j in range(C):
            ax[1].text(j, i, M[i, j], ha="center", va="center",
                       color="white" if M[i, j] > M.max() / 2 else "black")
    ax[1].set_xlabel("predicted")
    ax[1].set_ylabel("true")
    ax[1].set_title(f"Test confusion matrix (accuracy = {acc_te:.3f})")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    f1p = os.path.join(FIGDIR, "03_iris_training.pdf")
    fig.savefig(f1p)
    plt.close(fig)

    # decision regions using only the two petal measurements
    j1, j2 = 2, 3
    W2, b2, _ = train(X_tr[:, [j1, j2]], Y_tr, args.eta, args.iters, args.lam)
    gx, gy = np.meshgrid(
        np.linspace(X_tr[:, j1].min() - 1, X_tr[:, j1].max() + 1, 300),
        np.linspace(X_tr[:, j2].min() - 1, X_tr[:, j2].max() + 1, 300))
    grid = np.c_[gx.ravel(), gy.ravel()]
    zz = predict(grid, W2, b2).reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.contourf(gx, gy, zz, levels=[-0.5, 0.5, 1.5, 2.5], alpha=0.25,
                colors=["tab:blue", "tab:orange", "tab:green"])
    for c in range(C):
        m = y_tr == c
        ax.scatter(X_tr[m, j1], X_tr[m, j2], s=22, label=class_names[c],
                   edgecolor="k", lw=0.3)
    ax.set_xlabel(f"{names[j1]} (standardised)")
    ax.set_ylabel(f"{names[j2]} (standardised)")
    ax.set_title("Decision regions (the boundaries are straight lines)")
    ax.legend()
    fig.tight_layout()
    f2p = os.path.join(FIGDIR, "03_iris_decision_regions.pdf")
    fig.savefig(f2p)
    plt.close(fig)

    with open(os.path.join(RESDIR, "03_iris.json"), "w") as fh:
        json.dump({"train_acc": acc_tr, "test_acc": acc_te,
                   "train_loss": float(history[-1]),
                   "test_loss": cross_entropy(X_te, Y_te, W, b, 0.0),
                   "confusion": M.tolist(), "per_class": per_class,
                   "classes": class_names, "eta": args.eta,
                   "iterations": args.iters, "lam": args.lam}, fh, indent=2)
    print(f"Figures saved: {f1p}")
    print(f"               {f2p}")
    print(f"Results saved: {os.path.join(RESDIR, '03_iris.json')}")


if __name__ == "__main__":
    main()
