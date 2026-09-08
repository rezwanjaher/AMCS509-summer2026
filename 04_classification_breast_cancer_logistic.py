"""
04 - Classification: Logistic Regression on the Breast Cancer Wisconsin dataset
     (UCI / scikit-learn), trained with standard gradient descent.

Everything is written from scratch with NumPy: the sigmoid, the loss, the
gradient and the training loop.

----------------------------------------------------------------------
THE MODEL
----------------------------------------------------------------------
Each sample x has p features. We compute a linear score and squash it into a
probability with the sigmoid:

    z_i = w^T x_i + b
    p_i = sigma(z_i) = 1 / (1 + e^{-z_i})       so that 0 < p_i < 1

p_i is read as P(y_i = 1 | x_i). The decision rule is

    y_pred = 1 if p >= 0.5, else 0

and since sigma(z) >= 0.5 exactly when z >= 0, the decision boundary is
w^T x + b = 0, a straight line (a hyperplane): logistic regression is a
LINEAR classifier.

----------------------------------------------------------------------
DERIVATIVE OF THE SIGMOID
----------------------------------------------------------------------
    sigma(z)  = (1 + e^{-z})^{-1}
    sigma'(z) = -(1 + e^{-z})^{-2} * (-e^{-z})
              = e^{-z} / (1 + e^{-z})^2
              = [1/(1+e^{-z})] * [e^{-z}/(1+e^{-z})]
              = sigma(z) * (1 - sigma(z))

because e^{-z}/(1+e^{-z}) = 1 - sigma(z).

----------------------------------------------------------------------
THE LOSS: BINARY CROSS-ENTROPY (with an L2 penalty)
----------------------------------------------------------------------
    L(w,b) = -(1/N) sum_i [ y_i log(p_i) + (1-y_i) log(1-p_i) ] + lambda*||w||^2

Only one of the two terms is active per sample, because y is 0 or 1:
    y = 1 -> the loss is -log(p),     small when p is close to 1
    y = 0 -> the loss is -log(1-p),   small when p is close to 0
The L2 term is the same idea as in ridge regression: it keeps the weights
small. The bias b is NOT penalised - shrinking it would only pull the baseline
prediction towards 0.5, which has nothing to do with overfitting.

----------------------------------------------------------------------
THE GRADIENT (chain rule, exactly as in the lectures)
----------------------------------------------------------------------
The chain is    w -> z -> p -> L    so    dL/dw = (dL/dp)(dp/dz)(dz/dw).

    dL/dp = -(y - p) / [p(1-p)]        (differentiate the two log terms)
    dp/dz = sigma'(z) = p(1-p)         (result above)
    dz/dw = x

Multiplying the first two, the factor p(1-p) CANCELS and leaves simply

    dL/dz = p - y        <- prediction minus target

so, averaging over the N samples,

    dL/dw = (1/N) X^T (p - y) + 2*lambda*w
    dL/db = (1/N) sum_i (p_i - y_i)

This cancellation is why we use cross-entropy (and not squared error) with a
sigmoid output: there is no leftover sigma' factor, which would otherwise make
the gradient vanish exactly when the model is confidently wrong.

----------------------------------------------------------------------
TRAINING: GRADIENT DESCENT
----------------------------------------------------------------------
    w <- w - eta * dL/dw
    b <- b - eta * dL/db

repeated for a fixed number of iterations; eta is the learning rate.

----------------------------------------------------------------------
EVALUATION
----------------------------------------------------------------------
From the confusion matrix (TP, FP, FN, TN):
    Accuracy    = (TP + TN) / (TP + TN + FP + FN)
    Precision   = TP / (TP + FP)
    Recall      = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    F1          = 2 * Precision * Recall / (Precision + Recall)
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


# ======================================================================
# THE MATHEMATICS
# ======================================================================
def sigmoid(z):
    """sigma(z) = 1 / (1 + e^{-z}).

    Two branches only to avoid overflow: e^{-z} explodes for very negative z,
    so there we use the algebraically identical form e^z / (1 + e^z).
    """
    out = np.empty_like(z, dtype=float)
    pos, neg = z >= 0, z < 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[neg])
    out[neg] = e / (1.0 + e)
    return out


def predict_proba(X, w, b):
    """Forward pass: score z = Xw + b, then probability p = sigma(z)."""
    return sigmoid(X @ w + b)


def loss(X, y, w, b, lam):
    """Binary cross-entropy + L2 penalty on the weights (not on the bias)."""
    p = predict_proba(X, w, b)
    eps = 1e-12                                   # keeps log(0) from appearing
    bce = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    return float(bce + lam * np.sum(w ** 2))


def gradient(X, y, w, b, lam):
    """dL/dw = (1/N) X^T (p - y) + 2*lambda*w ;   dL/db = mean(p - y)."""
    N = X.shape[0]
    p = predict_proba(X, w, b)
    error = p - y                                 # this IS dL/dz
    grad_w = (X.T @ error) / N + 2 * lam * w
    grad_b = np.mean(error)
    return grad_w, grad_b


def train_gradient_descent(X, y, lam, eta, n_iter):
    """w <- w - eta*dL/dw and b <- b - eta*dL/db, repeated n_iter times."""
    w = np.zeros(X.shape[1])                      # start from all zeros
    b = 0.0
    history = [loss(X, y, w, b, lam)]
    for _ in range(n_iter):
        grad_w, grad_b = gradient(X, y, w, b, lam)
        w = w - eta * grad_w
        b = b - eta * grad_b
        history.append(loss(X, y, w, b, lam))
    return w, b, np.array(history)


# ======================================================================
# EVALUATION
# ======================================================================
def confusion_counts(y_true, y_pred):
    """TP, FP, FN, TN for the positive class (y = 1)."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return tp, fp, fn, tn


def metrics(y_true, p, threshold=0.5):
    """The standard classification metrics, all from the confusion matrix."""
    y_pred = (p >= threshold).astype(int)
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": accuracy,
            "precision": precision, "recall": recall,
            "specificity": specificity, "f1": f1}


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", type=float, default=0.5, help="learning rate")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--lam", type=float, default=1e-3, help="L2 strength")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(RESDIR, exist_ok=True)

    # ---------------- data ------------------------------------------------
    data = load_breast_cancer()
    X, y = data.data, data.target.astype(float)
    names = list(data.feature_names)

    print("=" * 72)
    print("04 | LOGISTIC REGRESSION  -  Breast Cancer Wisconsin (diagnostic)")
    print("=" * 72)
    print(f"Samples N = {X.shape[0]}, features p = {X.shape[1]}")
    print(f"Classes: {data.target_names[0]} = 0 ({int((y == 0).sum())} samples), "
          f"{data.target_names[1]} = 1 ({int((y == 1).sum())} samples)")
    print()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=args.seed, stratify=y)

    # Standardise: (x - mean) / std, using TRAINING statistics only, so that no
    # information from the test set leaks into the model.
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd
    print(f"Train / test split : {len(y_tr)} / {len(y_te)} (stratified)")
    print("Features standardised with the training mean and standard deviation.")
    print()

    # ---------------- check the sigmoid derivative ------------------------
    # Definition of a derivative: f'(z) = lim_{h->0} [f(z+h) - f(z-h)] / 2h.
    # Compare that slope with the formula sigma'(z) = sigma(z)(1 - sigma(z)).
    print("-" * 72)
    print("(a) CHECK:  sigma'(z) = sigma(z)(1 - sigma(z))")
    print("-" * 72)
    zz = np.array([-2.0, -0.5, 0.0, 0.7, 3.0])
    numerical = (sigmoid(zz + 1e-6) - sigmoid(zz - 1e-6)) / 2e-6
    formula = sigmoid(zz) * (1 - sigmoid(zz))
    print(f"{'z':>8}{'numerical slope':>18}{'sigma(1-sigma)':>18}")
    for a, n_, f_ in zip(zz, numerical, formula):
        print(f"{a:>8.1f}{n_:>18.8f}{f_:>18.8f}")
    print(f"largest difference = {np.max(np.abs(numerical - formula)):.3e}")
    print()

    # ---------------- training --------------------------------------------
    print("-" * 72)
    print(f"(b) TRAINING BY GRADIENT DESCENT "
          f"(eta = {args.eta}, iterations = {args.iters}, lambda = {args.lam})")
    print("-" * 72)
    w, b, history = train_gradient_descent(
        X_tr, y_tr, args.lam, args.eta, args.iters)

    print(f"{'iteration':>10}{'loss':>16}")
    for k in [0, 1, 10, 100, 500, 1000, 2000, args.iters]:
        print(f"{k:>10}{history[k]:>16.8f}")
    print()
    print("At iteration 0 all weights are zero, so p = sigma(0) = 0.5 for every")
    print(f"sample and the loss is -log(0.5) = log 2 = {np.log(2):.6f}.")
    grad_w, grad_b = gradient(X_tr, y_tr, w, b, args.lam)
    print(f"Size of the final gradient ||dL/dw|| = {np.linalg.norm(grad_w):.3e} "
          "(near zero = we have reached the minimum)")
    print()

    print("Ten largest weights (on standardised features):")
    order = np.argsort(-np.abs(w))[:10]
    for j in order:
        print(f"   {names[j]:>26} : {w[j]:+8.4f}")
    print(f"   {'bias b':>26} : {b:+8.4f}")
    print()

    # ---------------- evaluation ------------------------------------------
    p_tr = predict_proba(X_tr, w, b)
    p_te = predict_proba(X_te, w, b)
    m_tr = metrics(y_tr, p_tr)
    m_te = metrics(y_te, p_te)

    print("-" * 72)
    print("(c) PERFORMANCE  (threshold 0.5)")
    print("-" * 72)
    print(f"{'':<8}{'loss':>10}{'accuracy':>11}{'precision':>11}"
          f"{'recall':>9}{'F1':>9}{'specificity':>13}")
    for tag, m, Xs, ys in (("train", m_tr, X_tr, y_tr), ("test", m_te, X_te, y_te)):
        print(f"{tag:<8}{loss(Xs, ys, w, b, 0.0):>10.5f}{m['accuracy']:>11.4f}"
              f"{m['precision']:>11.4f}{m['recall']:>9.4f}{m['f1']:>9.4f}"
              f"{m['specificity']:>13.4f}")
    print()
    print("Test confusion matrix (rows = true, columns = predicted):")
    print(f"{'':>12}{'pred 0':>10}{'pred 1':>10}")
    print(f"{'true 0':>12}{m_te['tn']:>10}{m_te['fp']:>10}")
    print(f"{'true 1':>12}{m_te['fn']:>10}{m_te['tp']:>10}")
    print()

    # ---------------- plots -----------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history, lw=1.8)
    ax[0].axhline(np.log(2), color="gray", ls=":", lw=1,
                  label=r"$\log 2$ (all predictions $=0.5$)")
    ax[0].set_xlabel("gradient descent iteration")
    ax[0].set_ylabel("binary cross-entropy")
    ax[0].set_title(rf"Training loss, $\eta = {args.eta}$")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    zs = np.linspace(-8, 8, 400)
    ax[1].plot(zs, sigmoid(zs), lw=1.8, label=r"$\sigma(z)$")
    ax[1].plot(zs, sigmoid(zs) * (1 - sigmoid(zs)), lw=1.8,
               label=r"$\sigma'(z) = \sigma(z)(1-\sigma(z))$")
    ax[1].axhline(0.5, color="gray", ls=":", lw=1)
    ax[1].set_xlabel("$z$")
    ax[1].set_title("Sigmoid and its derivative")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    f1p = os.path.join(FIGDIR, "04_breast_cancer_training.pdf")
    fig.savefig(f1p)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    M = np.array([[m_te["tn"], m_te["fp"]], [m_te["fn"], m_te["tp"]]])
    im = ax[0].imshow(M, cmap="Blues")
    ax[0].set_xticks([0, 1], ["predicted 0", "predicted 1"])
    ax[0].set_yticks([0, 1], ["true 0", "true 1"])
    for i in range(2):
        for j in range(2):
            ax[0].text(j, i, M[i, j], ha="center", va="center",
                       color="white" if M[i, j] > M.max() / 2 else "black")
    ax[0].set_title(f"Test confusion matrix (accuracy = {m_te['accuracy']:.4f})")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].hist(p_te[y_te == 0], bins=25, alpha=0.7, label="true class 0 (malignant)")
    ax[1].hist(p_te[y_te == 1], bins=25, alpha=0.7, label="true class 1 (benign)")
    ax[1].axvline(0.5, color="k", ls="--", lw=1, label="threshold 0.5")
    ax[1].set_xlabel(r"predicted probability $p$")
    ax[1].set_ylabel("count")
    ax[1].set_title("Predicted probabilities by true class")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    f2p = os.path.join(FIGDIR, "04_breast_cancer_results.pdf")
    fig.savefig(f2p)
    plt.close(fig)

    with open(os.path.join(RESDIR, "04_breast_cancer.json"), "w") as fh:
        json.dump({"train": m_tr, "test": m_te,
                   "train_loss": loss(X_tr, y_tr, w, b, 0.0),
                   "test_loss": loss(X_te, y_te, w, b, 0.0),
                   "final_training_loss": float(history[-1]),
                   "iterations": args.iters, "eta": args.eta, "lam": args.lam,
                   "final_grad_norm": float(np.linalg.norm(grad_w)),
                   "loss_at": {str(k): float(history[k])
                               for k in [0, 1, 10, 100, 500, 1000, 2000, args.iters]},
                   "weights": {n: float(v) for n, v in zip(names, w)},
                   "bias": float(b)}, fh, indent=2)
    print(f"Figures saved: {f1p}")
    print(f"               {f2p}")
    print(f"Results saved: {os.path.join(RESDIR, '04_breast_cancer.json')}")


if __name__ == "__main__":
    main()
