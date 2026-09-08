"""
05 - Neural network: a fully connected network (MLP) written from scratch in
     NumPy and trained on Fashion-MNIST (Zalando Research, 10 clothing classes).

The four IDX files are downloaded from the official GitHub repository and cached
in ./data.

ARCHITECTURE
------------
    784 -> 256 (ReLU) -> 128 (ReLU) -> 10 (softmax)

MATHEMATICS IMPLEMENTED HERE  (lecture 3-4 "Mathematical Steps in Neural
Networks" and lecture 7-8 chain rule)
---------------------------------------------------------------------------
Forward pass, for layer l = 1..L:
    z[l] = a[l-1] W[l] + b[l]          (a[0] = x, rows are samples)
    a[l] = f(z[l]),   f = ReLU for hidden layers
    yhat = a[L] = softmax(z[L])

Loss (categorical cross-entropy over a mini-batch of size B):
    L = -(1/B) sum_i sum_c y_ic log yhat_ic

Backward pass. Write delta[l] = dL/dz[l]. At the output, the softmax +
cross-entropy pair collapses to
    delta[L] = (yhat - y) / B
and the chain rule propagates it backwards:
    dL/dW[l] = a[l-1]^T delta[l]
    dL/db[l] = 1^T delta[l]
    delta[l-1] = (delta[l] W[l]^T) * f'(z[l-1])
with the ReLU derivative
    f'(z) = 1 if z > 0, else 0.

Parameter update - mini-batch SGD with momentum:
    v <- mu v - eta dL/dtheta,     theta <- theta + v
Momentum accumulates an exponentially weighted average of past gradients, which
damps oscillations in the high-curvature directions of the loss surface.

Initialisation: He scaling W ~ N(0, 2/n_in) keeps the variance of the
activations roughly constant across ReLU layers.

Correctness of the analytic gradients is verified against central finite
differences before training.
"""

import argparse
import gzip
import json
import os
import time
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
RESDIR = os.path.join(HERE, "results")
DATADIR = os.path.join(HERE, "data", "fashion_mnist")
BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {
    "train_X": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_X": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def _download(fname):
    os.makedirs(DATADIR, exist_ok=True)
    path = os.path.join(DATADIR, fname)
    if not os.path.exists(path):
        print(f"  downloading {fname} ...")
        urllib.request.urlretrieve(BASE + fname, path)
    return path


def load_fashion_mnist():
    out = {}
    for key, fname in FILES.items():
        path = _download(fname)
        with gzip.open(path, "rb") as fh:
            raw = fh.read()
        if "X" in key:
            out[key] = np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 784)
        else:
            out[key] = np.frombuffer(raw, np.uint8, offset=8)
    return (out["train_X"].astype(np.float64) / 255.0, out["train_y"].astype(int),
            out["test_X"].astype(np.float64) / 255.0, out["test_y"].astype(int))


# ----------------------------------------------------------------------
# Core mathematics
# ----------------------------------------------------------------------
def relu(z):
    return np.maximum(0.0, z)


def relu_grad(z):
    return (z > 0).astype(z.dtype)


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def one_hot(y, C=10):
    Y = np.zeros((len(y), C))
    Y[np.arange(len(y)), y] = 1.0
    return Y


class MLP:
    """Fully connected network with ReLU hidden layers and a softmax output."""

    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.W = [rng.standard_normal((sizes[i], sizes[i + 1])) * np.sqrt(2.0 / sizes[i])
                  for i in range(len(sizes) - 1)]           # He initialisation
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(v) for v in self.b]

    # ---- forward ------------------------------------------------------
    def forward(self, X):
        a = X
        cache = {"a": [X], "z": []}
        L = len(self.W)
        for l in range(L):
            z = a @ self.W[l] + self.b[l]
            a = softmax(z) if l == L - 1 else relu(z)
            cache["z"].append(z)
            cache["a"].append(a)
        return a, cache

    # ---- loss ---------------------------------------------------------
    @staticmethod
    def loss(P, Y, W=None, lam=0.0):
        L = -np.sum(Y * np.log(P + 1e-12)) / Y.shape[0]
        if W is not None and lam > 0:
            L += lam * sum(np.sum(w * w) for w in W)
        return float(L)

    # ---- backward -----------------------------------------------------
    def backward(self, cache, Y, lam=0.0):
        B = Y.shape[0]
        L = len(self.W)
        gW = [None] * L
        gb = [None] * L
        delta = (cache["a"][-1] - Y) / B          # dL/dz at the output layer
        for l in range(L - 1, -1, -1):
            gW[l] = cache["a"][l].T @ delta + 2 * lam * self.W[l]
            gb[l] = delta.sum(axis=0)
            if l > 0:
                delta = (delta @ self.W[l].T) * relu_grad(cache["z"][l - 1])
        return gW, gb

    # ---- update -------------------------------------------------------
    def step(self, gW, gb, eta, mu=0.9):
        for l in range(len(self.W)):
            self.vW[l] = mu * self.vW[l] - eta * gW[l]
            self.vb[l] = mu * self.vb[l] - eta * gb[l]
            self.W[l] += self.vW[l]
            self.b[l] += self.vb[l]

    def predict(self, X, batch=2000):
        out = []
        for i in range(0, len(X), batch):
            P, _ = self.forward(X[i:i + batch])
            out.append(P)
        return np.vstack(out)


def gradient_check(net, X, Y, lam, n_checks=12, eps=1e-6, seed=0):
    """Compare analytic dL/dW[l] entries against central finite differences."""
    rng = np.random.default_rng(seed)
    P, cache = net.forward(X)
    gW, _ = net.backward(cache, Y, lam)
    errs = []
    for _ in range(n_checks):
        l = rng.integers(len(net.W))
        i = rng.integers(net.W[l].shape[0])
        j = rng.integers(net.W[l].shape[1])
        old = net.W[l][i, j]
        net.W[l][i, j] = old + eps
        Lp = net.loss(net.forward(X)[0], Y, net.W, lam)
        net.W[l][i, j] = old - eps
        Lm = net.loss(net.forward(X)[0], Y, net.W, lam)
        net.W[l][i, j] = old
        num = (Lp - Lm) / (2 * eps)
        errs.append(abs(num - gW[l][i, j]) / max(1e-12, abs(num) + abs(gW[l][i, j])))
    return float(np.max(errs))


def confusion(y, yhat, C=10):
    M = np.zeros((C, C), dtype=int)
    for t, p in zip(y, yhat):
        M[t, p] += 1
    return M


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--lam", type=float, default=1e-5)
    ap.add_argument("--decay", type=float, default=0.92, help="per-epoch lr decay")
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(RESDIR, exist_ok=True)

    print("=" * 72)
    print("05 | NEURAL NETWORK FROM SCRATCH  -  Fashion-MNIST")
    print("=" * 72)
    Xtr, ytr, Xte, yte = load_fashion_mnist()
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(Xtr))
    n_val = 6000
    val, tr = perm[:n_val], perm[n_val:]
    Xva, yva, Xtr, ytr = Xtr[val], ytr[val], Xtr[tr], ytr[tr]
    Ytr, Yva, Yte = one_hot(ytr), one_hot(yva), one_hot(yte)

    sizes = [784] + list(args.hidden) + [10]
    n_par = sum(s * t for s, t in zip(sizes[:-1], sizes[1:])) + sum(sizes[1:])
    print(f"Train / validation / test : {len(Xtr)} / {len(Xva)} / {len(Xte)}")
    print(f"Input dimension 28x28 = 784, pixel values scaled to [0,1]")
    print(f"Architecture : {' -> '.join(map(str, sizes))}  (ReLU hidden, softmax output)")
    print(f"Parameters   : {n_par:,}")
    print(f"Optimiser    : mini-batch SGD, batch = {args.batch}, eta0 = {args.eta}, "
          f"momentum = {args.momentum}, decay = {args.decay}/epoch, L2 lambda = {args.lam}")
    print()

    net = MLP(sizes, seed=args.seed)

    print("-" * 72)
    print("(a) GRADIENT CHECK on a mini-batch of 32 samples")
    print("-" * 72)
    rel = gradient_check(net, Xtr[:32], Ytr[:32], args.lam, seed=args.seed)
    print(f"max relative error between backprop and finite differences = {rel:.3e}")
    print("(a relative error of order 1e-5 or smaller confirms the backpropagation")
    print(" equations; ReLU kinks - points where a unit changes sign under the")
    print(" perturbation - keep the finite-difference estimate from being exact)")
    print()

    print("-" * 72)
    print("(b) TRAINING")
    print("-" * 72)
    print(f"{'epoch':>6}{'train loss':>13}{'train acc':>11}{'val loss':>11}{'val acc':>10}{'eta':>9}{'sec':>8}")
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    eta = args.eta
    n = len(Xtr)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        order = rng.permutation(n)
        run_loss, run_correct, seen = 0.0, 0, 0
        for s in range(0, n, args.batch):
            idx = order[s:s + args.batch]
            xb, yb = Xtr[idx], Ytr[idx]
            P, cache = net.forward(xb)
            gW, gb = net.backward(cache, yb, args.lam)
            net.step(gW, gb, eta, args.momentum)
            run_loss += net.loss(P, yb) * len(idx)
            run_correct += int(np.sum(np.argmax(P, 1) == ytr[idx]))
            seen += len(idx)
        Pva = net.predict(Xva)
        vl = net.loss(Pva, Yva)
        va = float(np.mean(np.argmax(Pva, 1) == yva))
        hist["train_loss"].append(run_loss / seen)
        hist["train_acc"].append(run_correct / seen)
        hist["val_loss"].append(vl)
        hist["val_acc"].append(va)
        print(f"{ep:>6}{run_loss/seen:13.5f}{run_correct/seen:11.4f}"
              f"{vl:11.5f}{va:10.4f}{eta:9.4f}{time.time()-t0:8.1f}")
        eta *= args.decay
    print()

    # --- evaluation -------------------------------------------------------
    Pte = net.predict(Xte)
    yhat = np.argmax(Pte, 1)
    acc = float(np.mean(yhat == yte))
    M = confusion(yte, yhat)
    print("-" * 72)
    print("(c) TEST PERFORMANCE")
    print("-" * 72)
    print(f"Test cross-entropy = {net.loss(Pte, Yte):.5f}")
    print(f"Test accuracy      = {acc:.4f}   (error rate {1-acc:.4f})")
    print("\nPer-class precision / recall / F1:")
    per = {}
    for i, c in enumerate(CLASSES):
        tp = M[i, i]
        prec = tp / M[:, i].sum() if M[:, i].sum() else 0.0
        rec = tp / M[i].sum() if M[i].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"precision": float(prec), "recall": float(rec), "f1": float(f1)}
        print(f"   {c:>12}: precision = {prec:.4f}, recall = {rec:.4f}, F1 = {f1:.4f}")
    off = M - np.diag(np.diag(M))
    i, j = np.unravel_index(np.argmax(off), off.shape)
    print(f"\nMost frequent confusion: true '{CLASSES[i]}' predicted as "
          f"'{CLASSES[j]}' ({off[i, j]} times)")
    print()

    # --- plots ------------------------------------------------------------
    ep = np.arange(1, args.epochs + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, hist["train_loss"], "o-", ms=3, label="train")
    ax[0].plot(ep, hist["val_loss"], "s-", ms=3, label="validation")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("cross-entropy")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(ep, hist["train_acc"], "o-", ms=3, label="train")
    ax[1].plot(ep, hist["val_acc"], "s-", ms=3, label="validation")
    ax[1].axhline(acc, color="crimson", ls="--", lw=1, label=f"test = {acc:.4f}")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    f1p = os.path.join(FIGDIR, "05_fashion_mnist_curves.pdf")
    fig.savefig(f1p)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(10), CLASSES, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(10), CLASSES, fontsize=7)
    for a in range(10):
        for b in range(10):
            ax.text(b, a, M[a, b], ha="center", va="center", fontsize=5.5,
                    color="white" if M[a, b] > M.max() / 2 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Fashion-MNIST test confusion matrix (acc = {acc:.4f})")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    f2p = os.path.join(FIGDIR, "05_fashion_mnist_confusion.pdf")
    fig.savefig(f2p)
    plt.close(fig)

    fig, axes = plt.subplots(3, 6, figsize=(9, 5))
    for k, a in enumerate(axes.ravel()):
        a.imshow(Xte[k].reshape(28, 28), cmap="gray")
        ok = yhat[k] == yte[k]
        a.set_title(f"{CLASSES[yhat[k]]}\n({CLASSES[yte[k]]})", fontsize=6,
                    color="green" if ok else "red")
        a.axis("off")
    fig.suptitle("Predictions (true class in brackets)", fontsize=10)
    fig.tight_layout()
    f3p = os.path.join(FIGDIR, "05_fashion_mnist_samples.pdf")
    fig.savefig(f3p)
    plt.close(fig)

    fig, axes = plt.subplots(4, 8, figsize=(9, 5))
    W1 = net.W[0]
    for k, a in enumerate(axes.ravel()):
        a.imshow(W1[:, k].reshape(28, 28), cmap="RdBu_r")
        a.axis("off")
    fig.suptitle(r"First-layer weight vectors $W^{[1]}_{:,j}$ as $28\times28$ images",
                 fontsize=10)
    fig.tight_layout()
    f4p = os.path.join(FIGDIR, "05_fashion_mnist_filters.pdf")
    fig.savefig(f4p)
    plt.close(fig)

    with open(os.path.join(RESDIR, "05_fashion_mnist.json"), "w") as fh:
        json.dump({"architecture": sizes, "parameters": int(n_par),
                   "epochs": args.epochs, "batch": args.batch, "eta0": args.eta,
                   "momentum": args.momentum, "lam": args.lam, "decay": args.decay,
                   "grad_check_rel_err": rel,
                   "test_acc": acc, "test_ce": net.loss(Pte, Yte),
                   "final_train_acc": hist["train_acc"][-1],
                   "final_val_acc": hist["val_acc"][-1],
                   "history": hist, "confusion": M.tolist(), "per_class": per,
                   "worst_confusion": [CLASSES[i], CLASSES[j], int(off[i, j])]},
                  fh, indent=2)
    for f in (f1p, f2p, f3p, f4p):
        print(f"Figure saved: {f}")
    print(f"Results saved: {os.path.join(RESDIR, '05_fashion_mnist.json')}")


if __name__ == "__main__":
    main()
