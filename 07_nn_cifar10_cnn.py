"""
07 - Neural network: a CONVOLUTIONAL neural network (CNN) written from scratch
     in NumPy and trained on CIFAR-10 (University of Toronto, 10 classes of
     32x32 colour images).

Data: the script first tries the official CIFAR-10 python archive from
cs.toronto.edu; if that host is unreachable it falls back to a GitHub mirror of
the same images (requires `pillow`). Either way the arrays are cached as an
.npz in ./data so later runs are instant.

ARCHITECTURE
------------
    input 3x32x32
      conv1 : 16 filters 3x3, pad 1  -> ReLU -> maxpool 2x2   (16x16x16)
      conv2 : 32 filters 3x3, pad 1  -> ReLU -> maxpool 2x2   (32x8x8)
      flatten (2048) -> dense 128 -> ReLU -> dense 10 -> softmax

MATHEMATICS IMPLEMENTED HERE
----------------------------
1. The convolution (strictly, cross-correlation) of an input a with a filter W:

       z[f, i, j] = sum_{c, u, v} W[f, c, u, v] * a[c, i+u, j+v] + b[f]

   Compare with the dense layer z = W a + b of script 05. A convolution IS a
   dense layer whose weight matrix is constrained in two ways:
     * sparsity   - z[f,i,j] depends only on a small patch of a, not on all of it;
     * weight tying - the SAME W[f] is used at every spatial position (i, j).
   Consequences: the layer is equivariant to translation (shifting the input
   shifts the output), and the parameter count no longer grows with image size.
   Here conv1 has 16*3*3*3 + 16 = 448 parameters; a dense layer with the same
   3072 inputs and 16384 outputs would need 5.0e+7.

2. im2col. Writing the patch centred on each output position as a row of a
   matrix "col" of shape (N*H_out*W_out, C*k*k) turns the convolution into one
   matrix product

       Z = col @ W_row^T + b,        W_row of shape (F, C*k*k)

   which is exactly the dense layer of script 05. Every gradient below then
   follows from the dense-layer rules already derived.

3. Backward pass through the convolution. With delta = dL/dZ (reshaped to the
   same (N*H_out*W_out, F) layout),

       dL/dW_row = delta^T @ col
       dL/db     = column sums of delta
       dL/dcol   = delta @ W_row

   and dL/dcol is scattered back to image coordinates by col2im. Because weight
   tying means one weight is used at many positions, dL/dW_row is a SUM over all
   positions and all samples - the multivariable chain rule again, exactly as in
   the recurrent network of script 06 where the sum runs over time instead of
   space. Overlapping patches also make col2im ACCUMULATE (+=) rather than
   assign: a pixel used by several output positions receives the sum of their
   gradients.

4. Max pooling: y = max over each 2x2 window. The derivative routes the incoming
   gradient to the arg-max position only and sends zero elsewhere,

       dL/da[c,i,j] = dL/dy[c,p,q]  if (i,j) = argmax of window (p,q), else 0.

   Pooling has no parameters; it discards spatial precision in exchange for a
   larger receptive field and fewer downstream activations.

5. Output layer and loss are unchanged from script 05: softmax with categorical
   cross-entropy, so delta at the output is again (P - Y)/B.

The analytic gradients of every layer are verified against central finite
differences before training.
"""

import argparse
import json
import os
import pickle
import subprocess
import tarfile
import time
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
RESDIR = os.path.join(HERE, "results")
DATADIR = os.path.join(HERE, "data")
CACHE = os.path.join(DATADIR, "cifar10.npz")
OFFICIAL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
MIRROR = "https://github.com/YoongiKim/CIFAR-10-images.git"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def _from_official():
    tgz = os.path.join(DATADIR, "cifar-10-python.tar.gz")
    if not os.path.exists(tgz):
        print(f"  trying official source {OFFICIAL} ...")
        urllib.request.urlretrieve(OFFICIAL, tgz)
    with tarfile.open(tgz) as tf:
        def rd(name):
            return pickle.load(tf.extractfile(name), encoding="bytes")
        Xtr, ytr = [], []
        for k in range(1, 6):
            d = rd(f"cifar-10-batches-py/data_batch_{k}")
            Xtr.append(d[b"data"])
            ytr += d[b"labels"]
        d = rd("cifar-10-batches-py/test_batch")
    Xtr = np.concatenate(Xtr).reshape(-1, 3, 32, 32)
    Xte = d[b"data"].reshape(-1, 3, 32, 32)
    return Xtr, np.array(ytr), Xte, np.array(d[b"labels"])


def _from_mirror():
    import glob
    import matplotlib.image as mpimg
    repo = os.path.join(DATADIR, "cifar10_png")
    if not os.path.isdir(repo):
        print(f"  official host unreachable; cloning mirror {MIRROR} ...")
        subprocess.run(["git", "clone", "--depth", "1", "-q", MIRROR, repo], check=True)
    out = {}
    for split in ("train", "test"):
        X, y = [], []
        for c, name in enumerate(CLASSES):
            files = sorted(glob.glob(os.path.join(repo, split, name, "*")))
            for f in files:
                X.append(mpimg.imread(f))
                y.append(c)
        out[split] = (np.array(X).transpose(0, 3, 1, 2), np.array(y))
    return out["train"][0], out["train"][1], out["test"][0], out["test"][1]


def load_cifar10():
    os.makedirs(DATADIR, exist_ok=True)
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z["Xtr"], z["ytr"], z["Xte"], z["yte"]
    try:
        Xtr, ytr, Xte, yte = _from_official()
    except Exception as exc:                       # blocked host, no internet, ...
        print(f"  official download failed ({type(exc).__name__}); using mirror")
        Xtr, ytr, Xte, yte = _from_mirror()
    np.savez_compressed(CACHE, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    return Xtr, ytr, Xte, yte


# ----------------------------------------------------------------------
# Core mathematics
# ----------------------------------------------------------------------
def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def one_hot(y, C=10):
    Y = np.zeros((len(y), C))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def im2col(X, k, pad):
    """(N,C,H,W) -> (N*Ho*Wo, C*k*k) with each row one receptive field."""
    N, C, H, W = X.shape
    Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Ho, Wo = H + 2 * pad - k + 1, W + 2 * pad - k + 1
    s = Xp.strides
    view = np.lib.stride_tricks.as_strided(
        Xp, shape=(N, C, Ho, Wo, k, k),
        strides=(s[0], s[1], s[2], s[3], s[2], s[3]), writeable=False)
    return view.transpose(0, 2, 3, 1, 4, 5).reshape(N * Ho * Wo, C * k * k), Ho, Wo


def col2im(cols, shape, k, pad, Ho, Wo):
    """Adjoint of im2col: scatter-ADD the patch gradients back to image pixels."""
    N, C, H, W = shape
    Xp = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    c = cols.reshape(N, Ho, Wo, C, k, k).transpose(0, 3, 4, 5, 1, 2)
    for u in range(k):
        for v in range(k):
            Xp[:, :, u:u + Ho, v:v + Wo] += c[:, :, u, v]
    return Xp[:, :, pad:pad + H, pad:pad + W] if pad else Xp


class Conv:
    def __init__(self, C_in, F, k, pad, rng):
        self.k, self.pad, self.F = k, pad, F
        fan_in = C_in * k * k
        self.W = rng.standard_normal((F, fan_in)) * np.sqrt(2.0 / fan_in)  # He init
        self.b = np.zeros(F)
        self.vW, self.vb = np.zeros_like(self.W), np.zeros_like(self.b)

    def forward(self, X):
        self.shape = X.shape
        cols, Ho, Wo = im2col(X, self.k, self.pad)
        self.cols, self.Ho, self.Wo = cols, Ho, Wo
        Z = cols @ self.W.T + self.b                       # (N*Ho*Wo, F)
        return Z.reshape(X.shape[0], Ho, Wo, self.F).transpose(0, 3, 1, 2)

    def backward(self, dZ, lam=0.0):
        d = dZ.transpose(0, 2, 3, 1).reshape(-1, self.F)
        self.gW = d.T @ self.cols + 2 * lam * self.W       # summed over positions
        self.gb = d.sum(axis=0)
        dcols = d @ self.W
        return col2im(dcols, self.shape, self.k, self.pad, self.Ho, self.Wo)

    def step(self, eta, mu):
        self.vW = mu * self.vW - eta * self.gW
        self.vb = mu * self.vb - eta * self.gb
        self.W += self.vW
        self.b += self.vb


class Dense:
    def __init__(self, n_in, n_out, rng):
        self.W = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
        self.b = np.zeros(n_out)
        self.vW, self.vb = np.zeros_like(self.W), np.zeros_like(self.b)

    def forward(self, A):
        self.A = A
        return A @ self.W + self.b

    def backward(self, dZ, lam=0.0):
        self.gW = self.A.T @ dZ + 2 * lam * self.W
        self.gb = dZ.sum(axis=0)
        return dZ @ self.W.T

    def step(self, eta, mu):
        self.vW = mu * self.vW - eta * self.gW
        self.vb = mu * self.vb - eta * self.gb
        self.W += self.vW
        self.b += self.vb


def maxpool_forward(X, k=2):
    N, C, H, W = X.shape
    Xr = X.reshape(N, C, H // k, k, W // k, k)
    out = Xr.max(axis=(3, 5))
    mask = (Xr == out[:, :, :, None, :, None])            # arg-max indicator
    return out, mask


def maxpool_backward(dout, mask, k=2):
    N, C, Ho, Wo = dout.shape
    d = mask * dout[:, :, :, None, :, None]               # route to the arg-max
    d = d / np.maximum(1, mask.sum(axis=(3, 5))[:, :, :, None, :, None])  # split ties
    return d.reshape(N, C, Ho * k, Wo * k)


class CNN:
    def __init__(self, seed=0, f1=16, f2=32, hidden=128):
        rng = np.random.default_rng(seed)
        self.c1 = Conv(3, f1, 3, 1, rng)
        self.c2 = Conv(f1, f2, 3, 1, rng)
        self.d1 = Dense(f2 * 8 * 8, hidden, rng)
        self.d2 = Dense(hidden, 10, rng)
        self.layers = [self.c1, self.c2, self.d1, self.d2]

    def forward(self, X):
        self.z1 = self.c1.forward(X)
        self.a1 = np.maximum(0.0, self.z1)
        self.p1, self.m1 = maxpool_forward(self.a1)
        self.z2 = self.c2.forward(self.p1)
        self.a2 = np.maximum(0.0, self.z2)
        self.p2, self.m2 = maxpool_forward(self.a2)
        self.flat = self.p2.reshape(len(X), -1)
        self.z3 = self.d1.forward(self.flat)
        self.a3 = np.maximum(0.0, self.z3)
        return softmax(self.d2.forward(self.a3))

    def backward(self, P, Y, lam=0.0):
        B = len(Y)
        d = (P - Y) / B                                    # softmax + CCE
        d = self.d2.backward(d, lam)
        d = d * (self.z3 > 0)
        d = self.d1.backward(d, lam).reshape(self.p2.shape)
        d = maxpool_backward(d, self.m2)
        d = d * (self.z2 > 0)
        d = self.c2.backward(d, lam)
        d = maxpool_backward(d, self.m1)
        d = d * (self.z1 > 0)
        self.c1.backward(d, lam)

    def step(self, eta, mu):
        for l in self.layers:
            l.step(eta, mu)

    def predict(self, X, batch=500):
        return np.vstack([self.forward(X[i:i + batch]) for i in range(0, len(X), batch)])


def loss_fn(P, Y):
    return float(-np.sum(Y * np.log(P + 1e-12)) / len(Y))


def gradient_check(net, X, Y, eps=1e-5, n=10, seed=0):
    rng = np.random.default_rng(seed)
    P = net.forward(X)
    net.backward(P, Y)
    errs = {}
    for name, layer in [("conv1", net.c1), ("conv2", net.c2),
                        ("dense1", net.d1), ("dense2", net.d2)]:
        e = []
        for _ in range(n):
            i = rng.integers(layer.W.shape[0])
            j = rng.integers(layer.W.shape[1])
            old = layer.W[i, j]
            layer.W[i, j] = old + eps
            lp = loss_fn(net.forward(X), Y)
            layer.W[i, j] = old - eps
            lm = loss_fn(net.forward(X), Y)
            layer.W[i, j] = old
            num = (lp - lm) / (2 * eps)
            ana = layer.gW[i, j]
            e.append(abs(num - ana) / max(1e-12, abs(num) + abs(ana)))
        errs[name] = float(np.max(e))
    return errs


def confusion(y, yhat, C=10):
    M = np.zeros((C, C), dtype=int)
    for t, p in zip(y, yhat):
        M[t, p] += 1
    return M


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--lam", type=float, default=1e-5)
    ap.add_argument("--decay", type=float, default=0.92)
    ap.add_argument("--n-train", type=int, default=12000,
                    help="training subset size (50000 = full set, much slower)")
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(RESDIR, exist_ok=True)

    print("=" * 72)
    print("07 | CONVOLUTIONAL NEURAL NETWORK FROM SCRATCH  -  CIFAR-10")
    print("=" * 72)
    Xtr_all, ytr_all, Xte_all, yte_all = load_cifar10()
    rng = np.random.default_rng(args.seed)
    tr = rng.permutation(len(Xtr_all))[:args.n_train]
    te = rng.permutation(len(Xte_all))[:args.n_test]
    X = Xtr_all[tr].astype(np.float64) / 255.0
    y = ytr_all[tr]
    Xte = Xte_all[te].astype(np.float64) / 255.0
    yte = yte_all[te]
    # per-channel standardisation using training statistics only
    mu = X.mean(axis=(0, 2, 3), keepdims=True)
    sd = X.std(axis=(0, 2, 3), keepdims=True)
    X, Xte = (X - mu) / sd, (Xte - mu) / sd
    n_val = max(1000, args.n_train // 10)
    Xva, yva, X, y = X[:n_val], y[:n_val], X[n_val:], y[n_val:]
    Y, Yva, Yte = one_hot(y), one_hot(yva), one_hot(yte)

    net = CNN(seed=args.seed)
    n_par = sum(l.W.size + l.b.size for l in net.layers)
    print(f"Train / validation / test : {len(X)} / {len(Xva)} / {len(Xte)}"
          f"   (subset of 50000 / 10000)")
    print(f"Input 3x32x32, per-channel standardised; mean RGB = "
          f"{np.round(mu.ravel()*255, 1).tolist()}")
    print("Architecture : conv(16,3x3)-ReLU-pool -> conv(32,3x3)-ReLU-pool "
          "-> dense(128)-ReLU -> dense(10)-softmax")
    print(f"Parameters   : {n_par:,}   "
          f"(conv1 {net.c1.W.size + net.c1.b.size}, conv2 "
          f"{net.c2.W.size + net.c2.b.size}, dense1 {net.d1.W.size + net.d1.b.size}, "
          f"dense2 {net.d2.W.size + net.d2.b.size})")
    print(f"Optimiser    : SGD, batch = {args.batch}, eta0 = {args.eta}, "
          f"momentum = {args.momentum}, decay = {args.decay}/epoch")
    print()

    print("-" * 72)
    print("(a) GRADIENT CHECK of every layer (mini-batch of 8 images)")
    print("-" * 72)
    errs = gradient_check(CNN(seed=args.seed), X[:8], Y[:8], seed=args.seed)
    for k, v in errs.items():
        print(f"   {k:>8}: max relative error = {v:.3e}")
    print("   (this verifies im2col/col2im, the max-pool routing and the")
    print("    softmax+cross-entropy delta all at once)")
    print()

    print("-" * 72)
    print("(b) TRAINING")
    print("-" * 72)
    print(f"{'epoch':>6}{'train loss':>13}{'train acc':>11}{'val loss':>11}"
          f"{'val acc':>10}{'eta':>9}{'sec':>8}")
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    eta = args.eta
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        order = rng.permutation(len(X))
        rl, rc, seen = 0.0, 0, 0
        for s in range(0, len(X), args.batch):
            idx = order[s:s + args.batch]
            P = net.forward(X[idx])
            net.backward(P, Y[idx], args.lam)
            net.step(eta, args.momentum)
            rl += loss_fn(P, Y[idx]) * len(idx)
            rc += int(np.sum(np.argmax(P, 1) == y[idx]))
            seen += len(idx)
        Pva = net.predict(Xva)
        vl, va = loss_fn(Pva, Yva), float(np.mean(np.argmax(Pva, 1) == yva))
        hist["train_loss"].append(rl / seen)
        hist["train_acc"].append(rc / seen)
        hist["val_loss"].append(vl)
        hist["val_acc"].append(va)
        print(f"{ep:>6}{rl/seen:13.5f}{rc/seen:11.4f}{vl:11.5f}{va:10.4f}"
              f"{eta:9.4f}{time.time()-t0:8.1f}")
        eta *= args.decay
    print()

    Pte = net.predict(Xte)
    yhat = np.argmax(Pte, 1)
    acc = float(np.mean(yhat == yte))
    M = confusion(yte, yhat)
    print("-" * 72)
    print("(c) TEST PERFORMANCE")
    print("-" * 72)
    print(f"Test cross-entropy = {loss_fn(Pte, Yte):.5f}")
    print(f"Test accuracy      = {acc:.4f}   (chance = 0.1000)")
    print("\nPer-class precision / recall / F1:")
    per = {}
    for i, c in enumerate(CLASSES):
        tp = M[i, i]
        pr = tp / M[:, i].sum() if M[:, i].sum() else 0.0
        rc_ = tp / M[i].sum() if M[i].sum() else 0.0
        f1 = 2 * pr * rc_ / (pr + rc_) if pr + rc_ else 0.0
        per[c] = {"precision": float(pr), "recall": float(rc_), "f1": float(f1)}
        print(f"   {c:>11}: precision = {pr:.4f}, recall = {rc_:.4f}, F1 = {f1:.4f}")
    off = M - np.diag(np.diag(M))
    i, j = np.unravel_index(np.argmax(off), off.shape)
    print(f"\nMost frequent confusion: true '{CLASSES[i]}' predicted as "
          f"'{CLASSES[j]}' ({off[i, j]} times)")
    print()

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
    ax[1].axhline(0.1, color="gray", ls=":", lw=1, label="chance")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    f1p = os.path.join(FIGDIR, "07_cifar10_curves.pdf")
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
    ax.set_title(f"CIFAR-10 test confusion matrix (acc = {acc:.4f})")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    f2p = os.path.join(FIGDIR, "07_cifar10_confusion.pdf")
    fig.savefig(f2p)
    plt.close(fig)

    fig, axes = plt.subplots(2, 8, figsize=(9, 2.8))
    W1 = net.c1.W.reshape(-1, 3, 3, 3)
    for k, a in enumerate(axes.ravel()):
        f = W1[k].transpose(1, 2, 0)
        a.imshow((f - f.min()) / (f.max() - f.min() + 1e-12))
        a.axis("off")
    fig.suptitle("Learned conv1 filters (16 x 3x3x3, contrast normalised)",
                 fontsize=10)
    fig.tight_layout()
    f3p = os.path.join(FIGDIR, "07_cifar10_filters.pdf")
    fig.savefig(f3p)
    plt.close(fig)

    with open(os.path.join(RESDIR, "07_cifar10.json"), "w") as fh:
        json.dump({"parameters": int(n_par), "n_train": int(len(X)),
                   "n_val": int(len(Xva)), "n_test": int(len(Xte)),
                   "epochs": args.epochs, "batch": args.batch, "eta0": args.eta,
                   "grad_check": errs, "test_acc": acc,
                   "test_ce": loss_fn(Pte, Yte),
                   "final_train_acc": hist["train_acc"][-1],
                   "final_val_acc": hist["val_acc"][-1],
                   "history": hist, "confusion": M.tolist(), "per_class": per,
                   "worst_confusion": [CLASSES[i], CLASSES[j], int(off[i, j])]},
                  fh, indent=2)
    for f in (f1p, f2p, f3p):
        print(f"Figure saved: {f}")
    print(f"Results saved: {os.path.join(RESDIR, '07_cifar10.json')}")


if __name__ == "__main__":
    main()
