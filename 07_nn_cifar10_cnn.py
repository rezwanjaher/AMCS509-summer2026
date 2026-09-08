"""
07 - Neural network: a small Convolutional Neural Network (CNN) written from
     scratch in NumPy and trained on CIFAR-10 (10 classes of 32x32 colour
     images).

======================================================================
ARCHITECTURE
======================================================================
    input 3 x 32 x 32
      convolution: 16 filters of size 3x3   -> ReLU -> 2x2 max pooling  (16x16x16)
      convolution: 32 filters of size 3x3   -> ReLU -> 2x2 max pooling  (32x8x8)
      flatten (2048) -> dense(128) -> ReLU -> dense(10) -> softmax

======================================================================
1. WHAT A CONVOLUTION IS
======================================================================
A filter is a small block of weights that slides over the image. At every
position we multiply the filter by the patch underneath it and add up:

    z[f, i, j] = sum_c sum_u sum_v  W[f, c, u, v] * a[c, i+u, j+v]  +  b[f]

Compare this with the dense layer z = Wx + b of script 05. A convolution is a
dense layer with two restrictions:
  * each output looks at a small 3x3 patch instead of the whole image;
  * the SAME filter is used at every position (the weights are shared).
Because of that, the filter has only 3*3*3 = 27 weights no matter how big the
image is, and a pattern is recognised wherever it appears.

Implementation note: instead of looping over the 32x32 output positions in
Python (slow), we loop over the 9 positions INSIDE the filter and shift the
image with a slice. The two are the same sum, just added in a different order.

======================================================================
2. BACKWARD PASS FOR A CONVOLUTION
======================================================================
With delta = dL/dz for this layer, and the same 9 filter offsets:

    dW[f, c, u, v] += sum over batch and positions of delta[f,i,j]*a[c,i+u,j+v]
    db[f]          += sum of delta[f, :, :]
    da[c, i+u, j+v] += sum_f W[f, c, u, v] * delta[f, i, j]

The "+=" appears because a shared weight is used many times, so the chain rule
adds up one contribution per use - and because neighbouring patches overlap, a
pixel receives gradient from several output positions.

======================================================================
3. MAX POOLING
======================================================================
Take the largest value in each 2x2 block. The derivative of a maximum is 1 for
the entry that WAS the maximum and 0 for the others, so the backward pass sends
the gradient only to the winning position. Pooling has no weights; it halves
the image size and makes the next layer see a wider area.

======================================================================
4. LOSS AND UPDATE
======================================================================
Softmax output with cross-entropy loss, exactly as in script 05, so at the
output layer delta = (P - Y) / B. Training is plain mini-batch gradient
descent:  W <- W - eta * dW.
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


# ======================================================================
# DATA
# ======================================================================
def _from_official():
    tgz = os.path.join(DATADIR, "cifar-10-python.tar.gz")
    if not os.path.exists(tgz):
        print(f"  downloading {OFFICIAL} ...")
        urllib.request.urlretrieve(OFFICIAL, tgz)
    with tarfile.open(tgz) as tf:
        def read(name):
            return pickle.load(tf.extractfile(name), encoding="bytes")
        X_train, y_train = [], []
        for k in range(1, 6):
            batch = read(f"cifar-10-batches-py/data_batch_{k}")
            X_train.append(batch[b"data"])
            y_train += batch[b"labels"]
        test = read("cifar-10-batches-py/test_batch")
    return (np.concatenate(X_train).reshape(-1, 3, 32, 32), np.array(y_train),
            test[b"data"].reshape(-1, 3, 32, 32), np.array(test[b"labels"]))


def _from_mirror():
    import glob
    import matplotlib.image as mpimg
    repo = os.path.join(DATADIR, "cifar10_png")
    if not os.path.isdir(repo):
        print(f"  official host unreachable; cloning {MIRROR} ...")
        subprocess.run(["git", "clone", "--depth", "1", "-q", MIRROR, repo],
                       check=True)
    out = {}
    for split in ("train", "test"):
        X, y = [], []
        for label, name in enumerate(CLASSES):
            for f in sorted(glob.glob(os.path.join(repo, split, name, "*"))):
                X.append(mpimg.imread(f))
                y.append(label)
        out[split] = (np.array(X).transpose(0, 3, 1, 2), np.array(y))
    return out["train"][0], out["train"][1], out["test"][0], out["test"][1]


def load_cifar10():
    os.makedirs(DATADIR, exist_ok=True)
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z["Xtr"], z["ytr"], z["Xte"], z["yte"]
    try:
        Xtr, ytr, Xte, yte = _from_official()
    except Exception as exc:
        print(f"  official download failed ({type(exc).__name__}); using mirror")
        Xtr, ytr, Xte, yte = _from_mirror()
    np.savez_compressed(CACHE, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    return Xtr, ytr, Xte, yte


# ======================================================================
# ACTIVATIONS AND LOSS
# ======================================================================
def relu(z):
    return np.maximum(0.0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def cross_entropy(P, Y):
    return float(-np.sum(Y * np.log(P + 1e-12)) / Y.shape[0])


def one_hot(y, C=10):
    Y = np.zeros((len(y), C))
    Y[np.arange(len(y)), y] = 1.0
    return Y


# ======================================================================
# CONVOLUTION AND POOLING
# ======================================================================
class Conv:
    """3x3 convolution with padding 1, so the output has the same H and W."""

    def __init__(self, channels_in, n_filters, rng, k=3, pad=1):
        self.k, self.pad = k, pad
        fan_in = channels_in * k * k
        self.W = rng.standard_normal((n_filters, channels_in, k, k)) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros(n_filters)

    def forward(self, A):
        """Slide every filter over the image and accumulate."""
        self.A = A
        B, C, H, Wd = A.shape
        F, k, pad = self.W.shape[0], self.k, self.pad
        A_pad = np.pad(A, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        Z = np.zeros((B, F, H, Wd))
        for u in range(k):                    # loop over the 3 rows of the filter
            for v in range(k):                # loop over the 3 columns
                # the patch of the image that this filter entry multiplies
                patch = A_pad[:, :, u:u + H, v:v + Wd]
                # for every filter f: sum over the input channels c
                Z += np.einsum("fc,bcij->bfij", self.W[:, :, u, v], patch)
        return Z + self.b.reshape(1, -1, 1, 1)

    def backward(self, dZ):
        """Gradients of the filters, the bias, and the input."""
        B, C, H, Wd = self.A.shape
        k, pad = self.k, self.pad
        A_pad = np.pad(self.A, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        dA_pad = np.zeros_like(A_pad)
        self.dW = np.zeros_like(self.W)
        self.db = dZ.sum(axis=(0, 2, 3))
        for u in range(k):
            for v in range(k):
                patch = A_pad[:, :, u:u + H, v:v + Wd]
                # one filter weight is used at every position, so we sum over
                # the batch and over all positions
                self.dW[:, :, u, v] = np.einsum("bfij,bcij->fc", dZ, patch)
                # send the gradient back to the pixels this weight multiplied
                dA_pad[:, :, u:u + H, v:v + Wd] += np.einsum(
                    "fc,bfij->bcij", self.W[:, :, u, v], dZ)
        return dA_pad[:, :, pad:pad + H, pad:pad + Wd]

    def update(self, eta):
        self.W -= eta * self.dW
        self.b -= eta * self.db


class Dense:
    def __init__(self, n_in, n_out, rng):
        self.W = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
        self.b = np.zeros(n_out)

    def forward(self, A):
        self.A = A
        return A @ self.W + self.b

    def backward(self, dZ):
        self.dW = self.A.T @ dZ
        self.db = dZ.sum(axis=0)
        return dZ @ self.W.T

    def update(self, eta):
        self.W -= eta * self.dW
        self.b -= eta * self.db


def maxpool_forward(A):
    """2x2 max pooling: take the biggest of the four values in each block."""
    top_left = A[:, :, 0::2, 0::2]
    top_right = A[:, :, 0::2, 1::2]
    bottom_left = A[:, :, 1::2, 0::2]
    bottom_right = A[:, :, 1::2, 1::2]
    out = np.maximum(np.maximum(top_left, top_right),
                     np.maximum(bottom_left, bottom_right))
    return out


def maxpool_backward(dOut, A):
    """Send the gradient only to the position that held the maximum."""
    out = maxpool_forward(A)
    dA = np.zeros_like(A)
    # winners[...] is True where the value equalled the block maximum
    dA[:, :, 0::2, 0::2] = (A[:, :, 0::2, 0::2] == out) * dOut
    dA[:, :, 0::2, 1::2] = (A[:, :, 0::2, 1::2] == out) * dOut
    dA[:, :, 1::2, 0::2] = (A[:, :, 1::2, 0::2] == out) * dOut
    dA[:, :, 1::2, 1::2] = (A[:, :, 1::2, 1::2] == out) * dOut
    return dA


# ======================================================================
# THE NETWORK
# ======================================================================
class CNN:
    def __init__(self, seed=0, f1=16, f2=32, hidden=128):
        rng = np.random.default_rng(seed)
        self.conv1 = Conv(3, f1, rng)
        self.conv2 = Conv(f1, f2, rng)
        self.dense1 = Dense(f2 * 8 * 8, hidden, rng)
        self.dense2 = Dense(hidden, 10, rng)
        self.layers = [self.conv1, self.conv2, self.dense1, self.dense2]

    def forward(self, X):
        self.z1 = self.conv1.forward(X)         # convolution 1
        self.a1 = relu(self.z1)                 # ReLU
        self.p1 = maxpool_forward(self.a1)      # 2x2 max pooling
        self.z2 = self.conv2.forward(self.p1)   # convolution 2
        self.a2 = relu(self.z2)
        self.p2 = maxpool_forward(self.a2)
        self.flat = self.p2.reshape(len(X), -1)  # flatten for the dense layers
        self.z3 = self.dense1.forward(self.flat)
        self.a3 = relu(self.z3)
        return softmax(self.dense2.forward(self.a3))

    def backward(self, P, Y):
        B = len(Y)
        delta = (P - Y) / B                          # softmax + cross-entropy
        delta = self.dense2.backward(delta)
        delta = delta * relu_derivative(self.z3)     # through the ReLU
        delta = self.dense1.backward(delta)
        delta = delta.reshape(self.p2.shape)         # undo the flatten
        delta = maxpool_backward(delta, self.a2)     # through the pooling
        delta = delta * relu_derivative(self.z2)
        delta = self.conv2.backward(delta)
        delta = maxpool_backward(delta, self.a1)
        delta = delta * relu_derivative(self.z1)
        self.conv1.backward(delta)

    def update(self, eta):
        for layer in self.layers:
            layer.update(eta)

    def predict(self, X, batch=500):
        return np.vstack([self.forward(X[i:i + batch])
                          for i in range(0, len(X), batch)])


def confusion_matrix(y_true, y_pred, C=10):
    M = np.zeros((C, C), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[t, p] += 1
    return M


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--n-train", type=int, default=10000,
                    help="training subset (50000 = full set, much slower)")
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
    mean = X.mean(axis=(0, 2, 3), keepdims=True)
    std = X.std(axis=(0, 2, 3), keepdims=True)
    X, Xte = (X - mean) / std, (Xte - mean) / std
    n_val = max(1000, args.n_train // 10)
    Xva, yva, X, y = X[:n_val], y[:n_val], X[n_val:], y[n_val:]
    Y, Yva, Yte = one_hot(y), one_hot(yva), one_hot(yte)

    net = CNN(seed=args.seed)
    n_par = sum(l.W.size + l.b.size for l in net.layers)
    print(f"Train / validation / test : {len(X)} / {len(Xva)} / {len(Xte)}"
          "   (subset of 50000 / 10000)")
    print("Architecture: conv(16) - ReLU - pool - conv(32) - ReLU - pool "
          "- dense(128) - ReLU - dense(10) - softmax")
    print(f"Parameters  : {n_par:,} "
          f"(only {net.conv1.W.size + net.conv2.W.size:,} of them are filter weights)")
    print(f"Training    : mini-batch gradient descent, batch = {args.batch}, "
          f"eta = {args.eta}")
    print()

    print("-" * 72)
    print("TRAINING")
    print("-" * 72)
    print(f"{'epoch':>6}{'train loss':>13}{'train acc':>11}"
          f"{'val loss':>11}{'val acc':>10}{'sec':>8}")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        order = rng.permutation(len(X))
        total_loss, correct, seen = 0.0, 0, 0
        for start in range(0, len(X), args.batch):
            idx = order[start:start + args.batch]
            P = net.forward(X[idx])               # forward
            net.backward(P, Y[idx])               # backward (chain rule)
            net.update(args.eta)                  # gradient descent step
            total_loss += cross_entropy(P, Y[idx]) * len(idx)
            correct += int(np.sum(np.argmax(P, axis=1) == y[idx]))
            seen += len(idx)
        Pva = net.predict(Xva)
        vloss = cross_entropy(Pva, Yva)
        vacc = float(np.mean(np.argmax(Pva, axis=1) == yva))
        history["train_loss"].append(total_loss / seen)
        history["train_acc"].append(correct / seen)
        history["val_loss"].append(vloss)
        history["val_acc"].append(vacc)
        print(f"{epoch:>6}{total_loss/seen:13.5f}{correct/seen:11.4f}"
              f"{vloss:11.5f}{vacc:10.4f}{time.time()-t0:8.1f}")
    print()

    Pte = net.predict(Xte)
    y_pred = np.argmax(Pte, axis=1)
    accuracy = float(np.mean(y_pred == yte))
    M = confusion_matrix(yte, y_pred)
    print("-" * 72)
    print("TEST PERFORMANCE")
    print("-" * 72)
    print(f"Test cross-entropy = {cross_entropy(Pte, Yte):.5f}")
    print(f"Test accuracy      = {accuracy:.4f}   (random guessing = 0.1000)")
    print("\nPer-class precision / recall / F1:")
    per_class = {}
    for i, name in enumerate(CLASSES):
        tp = M[i, i]
        precision = tp / M[:, i].sum() if M[:, i].sum() else 0.0
        recall = tp / M[i].sum() if M[i].sum() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[name] = {"precision": float(precision), "recall": float(recall),
                           "f1": float(f1)}
        print(f"   {name:>11}: precision = {precision:.4f}, "
              f"recall = {recall:.4f}, F1 = {f1:.4f}")
    off = M - np.diag(np.diag(M))
    i, j = np.unravel_index(np.argmax(off), off.shape)
    print(f"\nMost frequent mistake: true '{CLASSES[i]}' predicted as "
          f"'{CLASSES[j]}' ({off[i, j]} times)")
    print()

    ep = np.arange(1, args.epochs + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, history["train_loss"], "o-", ms=3, label="train")
    ax[0].plot(ep, history["val_loss"], "s-", ms=3, label="validation")
    ax[0].axhline(np.log(10), color="gray", ls=":", lw=1, label=r"$\log 10$")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("cross-entropy")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(ep, history["train_acc"], "o-", ms=3, label="train")
    ax[1].plot(ep, history["val_acc"], "s-", ms=3, label="validation")
    ax[1].axhline(accuracy, color="crimson", ls="--", lw=1,
                  label=f"test = {accuracy:.4f}")
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
    ax.set_title(f"CIFAR-10 test confusion matrix (accuracy = {accuracy:.4f})")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    f2p = os.path.join(FIGDIR, "07_cifar10_confusion.pdf")
    fig.savefig(f2p)
    plt.close(fig)

    fig, axes = plt.subplots(2, 8, figsize=(9, 2.8))
    for k, a in enumerate(axes.ravel()):
        f = net.conv1.W[k].transpose(1, 2, 0)
        a.imshow((f - f.min()) / (f.max() - f.min() + 1e-12))
        a.axis("off")
    fig.suptitle("The 16 learned 3x3 filters of the first layer", fontsize=10)
    fig.tight_layout()
    f3p = os.path.join(FIGDIR, "07_cifar10_filters.pdf")
    fig.savefig(f3p)
    plt.close(fig)

    with open(os.path.join(RESDIR, "07_cifar10.json"), "w") as fh:
        json.dump({"parameters": int(n_par), "n_train": int(len(X)),
                   "n_val": int(len(Xva)), "n_test": int(len(Xte)),
                   "epochs": args.epochs, "batch": args.batch, "eta": args.eta,
                   "test_acc": accuracy, "test_ce": cross_entropy(Pte, Yte),
                   "final_train_acc": history["train_acc"][-1],
                   "final_val_acc": history["val_acc"][-1],
                   "history": history, "confusion": M.tolist(),
                   "per_class": per_class,
                   "worst_confusion": [CLASSES[i], CLASSES[j], int(off[i, j])]},
                  fh, indent=2)
    for f in (f1p, f2p, f3p):
        print(f"Figure saved: {f}")
    print(f"Results saved: {os.path.join(RESDIR, '07_cifar10.json')}")


if __name__ == "__main__":
    main()
