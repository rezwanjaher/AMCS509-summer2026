"""
05 - Neural network: a 3-layer fully connected network (MLP) written from
     scratch in NumPy and trained on Fashion-MNIST (10 clothing classes,
     28x28 greyscale images).

The four data files are downloaded from the official GitHub repository the
first time the script runs and cached in ./data.

======================================================================
ARCHITECTURE
======================================================================
    input 784  ->  hidden 256 (ReLU)  ->  hidden 128 (ReLU)  ->  output 10 (softmax)

======================================================================
1. FORWARD PASS
======================================================================
Each layer does the same two things: a linear step, then an activation.

    Layer 1:   z1 = X  W1 + b1        a1 = ReLU(z1)
    Layer 2:   z2 = a1 W2 + b2        a2 = ReLU(z2)
    Layer 3:   z3 = a2 W3 + b3        P  = softmax(z3)

    ReLU(z)    = max(0, z)
    softmax(z)_c = e^{z_c} / sum_k e^{z_k}      (outputs are positive, sum to 1)

P[i, c] is the predicted probability that image i belongs to class c.

======================================================================
2. LOSS: CATEGORICAL CROSS-ENTROPY
======================================================================
With one-hot targets Y (Y[i, c] = 1 for the true class, 0 elsewhere) and a
mini-batch of B images:

    L = -(1/B) sum_i sum_c Y[i,c] * log(P[i,c])

If the model predicted every class equally (P = 1/10), the loss would be
log 10 = 2.3026. That is the "no learning yet" reference line.

======================================================================
3. BACKWARD PASS: THE CHAIN RULE, LAYER BY LAYER
======================================================================
We need dL/dW and dL/db for each layer. The chain of dependencies is

    W3 -> z3 -> P -> L
    W2 -> z2 -> a2 -> z3 -> P -> L
    W1 -> z1 -> a1 -> z2 -> a2 -> z3 -> P -> L

so we work from the loss backwards, one arrow at a time. Write
dz_l = dL/dz_l for short.

STEP 1 - output layer.  Combining the derivative of the log and the softmax
Jacobian, everything cancels and leaves the simplest possible result:

    dz3 = (P - Y) / B          <- prediction minus target

STEP 2 - for ANY layer, once you know dz_l, the linear step z = a_prev W + b
gives three derivatives directly:

    dL/dW_l = a_prev^T  dz_l           (because dz/dW = a_prev)
    dL/db_l = column sums of dz_l      (because dz/db = 1)
    dL/da_prev = dz_l  W_l^T           (because dz/da_prev = W_l)

STEP 3 - to go one layer further back we pass dL/da_prev through the ReLU of
that layer, multiplying element by element by its derivative:

    ReLU'(z) = 1 if z > 0, else 0

    dz_prev = dL/da_prev * ReLU'(z_prev)

Note the consequence: if a unit's z was negative, ReLU'(z) = 0, so that unit
gets NO gradient. If that happens for every input, the unit is stuck forever -
the "dead ReLU" problem.

Applying steps 2 and 3 repeatedly:

    dz3 = (P - Y)/B
    dW3 = a2^T dz3      db3 = sum(dz3)      da2 = dz3 W3^T
    dz2 = da2 * ReLU'(z2)
    dW2 = a1^T dz2      db2 = sum(dz2)      da1 = dz2 W2^T
    dz1 = da1 * ReLU'(z1)
    dW1 = X^T  dz1      db1 = sum(dz1)

======================================================================
4. PARAMETER UPDATE: MINI-BATCH GRADIENT DESCENT
======================================================================
    W <- W - eta * dL/dW
    b <- b - eta * dL/db

Instead of using all 54,000 training images for every update (slow), we use a
mini-batch of 128 at a time. The mini-batch gradient is a noisy but unbiased
estimate of the full gradient, and we get hundreds of updates per pass over the
data instead of one. One pass over the whole training set is called an epoch.

Initial weights are small random numbers scaled by sqrt(2 / n_in): if they are
too large the activations blow up layer by layer, and if they are all zero
every unit in a layer computes the same thing and stays identical forever.
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
FILES = {"train_X": "train-images-idx3-ubyte.gz",
         "train_y": "train-labels-idx1-ubyte.gz",
         "test_X": "t10k-images-idx3-ubyte.gz",
         "test_y": "t10k-labels-idx1-ubyte.gz"}
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# ======================================================================
# DATA
# ======================================================================
def load_fashion_mnist():
    """Download (once) and read the four IDX files into NumPy arrays."""
    os.makedirs(DATADIR, exist_ok=True)
    out = {}
    for key, fname in FILES.items():
        path = os.path.join(DATADIR, fname)
        if not os.path.exists(path):
            print(f"  downloading {fname} ...")
            urllib.request.urlretrieve(BASE + fname, path)
        with gzip.open(path, "rb") as fh:
            raw = fh.read()
        if "X" in key:                       # images: 16-byte header, then pixels
            out[key] = np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 784)
        else:                                # labels: 8-byte header, then labels
            out[key] = np.frombuffer(raw, np.uint8, offset=8)
    # pixels are 0..255; divide by 255 so every input is in [0, 1]
    return (out["train_X"].astype(np.float64) / 255.0, out["train_y"].astype(int),
            out["test_X"].astype(np.float64) / 255.0, out["test_y"].astype(int))


# ======================================================================
# ACTIVATIONS AND LOSS
# ======================================================================
def relu(z):
    """ReLU(z) = max(0, z)."""
    return np.maximum(0.0, z)


def relu_derivative(z):
    """ReLU'(z) = 1 if z > 0, else 0."""
    return (z > 0).astype(float)


def softmax(Z):
    """Row-wise softmax.

    We subtract the row maximum first. This changes nothing mathematically -
    e^{z-m}/sum e^{z-m} = e^z/sum e^z - but it keeps every exponent <= 0 so
    e^{...} can never overflow.
    """
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def cross_entropy(P, Y):
    """L = -(1/B) sum over samples and classes of Y * log(P)."""
    return float(-np.sum(Y * np.log(P + 1e-12)) / Y.shape[0])


def one_hot(y, C=10):
    """Label 3 becomes the row [0,0,0,1,0,0,0,0,0,0]."""
    Y = np.zeros((len(y), C))
    Y[np.arange(len(y)), y] = 1.0
    return Y


# ======================================================================
# THE NETWORK
# ======================================================================
class MLP:
    """784 -> 256 (ReLU) -> 128 (ReLU) -> 10 (softmax)."""

    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        # One weight matrix and one bias vector per layer.
        # W[l] has shape (units in, units out); b[l] has one entry per output.
        self.W = [rng.standard_normal((sizes[i], sizes[i + 1])) * np.sqrt(2.0 / sizes[i])
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    # ---------------- FORWARD PASS ------------------------------------
    def forward(self, X):
        """Compute the class probabilities and remember the intermediate values.

        We store z1, a1, z2, a2 because the backward pass needs them:
        dW needs the layer's input a_prev, and the ReLU step needs z.
        """
        self.X = X
        self.z1 = X @ self.W[0] + self.b[0]        # linear step, layer 1
        self.a1 = relu(self.z1)                    # activation, layer 1
        self.z2 = self.a1 @ self.W[1] + self.b[1]  # linear step, layer 2
        self.a2 = relu(self.z2)                    # activation, layer 2
        self.z3 = self.a2 @ self.W[2] + self.b[2]  # linear step, output layer
        self.P = softmax(self.z3)                  # probabilities, one row per image
        return self.P

    # ---------------- BACKWARD PASS -----------------------------------
    def backward(self, Y):
        """Chain rule, layer by layer, from the loss back to W1.

        Returns the gradients in the same order as self.W and self.b.
        """
        B = Y.shape[0]                             # mini-batch size

        # --- output layer -------------------------------------------------
        # softmax + cross-entropy together give simply (prediction - target)
        dz3 = (self.P - Y) / B                     # dL/dz3
        dW3 = self.a2.T @ dz3                      # dL/dW3 = a2^T dz3
        db3 = dz3.sum(axis=0)                      # dL/db3 = sum of dz3
        da2 = dz3 @ self.W[2].T                    # dL/da2 = dz3 W3^T

        # --- hidden layer 2 -----------------------------------------------
        dz2 = da2 * relu_derivative(self.z2)       # through the ReLU
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W[1].T

        # --- hidden layer 1 -----------------------------------------------
        dz1 = da1 * relu_derivative(self.z1)       # through the ReLU
        dW1 = self.X.T @ dz1
        db1 = dz1.sum(axis=0)

        return [dW1, dW2, dW3], [db1, db2, db3]

    # ---------------- UPDATE ------------------------------------------
    def update(self, gW, gb, eta):
        """Gradient descent step: W <- W - eta*dL/dW,  b <- b - eta*dL/db."""
        for l in range(len(self.W)):
            self.W[l] = self.W[l] - eta * gW[l]
            self.b[l] = self.b[l] - eta * gb[l]

    def predict(self, X, batch=2000):
        """Forward pass only, in chunks so we do not build huge arrays."""
        return np.vstack([self.forward(X[i:i + batch]) for i in range(0, len(X), batch)])


def check_gradient(net, X, Y, n_checks=8, eps=1e-6, seed=0):
    """Check the backward pass against the definition of a derivative.

    dL/dW[i,j] should equal the slope [L(W + eps) - L(W - eps)] / (2*eps).
    If the two agree, the chain-rule code above is correct.
    """
    rng = np.random.default_rng(seed)
    net.forward(X)
    gW, _ = net.backward(Y)
    worst = 0.0
    for _ in range(n_checks):
        l = int(rng.integers(len(net.W)))
        i = int(rng.integers(net.W[l].shape[0]))
        j = int(rng.integers(net.W[l].shape[1]))
        original = net.W[l][i, j]
        net.W[l][i, j] = original + eps
        loss_plus = cross_entropy(net.forward(X), Y)
        net.W[l][i, j] = original - eps
        loss_minus = cross_entropy(net.forward(X), Y)
        net.W[l][i, j] = original                  # put the weight back
        slope = (loss_plus - loss_minus) / (2 * eps)
        analytic = gW[l][i, j]
        worst = max(worst, abs(slope - analytic) / max(1e-12,
                                                       abs(slope) + abs(analytic)))
    return float(worst)


def confusion_matrix(y_true, y_pred, C=10):
    M = np.zeros((C, C), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[t, p] += 1
    return M


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eta", type=float, default=0.5, help="learning rate")
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(RESDIR, exist_ok=True)

    print("=" * 72)
    print("05 | NEURAL NETWORK FROM SCRATCH  -  Fashion-MNIST")
    print("=" * 72)
    Xtr, ytr, Xte, yte = load_fashion_mnist()

    # hold out 6000 training images to watch for overfitting
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(Xtr))
    val, tr = perm[:6000], perm[6000:]
    Xva, yva, Xtr, ytr = Xtr[val], ytr[val], Xtr[tr], ytr[tr]
    Ytr, Yva, Yte = one_hot(ytr), one_hot(yva), one_hot(yte)

    sizes = [784] + list(args.hidden) + [10]
    n_par = sum(a * b for a, b in zip(sizes[:-1], sizes[1:])) + sum(sizes[1:])
    print(f"Train / validation / test : {len(Xtr)} / {len(Xva)} / {len(Xte)}")
    print("Each image is 28x28 = 784 pixels, scaled to [0, 1].")
    print(f"Architecture : {' -> '.join(map(str, sizes))}  (ReLU hidden, softmax output)")
    print(f"Parameters   : {n_par:,}")
    print(f"Training     : mini-batch gradient descent, batch = {args.batch}, "
          f"eta = {args.eta}")
    print(f"Loss at the start should be about log 10 = {np.log(10):.4f}")
    print()

    net = MLP(sizes, seed=args.seed)

    # ---------------- gradient check --------------------------------------
    print("-" * 72)
    print("(a) CHECKING THE BACKWARD PASS on a mini-batch of 32 images")
    print("-" * 72)
    err = check_gradient(MLP(sizes, seed=args.seed), Xtr[:32], Ytr[:32], seed=args.seed)
    print(f"largest relative difference between the chain-rule gradient")
    print(f"and the numerical slope = {err:.3e}")
    print("(a small number confirms the backward pass matches the derivatives)")
    print()

    # ---------------- training loop ---------------------------------------
    print("-" * 72)
    print("(b) TRAINING")
    print("-" * 72)
    print(f"{'epoch':>6}{'train loss':>13}{'train acc':>11}"
          f"{'val loss':>11}{'val acc':>10}{'sec':>8}")
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    n = len(Xtr)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        order = rng.permutation(n)                 # reshuffle every epoch
        total_loss, correct, seen = 0.0, 0, 0
        for start in range(0, n, args.batch):
            idx = order[start:start + args.batch]
            Xb, Yb = Xtr[idx], Ytr[idx]
            P = net.forward(Xb)                    # 1. forward
            gW, gb = net.backward(Yb)              # 2. backward (chain rule)
            net.update(gW, gb, args.eta)           # 3. gradient descent step
            total_loss += cross_entropy(P, Yb) * len(idx)
            correct += int(np.sum(np.argmax(P, axis=1) == ytr[idx]))
            seen += len(idx)
        Pva = net.predict(Xva)
        vloss = cross_entropy(Pva, Yva)
        vacc = float(np.mean(np.argmax(Pva, axis=1) == yva))
        hist["train_loss"].append(total_loss / seen)
        hist["train_acc"].append(correct / seen)
        hist["val_loss"].append(vloss)
        hist["val_acc"].append(vacc)
        print(f"{epoch:>6}{total_loss/seen:13.5f}{correct/seen:11.4f}"
              f"{vloss:11.5f}{vacc:10.4f}{time.time()-t0:8.1f}")
    print()

    # ---------------- test evaluation -------------------------------------
    Pte = net.predict(Xte)
    y_pred = np.argmax(Pte, axis=1)                # predicted class = highest probability
    accuracy = float(np.mean(y_pred == yte))
    M = confusion_matrix(yte, y_pred)

    print("-" * 72)
    print("(c) TEST PERFORMANCE")
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
        print(f"   {name:>12}: precision = {precision:.4f}, "
              f"recall = {recall:.4f}, F1 = {f1:.4f}")
    off_diagonal = M - np.diag(np.diag(M))
    i, j = np.unravel_index(np.argmax(off_diagonal), off_diagonal.shape)
    print(f"\nMost frequent mistake: true '{CLASSES[i]}' predicted as "
          f"'{CLASSES[j]}' ({off_diagonal[i, j]} times)")
    print()

    # ---------------- plots -----------------------------------------------
    ep = np.arange(1, args.epochs + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, hist["train_loss"], "o-", ms=3, label="train")
    ax[0].plot(ep, hist["val_loss"], "s-", ms=3, label="validation")
    ax[0].axhline(np.log(10), color="gray", ls=":", lw=1, label=r"$\log 10$ (guessing)")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("cross-entropy")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(ep, hist["train_acc"], "o-", ms=3, label="train")
    ax[1].plot(ep, hist["val_acc"], "s-", ms=3, label="validation")
    ax[1].axhline(accuracy, color="crimson", ls="--", lw=1, label=f"test = {accuracy:.4f}")
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
        for b_ in range(10):
            ax.text(b_, a, M[a, b_], ha="center", va="center", fontsize=5.5,
                    color="white" if M[a, b_] > M.max() / 2 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Fashion-MNIST test confusion matrix (accuracy = {accuracy:.4f})")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    f2p = os.path.join(FIGDIR, "05_fashion_mnist_confusion.pdf")
    fig.savefig(f2p)
    plt.close(fig)

    fig, axes = plt.subplots(3, 6, figsize=(9, 5))
    for k, a in enumerate(axes.ravel()):
        a.imshow(Xte[k].reshape(28, 28), cmap="gray")
        correct_k = y_pred[k] == yte[k]
        a.set_title(f"{CLASSES[y_pred[k]]}\n({CLASSES[yte[k]]})", fontsize=6,
                    color="green" if correct_k else "red")
        a.axis("off")
    fig.suptitle("Predictions (true class in brackets)", fontsize=10)
    fig.tight_layout()
    f3p = os.path.join(FIGDIR, "05_fashion_mnist_samples.pdf")
    fig.savefig(f3p)
    plt.close(fig)

    fig, axes = plt.subplots(4, 8, figsize=(9, 5))
    for k, a in enumerate(axes.ravel()):
        a.imshow(net.W[0][:, k].reshape(28, 28), cmap="RdBu_r")
        a.axis("off")
    fig.suptitle("First-layer weights, one column of $W_1$ per image", fontsize=10)
    fig.tight_layout()
    f4p = os.path.join(FIGDIR, "05_fashion_mnist_filters.pdf")
    fig.savefig(f4p)
    plt.close(fig)

    with open(os.path.join(RESDIR, "05_fashion_mnist.json"), "w") as fh:
        json.dump({"architecture": sizes, "parameters": int(n_par),
                   "epochs": args.epochs, "batch": args.batch, "eta": args.eta,
                   "grad_check_rel_err": err,
                   "test_acc": accuracy, "test_ce": cross_entropy(Pte, Yte),
                   "final_train_acc": hist["train_acc"][-1],
                   "final_train_loss": hist["train_loss"][-1],
                   "final_val_acc": hist["val_acc"][-1],
                   "final_val_loss": hist["val_loss"][-1],
                   "history": hist, "confusion": M.tolist(),
                   "per_class": per_class,
                   "worst_confusion": [CLASSES[i], CLASSES[j],
                                       int(off_diagonal[i, j])]}, fh, indent=2)
    for f in (f1p, f2p, f3p, f4p):
        print(f"Figure saved: {f}")
    print(f"Results saved: {os.path.join(RESDIR, '05_fashion_mnist.json')}")


if __name__ == "__main__":
    main()
