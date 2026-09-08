"""
08 - Neural network: sentiment classification of IMDB movie reviews.

Each review is a piece of text; the label is 1 for a positive review and 0 for
a negative one.

======================================================================
THE MODEL
======================================================================
    review -> list of word indices
           -> look up one vector per word  (embedding matrix E)
           -> average those vectors        (mean pooling)
           -> dense(64) -> ReLU
           -> dense(1)  -> sigmoid         -> probability the review is positive

======================================================================
1. TURNING WORDS INTO NUMBERS
======================================================================
We keep the V most frequent words and give each one an index. Word number k is
represented by row k of the embedding matrix E (V x d). The rows start as small
random numbers and are learned like any other weights.

2. MEAN POOLING (handles reviews of different lengths)

    u = (1/T) * sum over the T words of the review of E[word_t]      (length d)

3. THE REST IS A SMALL NEURAL NETWORK (same as script 05, but binary)

    z1 = u W1 + b1        a1 = ReLU(z1)
    z2 = a1 W2 + b2       p  = sigmoid(z2)

4. LOSS: binary cross-entropy, as in script 04

    L = -(1/B) sum [ y log p + (1-y) log(1-p) ]

5. BACKWARD PASS (chain rule, layer by layer)

    dz2 = (p - y)/B                        (sigmoid + BCE, the usual result)
    dW2 = a1^T dz2      db2 = sum(dz2)     da1 = dz2 W2^T
    dz1 = da1 * ReLU'(z1)
    dW1 = u^T dz1       db1 = sum(dz1)     du  = dz1 W1^T

    and back through the averaging: every word of the review gets an equal
    share of du,

        dE[word_t] += (1/T) * du

    The "+=" matters: a word that appears three times in a review collects
    three contributions, and only the words that actually appear get any
    gradient at all.

6. UPDATE: standard gradient descent,  W <- W - eta * dW.

WHAT THIS MODEL CAN AND CANNOT DO
Averaging throws away word order, so "not good" and "good not" look identical
to it. It still works well here because sentiment is mostly carried by the
individual words. Word order needs a recurrent model like script 06.
"""

import argparse
import json
import os
import re
import tarfile
import urllib.request
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
RESDIR = os.path.join(HERE, "results")
DATADIR = os.path.join(HERE, "data")
CACHE = os.path.join(DATADIR, "imdb_raw.npz")
OFFICIAL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
MIRROR = ("https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/"
          "master/IMDB-Dataset.csv")
WORD = re.compile(r"[a-z']+")


# ======================================================================
# DATA
# ======================================================================
def _from_official():
    tgz = os.path.join(DATADIR, "aclImdb_v1.tar.gz")
    if not os.path.exists(tgz):
        print(f"  downloading {OFFICIAL} ...")
        urllib.request.urlretrieve(OFFICIAL, tgz)
    texts, labels = [], []
    with tarfile.open(tgz) as tf:
        for member in tf.getmembers():
            parts = member.name.split("/")
            if (len(parts) == 4 and parts[1] in ("train", "test")
                    and parts[2] in ("pos", "neg") and member.isfile()):
                texts.append(tf.extractfile(member).read().decode("utf-8", "ignore"))
                labels.append(1 if parts[2] == "pos" else 0)
    return texts, np.array(labels)


def _from_mirror():
    import pandas as pd
    path = os.path.join(DATADIR, "imdb.csv")
    if not os.path.exists(path):
        print(f"  official host unreachable; downloading {MIRROR} ...")
        urllib.request.urlretrieve(MIRROR, path)
    df = pd.read_csv(path)
    text_col = "review" if "review" in df.columns else df.columns[0]
    label_col = "sentiment" if "sentiment" in df.columns else df.columns[1]
    labels = df[label_col].map(
        lambda v: 1 if str(v).strip().lower() in ("positive", "pos", "1") else 0)
    return df[text_col].astype(str).tolist(), labels.to_numpy()


def load_imdb():
    os.makedirs(DATADIR, exist_ok=True)
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return list(z["texts"]), z["labels"]
    try:
        texts, labels = _from_official()
    except Exception as exc:
        print(f"  official download failed ({type(exc).__name__}); using mirror")
        texts, labels = _from_mirror()
    np.savez_compressed(CACHE, texts=np.array(texts, dtype=object), labels=labels)
    return texts, labels


# ======================================================================
# THE NETWORK
# ======================================================================
def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos, neg = z >= 0, z < 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[neg])
    out[neg] = e / (1.0 + e)
    return out


class SentimentNet:
    def __init__(self, V, d, hidden, seed=0):
        rng = np.random.default_rng(seed)
        self.E = rng.standard_normal((V, d)) * 0.05       # one vector per word
        self.W1 = rng.standard_normal((d, hidden)) * np.sqrt(2.0 / d)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, 1)) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(1)

    # ---------------- forward -----------------------------------------
    def forward(self, documents):
        """documents is a list of arrays of word indices, one per review."""
        self.documents = documents
        # mean pooling: average the word vectors of each review
        self.U = np.zeros((len(documents), self.E.shape[1]))
        for i, word_indices in enumerate(documents):
            self.U[i] = self.E[word_indices].mean(axis=0)
        self.z1 = self.U @ self.W1 + self.b1
        self.a1 = np.maximum(0.0, self.z1)               # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        return sigmoid(self.z2).ravel()

    # ---------------- loss --------------------------------------------
    @staticmethod
    def loss(p, y):
        eps = 1e-12
        return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    # ---------------- backward ----------------------------------------
    def backward(self, p, y):
        B = len(y)
        dz2 = ((p - y) / B)[:, None]                     # sigmoid + BCE
        self.dW2 = self.a1.T @ dz2
        self.db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)                        # through the ReLU
        self.dW1 = self.U.T @ dz1
        self.db1 = dz1.sum(axis=0)
        dU = dz1 @ self.W1.T                             # gradient of the average
        # back through the averaging: split dU equally among the words used
        self.dE = np.zeros_like(self.E)
        for i, word_indices in enumerate(self.documents):
            share = dU[i] / len(word_indices)
            np.add.at(self.dE, word_indices, share)      # add, do not overwrite

    # ---------------- update ------------------------------------------
    def update(self, eta):
        self.E -= eta * self.dE
        self.W1 -= eta * self.dW1
        self.b1 -= eta * self.db1
        self.W2 -= eta * self.dW2
        self.b2 -= eta * self.db2


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(RESDIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 72)
    print("08 | SENTIMENT NEURAL NETWORK FROM SCRATCH  -  IMDB reviews")
    print("=" * 72)
    texts, labels = load_imdb()
    tokens = [WORD.findall(t.lower()) for t in texts]
    lengths = np.array([len(t) for t in tokens])
    print(f"Reviews       : {len(texts):,} "
          f"({int(labels.sum()):,} positive, {int((1 - labels).sum()):,} negative)")
    print(f"Review length : {lengths.mean():.1f} words on average")

    order = rng.permutation(len(texts))
    n_train = int(0.5 * len(texts))
    n_val = int(0.1 * len(texts))
    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]

    # vocabulary from the TRAINING reviews only
    counts = Counter()
    for i in train_idx:
        counts.update(tokens[i])
    vocabulary = {word: k + 1 for k, (word, _) in
                  enumerate(counts.most_common(args.vocab - 1))}
    print(f"Vocabulary    : {args.vocab:,} most frequent words "
          f"(out of {len(counts):,} distinct)")

    def encode(i):
        idx = [vocabulary[w] for w in tokens[i][:args.max_len] if w in vocabulary]
        return np.array(idx if idx else [0], dtype=int)

    encoded = [encode(i) for i in range(len(texts))]
    D_train = [encoded[i] for i in train_idx]
    D_val = [encoded[i] for i in val_idx]
    D_test = [encoded[i] for i in test_idx]
    y_train = labels[train_idx].astype(float)
    y_val = labels[val_idx].astype(float)
    y_test = labels[test_idx].astype(float)
    print(f"Train / validation / test : {len(D_train):,} / {len(D_val):,} / "
          f"{len(D_test):,}")

    net = SentimentNet(args.vocab, args.dim, args.hidden, seed=args.seed)
    n_par = net.E.size + net.W1.size + net.b1.size + net.W2.size + net.b2.size
    print(f"Model  : embedding {args.vocab}x{args.dim} -> average -> "
          f"dense({args.hidden}) -> ReLU -> dense(1) -> sigmoid")
    print(f"Parameters : {n_par:,}")
    print(f"Training   : mini-batch gradient descent, batch = {args.batch}, "
          f"eta = {args.eta}")
    print()

    def evaluate(documents, y, batch=500):
        p = np.concatenate([net.forward(documents[i:i + batch])
                            for i in range(0, len(documents), batch)])
        return net.loss(p, y), float(np.mean((p >= 0.5) == (y == 1))), p

    print("-" * 72)
    print("TRAINING")
    print("-" * 72)
    print(f"{'epoch':>6}{'train loss':>13}{'train acc':>11}"
          f"{'val loss':>11}{'val acc':>10}")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, args.epochs + 1):
        shuffled = rng.permutation(len(D_train))
        total_loss, correct, seen = 0.0, 0, 0
        for start in range(0, len(shuffled), args.batch):
            idx = shuffled[start:start + args.batch]
            batch_docs = [D_train[i] for i in idx]
            y_batch = y_train[idx]
            p = net.forward(batch_docs)            # forward
            net.backward(p, y_batch)               # backward
            net.update(args.eta)                   # gradient descent step
            total_loss += net.loss(p, y_batch) * len(idx)
            correct += int(np.sum((p >= 0.5) == (y_batch == 1)))
            seen += len(idx)
        vloss, vacc, _ = evaluate(D_val, y_val)
        history["train_loss"].append(total_loss / seen)
        history["train_acc"].append(correct / seen)
        history["val_loss"].append(vloss)
        history["val_acc"].append(vacc)
        print(f"{epoch:>6}{total_loss/seen:13.5f}{correct/seen:11.4f}"
              f"{vloss:11.5f}{vacc:10.4f}")
    print()

    test_loss, test_acc, p_test = evaluate(D_test, y_test)
    y_pred = (p_test >= 0.5).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_test == 1)))
    tn = int(np.sum((y_pred == 0) & (y_test == 0)))
    fp = int(np.sum((y_pred == 1) & (y_test == 0)))
    fn = int(np.sum((y_pred == 0) & (y_test == 1)))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    print("-" * 72)
    print("TEST PERFORMANCE")
    print("-" * 72)
    print(f"Loss      = {test_loss:.5f}")
    print(f"Accuracy  = {test_acc:.4f}   (random guessing = 0.5000)")
    print(f"Precision = {precision:.4f},  Recall = {recall:.4f},  F1 = {f1:.4f}")
    print(f"Confusion: TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}")
    print()

    ep = np.arange(1, args.epochs + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, history["train_loss"], "o-", ms=3, label="train")
    ax[0].plot(ep, history["val_loss"], "s-", ms=3, label="validation")
    ax[0].axhline(np.log(2), color="gray", ls=":", lw=1, label=r"$\log 2$")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("binary cross-entropy")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(ep, history["train_acc"], "o-", ms=3, label="train")
    ax[1].plot(ep, history["val_acc"], "s-", ms=3, label="validation")
    ax[1].axhline(test_acc, color="crimson", ls="--", lw=1,
                  label=f"test = {test_acc:.4f}")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    f1p = os.path.join(FIGDIR, "08_imdb_curves.pdf")
    fig.savefig(f1p)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    M = np.array([[tn, fp], [fn, tp]])
    im = ax[0].imshow(M, cmap="Blues")
    ax[0].set_xticks([0, 1], ["predicted 0", "predicted 1"])
    ax[0].set_yticks([0, 1], ["true 0", "true 1"])
    for i in range(2):
        for j in range(2):
            ax[0].text(j, i, M[i, j], ha="center", va="center",
                       color="white" if M[i, j] > M.max() / 2 else "black")
    ax[0].set_title(f"Test confusion matrix (accuracy = {test_acc:.4f})")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].hist(p_test[y_test == 0], bins=25, alpha=0.7, label="negative reviews")
    ax[1].hist(p_test[y_test == 1], bins=25, alpha=0.7, label="positive reviews")
    ax[1].axvline(0.5, color="k", ls="--", lw=1)
    ax[1].set_xlabel("predicted probability of being positive")
    ax[1].set_ylabel("count")
    ax[1].set_title("Predicted probabilities")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    f2p = os.path.join(FIGDIR, "08_imdb_results.pdf")
    fig.savefig(f2p)
    plt.close(fig)

    with open(os.path.join(RESDIR, "08_imdb.json"), "w") as fh:
        json.dump({"parameters": int(n_par), "vocab": args.vocab,
                   "dim": args.dim, "hidden": args.hidden,
                   "epochs": args.epochs, "eta": args.eta,
                   "n_train": len(D_train), "n_test": len(D_test),
                   "mean_length": float(lengths.mean()),
                   "test_acc": test_acc, "test_loss": test_loss,
                   "precision": precision, "recall": recall, "f1": f1,
                   "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                   "history": history}, fh, indent=2)
    print(f"Figure saved: {f1p}")
    print(f"Figure saved: {f2p}")
    print(f"Results saved: {os.path.join(RESDIR, '08_imdb.json')}")


if __name__ == "__main__":
    main()
