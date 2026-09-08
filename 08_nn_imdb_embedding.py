"""
08 - Neural network: sentiment classification of IMDB movie reviews with an
     averaged-embedding network (a "deep averaging network") written from
     scratch in NumPy.

Data: the official Stanford aclImdb archive is tried first; if that host is
unreachable the script falls back to a GitHub mirror of the same 50,000
labelled reviews. Parsed arrays are cached in ./data.

MODEL
-----
    review -> token indices -> embedding lookup -> mean over tokens
           -> dense(64) -> ReLU -> dense(1) -> sigmoid -> P(positive)

MATHEMATICS IMPLEMENTED HERE
----------------------------
1. Representation. A review is a variable-length sequence of token indices
   t_1..t_T. Each index selects a row of the embedding matrix E in R^{V x d};
   the lookup E[t] is exactly the matrix product e_t^T E with e_t the one-hot
   vector of the token, so an embedding layer is a dense layer whose input is
   one-hot. Implementing it as a lookup rather than a matrix product avoids
   multiplying by 20,000-dimensional vectors that are zero almost everywhere.

2. Mean pooling handles the variable length:

       u = (1/T) sum_{t=1..T} E[t_t]      in R^d

   This is a linear map, so its derivative is trivial: each token receives the
   same share of the incoming gradient,

       dL/dE[t_k] += (1/T) dL/du        for every k = 1..T.

   The "+=" matters. A word appearing several times in one review, or in many
   reviews of a batch, accumulates a contribution from each occurrence - the
   same weight-tying sum seen over space in the CNN (script 07) and over time in
   the RNN (script 06). Only the rows of E for tokens actually present receive a
   gradient, so the update is sparse: this is why Adagrad, which gives every
   parameter its own step size

       G <- G + g^2,    theta <- theta - eta g / sqrt(G + eps),

   is the natural optimiser here - a word occurring in 0.1% of reviews would
   otherwise never move under a single global learning rate.

3. Head and loss. The rest is the binary classifier of script 04 stacked on
   top of a hidden ReLU layer: with z = w^T a + b and p = sigma(z), the binary
   cross-entropy gives dL/dz = (p - y)/B by the same cancellation of
   sigma' = sigma(1 - sigma) derived there, and the hidden layer follows the
   backpropagation recursion of script 05.

4. Early stopping. The validation loss is monitored each epoch and the
   parameters of the best epoch are restored at the end. Gradient descent
   traverses a path from the simple initial model to increasingly complex fits;
   stopping partway is a regularisation that limits the effective capacity
   without changing the objective, and it is the cheapest defence against the
   overfitting that 1.3 million parameters on 25,000 reviews invites.

5. What this model can and cannot represent. Averaging destroys word order, so
   the model is a learned weighted bag of words: "not good" and "good not"
   receive identical representations. It nevertheless does well on sentiment,
   because the task is largely lexical. Capturing negation and scope needs
   order-sensitive architecture (the recurrence of script 06, or attention).

Gradients for every layer are verified against central finite differences.
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
TOKEN = re.compile(r"[a-z']+")


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def _from_official():
    tgz = os.path.join(DATADIR, "aclImdb_v1.tar.gz")
    if not os.path.exists(tgz):
        print(f"  trying official source {OFFICIAL} ...")
        urllib.request.urlretrieve(OFFICIAL, tgz)
    texts, labels = [], []
    with tarfile.open(tgz) as tf:
        for m in tf.getmembers():
            parts = m.name.split("/")
            if len(parts) == 4 and parts[1] in ("train", "test") \
                    and parts[2] in ("pos", "neg") and m.isfile():
                texts.append(tf.extractfile(m).read().decode("utf-8", "ignore"))
                labels.append(1 if parts[2] == "pos" else 0)
    return texts, np.array(labels)


def _from_mirror():
    import pandas as pd
    path = os.path.join(DATADIR, "imdb.csv")
    if not os.path.exists(path):
        print(f"  official host unreachable; downloading mirror {MIRROR} ...")
        urllib.request.urlretrieve(MIRROR, path)
    df = pd.read_csv(path)
    txt_col = "review" if "review" in df.columns else df.columns[0]
    lab_col = "sentiment" if "sentiment" in df.columns else df.columns[1]
    lab = df[lab_col].map(lambda v: 1 if str(v).strip().lower() in
                          ("positive", "pos", "1") else 0).to_numpy()
    return df[txt_col].astype(str).tolist(), lab


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


def build_vocab(token_lists, V):
    """Keep the V-2 most frequent tokens; index 0 = padding/unknown."""
    cnt = Counter()
    for toks in token_lists:
        cnt.update(toks)
    words = [w for w, _ in cnt.most_common(V - 1)]
    return {w: i + 1 for i, w in enumerate(words)}, cnt


# ----------------------------------------------------------------------
# Core mathematics
# ----------------------------------------------------------------------
def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos, neg = z >= 0, z < 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[neg])
    out[neg] = e / (1.0 + e)
    return out


class AvgEmbeddingNet:
    def __init__(self, V, d, hidden, seed=0):
        rng = np.random.default_rng(seed)
        self.E = rng.standard_normal((V, d)) * 0.05
        self.W1 = rng.standard_normal((d, hidden)) * np.sqrt(2.0 / d)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, 1)) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(1)
        self.names = ["E", "W1", "b1", "W2", "b2"]
        self.G = {n: np.zeros_like(getattr(self, n)) for n in self.names}

    def pool(self, docs):
        """U[i] = mean of the embedding rows of document i."""
        U = np.zeros((len(docs), self.E.shape[1]))
        for i, idx in enumerate(docs):
            U[i] = self.E[idx].mean(axis=0)
        return U

    def forward(self, docs):
        U = self.pool(docs)
        Z1 = U @ self.W1 + self.b1
        A1 = np.maximum(0.0, Z1)
        Z2 = A1 @ self.W2 + self.b2
        P = sigmoid(Z2).ravel()
        self.cache = (docs, U, Z1, A1)
        return P

    @staticmethod
    def bce(P, y):
        return float(-np.mean(y * np.log(P + 1e-12) + (1 - y) * np.log(1 - P + 1e-12)))

    def backward(self, P, y, lam=0.0):
        docs, U, Z1, A1 = self.cache
        B = len(y)
        dZ2 = ((P - y) / B)[:, None]                 # sigmoid + BCE cancellation
        g = {}
        g["W2"] = A1.T @ dZ2 + 2 * lam * self.W2
        g["b2"] = dZ2.sum(axis=0)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)                          # ReLU derivative
        g["W1"] = U.T @ dZ1 + 2 * lam * self.W1
        g["b1"] = dZ1.sum(axis=0)
        dU = dZ1 @ self.W1.T
        gE = np.zeros_like(self.E)
        for i, idx in enumerate(docs):                # scatter-add over tokens
            np.add.at(gE, idx, dU[i] / len(idx))
        g["E"] = gE
        return g

    def adagrad_step(self, g, eta):
        for n in self.names:
            self.G[n] += g[n] ** 2
            setattr(self, n, getattr(self, n) - eta * g[n] / np.sqrt(self.G[n] + 1e-8))


def gradient_check(net, docs, y, eps=1e-6, n=6, seed=0):
    rng = np.random.default_rng(seed)
    P = net.forward(docs)
    g = net.backward(P, y)
    errs = {}
    for name in net.names:
        A = np.atleast_2d(getattr(net, name))
        e = []
        for _ in range(n):
            i = int(rng.integers(A.shape[0]))
            j = int(rng.integers(A.shape[1]))
            old = A[i, j]
            A[i, j] = old + eps
            lp = net.bce(net.forward(docs), y)
            A[i, j] = old - eps
            lm = net.bce(net.forward(docs), y)
            A[i, j] = old
            num = (lp - lm) / (2 * eps)
            ana = np.atleast_2d(g[name])[i, j]
            e.append(abs(num - ana) / max(1e-12, abs(num) + abs(ana)))
        errs[name] = float(np.max(e))
    return errs


def roc_auc(y, s):
    order = np.argsort(-s)
    y = y[order]
    P, N = y.sum(), len(y) - y.sum()
    tpr = np.concatenate([[0], np.cumsum(y) / P])
    fpr = np.concatenate([[0], np.cumsum(1 - y) / N])
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return fpr, tpr, float(trap(tpr, fpr))


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(RESDIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 72)
    print("08 | AVERAGED-EMBEDDING NETWORK FROM SCRATCH  -  IMDB reviews")
    print("=" * 72)
    texts, labels = load_imdb()
    toks = [TOKEN.findall(t.lower()) for t in texts]
    lens = np.array([len(t) for t in toks])
    print(f"Reviews         : {len(texts):,}  "
          f"({int(labels.sum()):,} positive, {int((1-labels).sum()):,} negative)")
    print(f"Review length   : mean {lens.mean():.1f}, median {int(np.median(lens))}, "
          f"max {lens.max()} tokens")

    perm = rng.permutation(len(texts))
    n_tr = int(0.5 * len(texts))
    n_va = int(0.1 * len(texts))
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    vocab, cnt = build_vocab([toks[i] for i in tr], args.vocab)
    print(f"Distinct tokens : {len(cnt):,} in the training split; "
          f"vocabulary capped at V = {args.vocab:,}")
    cover = sum(cnt[w] for w in vocab) / sum(cnt.values())
    print(f"Token coverage  : {cover:.4f} of all training tokens are in vocabulary")

    def encode(i):
        idx = [vocab[w] for w in toks[i][:args.max_len] if w in vocab]
        return np.array(idx if idx else [0], dtype=int)

    D = [encode(i) for i in range(len(texts))]
    Dtr = [D[i] for i in tr]
    Dva = [D[i] for i in va]
    Dte = [D[i] for i in te]
    ytr, yva, yte = labels[tr].astype(float), labels[va].astype(float), labels[te].astype(float)
    print(f"Train / validation / test : {len(Dtr):,} / {len(Dva):,} / {len(Dte):,}")

    net = AvgEmbeddingNet(args.vocab, args.dim, args.hidden, seed=args.seed)
    n_par = net.E.size + net.W1.size + net.b1.size + net.W2.size + net.b2.size
    print(f"Model : embedding {args.vocab}x{args.dim} -> mean -> dense({args.hidden}) "
          f"-> ReLU -> dense(1) -> sigmoid")
    print(f"Parameters : {n_par:,} "
          f"(of which {net.E.size:,} = {100*net.E.size/n_par:.1f}% are embeddings)")
    print(f"Optimiser  : Adagrad, eta = {args.eta}, batch = {args.batch}, "
          f"max length {args.max_len} tokens")
    print()

    print("-" * 72)
    print("(a) GRADIENT CHECK of every parameter block (16 reviews)")
    print("-" * 72)
    errs = gradient_check(AvgEmbeddingNet(args.vocab, args.dim, args.hidden,
                                          seed=args.seed),
                          Dtr[:16], ytr[:16], seed=args.seed)
    for k, v in errs.items():
        print(f"   {k:>3}: max relative error = {v:.3e}")
    print("   (E verifies the sparse scatter-add through the mean pooling)")
    print()

    print("-" * 72)
    print("(b) TRAINING")
    print("-" * 72)
    print(f"{'epoch':>6}{'train BCE':>12}{'train acc':>11}{'val BCE':>10}{'val acc':>10}")
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_vl, best_ep = np.inf, 0
    best_state = {n: np.copy(getattr(net, n)) for n in net.names}

    def evaluate(docs, y, batch=500):
        P = np.concatenate([net.forward(docs[i:i + batch])
                            for i in range(0, len(docs), batch)])
        return net.bce(P, y), float(np.mean((P >= 0.5) == (y == 1))), P

    for ep in range(1, args.epochs + 1):
        order = rng.permutation(len(Dtr))
        rl, rc, seen = 0.0, 0, 0
        for s in range(0, len(order), args.batch):
            idx = order[s:s + args.batch]
            docs = [Dtr[i] for i in idx]
            yb = ytr[idx]
            P = net.forward(docs)
            net.adagrad_step(net.backward(P, yb, args.lam), args.eta)
            rl += net.bce(P, yb) * len(idx)
            rc += int(np.sum((P >= 0.5) == (yb == 1)))
            seen += len(idx)
        vl, vacc, _ = evaluate(Dva, yva)
        hist["train_loss"].append(rl / seen)
        hist["train_acc"].append(rc / seen)
        hist["val_loss"].append(vl)
        hist["val_acc"].append(vacc)
        mark = ""
        if vl < best_vl:
            best_vl, best_ep = vl, ep
            best_state = {n: np.copy(getattr(net, n)) for n in net.names}
            mark = "  <- best so far"
        print(f"{ep:>6}{rl/seen:12.5f}{rc/seen:11.4f}{vl:10.5f}{vacc:10.4f}{mark}")
    for n in net.names:                      # early stopping: restore the best
        setattr(net, n, best_state[n])
    print(f"\nEarly stopping: parameters restored to epoch {best_ep} "
          f"(lowest validation loss {best_vl:.5f}).")
    print("Training loss keeps falling after that epoch while the validation loss")
    print("rises - the model starts memorising individual reviews, and stopping")
    print("early is the cheapest form of regularisation.")
    print()

    tl, tacc, Pte = evaluate(Dte, yte)
    yhat = (Pte >= 0.5).astype(int)
    tp = int(np.sum((yhat == 1) & (yte == 1)))
    tn = int(np.sum((yhat == 0) & (yte == 0)))
    fp = int(np.sum((yhat == 1) & (yte == 0)))
    fn = int(np.sum((yhat == 0) & (yte == 1)))
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    fpr, tpr, auc = roc_auc(yte.astype(int), Pte)
    print("-" * 72)
    print("(c) TEST PERFORMANCE")
    print("-" * 72)
    print(f"Cross-entropy = {tl:.5f}")
    print(f"Accuracy      = {tacc:.4f}   (chance = 0.5000)")
    print(f"Precision     = {prec:.4f},  Recall = {rec:.4f},  F1 = {f1:.4f}")
    print(f"AUC           = {auc:.4f}")
    print(f"Confusion: TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}")
    print()

    # --- what the embedding learned ---------------------------------------
    # project each word onto the output direction of the network to obtain a
    # scalar "sentiment loading": s_w = ReLU(E_w W1 + b1) W2  (the model's own
    # score for a one-word review)
    inv = {i: w for w, i in vocab.items()}
    common = [i for i in range(1, args.vocab)
              if inv.get(i) and cnt[inv[i]] >= 200]
    Ew = net.E[common]
    score = (np.maximum(0.0, Ew @ net.W1 + net.b1) @ net.W2).ravel() + net.b2[0]
    order = np.argsort(score)
    print("-" * 72)
    print("(d) MOST POLARISED WORDS (single-word score of the trained network,")
    print("    restricted to words occurring at least 200 times)")
    print("-" * 72)
    neg = [inv[common[k]] for k in order[:15]]
    pos = [inv[common[k]] for k in order[-15:]][::-1]
    print("  most negative: " + ", ".join(neg))
    print("  most positive: " + ", ".join(pos))
    print()

    ep = np.arange(1, args.epochs + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, hist["train_loss"], "o-", ms=3, label="train")
    ax[0].plot(ep, hist["val_loss"], "s-", ms=3, label="validation")
    ax[0].axhline(np.log(2), color="gray", ls=":", lw=1, label=r"$\log 2$ (chance)")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("binary cross-entropy")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(ep, hist["train_acc"], "o-", ms=3, label="train")
    ax[1].plot(ep, hist["val_acc"], "s-", ms=3, label="validation")
    ax[1].axhline(tacc, color="crimson", ls="--", lw=1, label=f"test = {tacc:.4f}")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    f1p = os.path.join(FIGDIR, "08_imdb_curves.pdf")
    fig.savefig(f1p)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].plot(fpr, tpr, lw=1.8, label=f"AUC = {auc:.4f}")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax[0].set_xlabel("false positive rate")
    ax[0].set_ylabel("true positive rate")
    ax[0].set_title("ROC curve (test set)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    show = neg[:10][::-1] + pos[:10][::-1]
    vals = list(np.sort(score)[:10][::-1]) + list(np.sort(score)[-10:])
    colors = ["tab:red"] * 10 + ["tab:green"] * 10
    ax[1].barh(range(len(show)), vals, color=colors)
    ax[1].set_yticks(range(len(show)), show, fontsize=7)
    ax[1].set_xlabel("single-word network score (logit)")
    ax[1].set_title("Most polarised vocabulary items")
    ax[1].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    f2p = os.path.join(FIGDIR, "08_imdb_roc_words.pdf")
    fig.savefig(f2p)
    plt.close(fig)

    with open(os.path.join(RESDIR, "08_imdb.json"), "w") as fh:
        json.dump({"parameters": int(n_par), "vocab": args.vocab, "dim": args.dim,
                   "hidden": args.hidden, "max_len": args.max_len,
                   "epochs": args.epochs, "eta": args.eta,
                   "n_train": len(Dtr), "n_val": len(Dva), "n_test": len(Dte),
                   "token_coverage": float(cover),
                   "mean_length": float(lens.mean()),
                   "grad_check": errs, "test_acc": tacc, "test_bce": tl,
                   "precision": prec, "recall": rec, "f1": f1, "auc": auc,
                   "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                   "best_epoch": int(best_ep), "best_val_loss": float(best_vl),
                   "final_train_acc": hist["train_acc"][-1],
                   "final_val_acc": hist["val_acc"][-1],
                   "history": hist,
                   "most_negative": neg, "most_positive": pos}, fh, indent=2)
    print(f"Figure saved: {f1p}")
    print(f"Figure saved: {f2p}")
    print(f"Results saved: {os.path.join(RESDIR, '08_imdb.json')}")


if __name__ == "__main__":
    main()
