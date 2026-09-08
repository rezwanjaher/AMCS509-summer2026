"""
06 - Neural network: a character-level recurrent neural network (RNN) written
     from scratch in NumPy, trained on the Tiny Shakespeare corpus
     (Karpathy, char-rnn). Sequence modelling / text generation.

MATHEMATICS IMPLEMENTED HERE
----------------------------
One-hot encode each character, x_t in R^V (V = vocabulary size).

Recurrence (tanh activation, lecture 3-4):
    h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)
    o_t = W_hy h_t + b_y
    p_t = softmax(o_t)

Derivative of tanh (Task in lecture 3-4):
    tanh(z) = (e^z - e^-z)/(e^z + e^-z)
    d/dz tanh(z) = 1 - tanh^2(z)
proved by the quotient rule; this factor appears at every time step below.

Loss over a sequence of length T (cross-entropy of the next character):
    L = - sum_{t=1..T} log p_t[target_t]

Backpropagation Through Time. The same weights are reused at every step, so the
gradient of L with respect to a weight is the SUM of its contributions over
time - this is the multivariable chain rule
    dL/dW = sum_t (dL/dh_t)(dh_t/dW).
Explicitly, with do_t = p_t - y_t:
    dW_hy += do_t h_t^T,                 db_y += do_t
    dh_t   = W_hy^T do_t + dh_next
    dr_t   = (1 - h_t^2) * dh_t          (through the tanh)
    dW_xh += dr_t x_t^T,  dW_hh += dr_t h_{t-1}^T,  db_h += dr_t
    dh_next = W_hh^T dr_t                (passed to step t-1)

Repeated multiplication by W_hh^T is exactly why plain RNNs suffer from
exploding / vanishing gradients; here the gradients are clipped to [-5, 5],
which is the standard cheap remedy for the exploding half.

Optimiser: Adagrad, an adaptive per-parameter step size,
    G <- G + g^2,     theta <- theta - eta * g / sqrt(G + eps)
so that rarely-updated parameters (rare characters) still move.

Sampling: characters are drawn from p_t and fed back as the next input, so the
model generates text one character at a time.

Reported metric: cross-entropy per character and the derived perplexity
exp(L/T); a uniform model over V characters would score log V.
"""

import argparse
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
DATADIR = os.path.join(HERE, "data")
URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
       "tinyshakespeare/input.txt")


def load_text():
    os.makedirs(DATADIR, exist_ok=True)
    path = os.path.join(DATADIR, "tinyshakespeare.txt")
    if not os.path.exists(path):
        print(f"Downloading Tiny Shakespeare from {URL} ...")
        urllib.request.urlretrieve(URL, path)
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


class CharRNN:
    def __init__(self, V, H, seed=0):
        rng = np.random.default_rng(seed)
        self.V, self.H = V, H
        self.Wxh = rng.standard_normal((H, V)) * 0.01
        self.Whh = rng.standard_normal((H, H)) * 0.01
        self.Why = rng.standard_normal((V, H)) * 0.01
        self.bh = np.zeros((H, 1))
        self.by = np.zeros((V, 1))
        self.params = ["Wxh", "Whh", "Why", "bh", "by"]
        self.mem = {p: np.zeros_like(getattr(self, p)) for p in self.params}

    # ---- forward + BPTT over one sequence -----------------------------
    def loss_and_grads(self, inputs, targets, hprev):
        xs, hs, ps = {}, {-1: np.copy(hprev)}, {}
        loss = 0.0
        for t, ix in enumerate(inputs):                      # forward pass
            xs[t] = np.zeros((self.V, 1))
            xs[t][ix] = 1.0
            hs[t] = np.tanh(self.Wxh @ xs[t] + self.Whh @ hs[t - 1] + self.bh)
            o = self.Why @ hs[t] + self.by
            o -= o.max()
            e = np.exp(o)
            ps[t] = e / e.sum()
            loss += -np.log(ps[t][targets[t], 0] + 1e-12)

        g = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        dh_next = np.zeros((self.H, 1))
        for t in reversed(range(len(inputs))):               # backward pass
            do = np.copy(ps[t])
            do[targets[t]] -= 1.0                            # dL/do = p - y
            g["Why"] += do @ hs[t].T
            g["by"] += do
            dh = self.Why.T @ do + dh_next
            dr = (1.0 - hs[t] ** 2) * dh                     # through tanh
            g["bh"] += dr
            g["Wxh"] += dr @ xs[t].T
            g["Whh"] += dr @ hs[t - 1].T
            dh_next = self.Whh.T @ dr
        return loss, g, hs[len(inputs) - 1]

    def adagrad_step(self, g, eta, clip=5.0):
        for p in self.params:
            np.clip(g[p], -clip, clip, out=g[p])
            self.mem[p] += g[p] ** 2
            setattr(self, p, getattr(self, p) - eta * g[p] / np.sqrt(self.mem[p] + 1e-8))

    def sample(self, h, seed_ix, n, rng, temperature=1.0):
        x = np.zeros((self.V, 1))
        x[seed_ix] = 1.0
        out = []
        for _ in range(n):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            o = (self.Why @ h + self.by) / temperature
            o -= o.max()
            e = np.exp(o)
            p = (e / e.sum()).ravel()
            ix = int(rng.choice(self.V, p=p))
            x = np.zeros((self.V, 1))
            x[ix] = 1.0
            out.append(ix)
        return out


def gradient_check(net, inputs, targets, eps=1e-5, n_checks=8, seed=0):
    rng = np.random.default_rng(seed)
    h0 = np.zeros((net.H, 1))
    _, g, _ = net.loss_and_grads(inputs, targets, h0)
    errs = []
    for _ in range(n_checks):
        p = net.params[int(rng.integers(len(net.params)))]
        A = getattr(net, p)
        i = int(rng.integers(A.shape[0]))
        j = int(rng.integers(A.shape[1]))
        old = A[i, j]
        A[i, j] = old + eps
        lp, _, _ = net.loss_and_grads(inputs, targets, h0)
        A[i, j] = old - eps
        lm, _, _ = net.loss_and_grads(inputs, targets, h0)
        A[i, j] = old
        num = (lp - lm) / (2 * eps)
        ana = g[p][i, j]
        errs.append(abs(num - ana) / max(1e-12, abs(num) + abs(ana)))
    return float(np.max(errs))


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--seq", type=int, default=25, help="BPTT truncation length")
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(RESDIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    text = load_text()
    chars = sorted(set(text))
    V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=int)
    n_test = 100000
    train, test = data[:-n_test], data[-n_test:]

    print("=" * 72)
    print("06 | CHARACTER-LEVEL RNN FROM SCRATCH  -  Tiny Shakespeare")
    print("=" * 72)
    print(f"Corpus length   : {len(text):,} characters")
    print(f"Vocabulary V    : {V} distinct characters")
    print(f"Train / held-out: {len(train):,} / {len(test):,} characters")
    print(f"Hidden units H  : {args.hidden}, BPTT length T = {args.seq}")
    n_par = args.hidden * V + args.hidden ** 2 + V * args.hidden + args.hidden + V
    print(f"Parameters      : {n_par:,}")
    print(f"Chance-level loss (uniform over V) = log V = {np.log(V):.4f} nats/char")
    print()

    net = CharRNN(V, args.hidden, seed=args.seed)

    print("-" * 72)
    print("(a) GRADIENT CHECK of the BPTT equations (sequence of 12 characters)")
    print("-" * 72)
    small = CharRNN(V, 16, seed=args.seed)
    rel = gradient_check(small, list(train[:12]), list(train[1:13]), seed=args.seed)
    print(f"max relative error (analytic BPTT vs finite differences) = {rel:.3e}")
    zz = np.array([-1.5, -0.2, 0.0, 0.9, 2.0])
    print("max |d/dz tanh(z) - (1 - tanh^2 z)| = "
          f"{np.max(np.abs((np.tanh(zz+1e-6)-np.tanh(zz-1e-6))/2e-6 - (1-np.tanh(zz)**2))):.3e}")
    print()

    print("-" * 72)
    print(f"(b) TRAINING  (Adagrad, eta = {args.eta}, gradient clipping at +/-5)")
    print("-" * 72)
    ptr, h = 0, np.zeros((args.hidden, 1))
    smooth = -np.log(1.0 / V)
    hist_it, hist_loss = [], []
    t0 = time.time()
    for it in range(1, args.iters + 1):
        if ptr + args.seq + 1 >= len(train):
            ptr, h = 0, np.zeros((args.hidden, 1))
        inputs = list(train[ptr:ptr + args.seq])
        targets = list(train[ptr + 1:ptr + args.seq + 1])
        loss, g, h = net.loss_and_grads(inputs, targets, h)
        net.adagrad_step(g, args.eta)
        smooth = 0.999 * smooth + 0.001 * (loss / args.seq)
        ptr += args.seq
        if it % 500 == 0:
            hist_it.append(it)
            hist_loss.append(smooth)
        if it % 10000 == 0 or it == 1000:
            print(f"  iter {it:>7} | smoothed loss = {smooth:.4f} nats/char "
                  f"| perplexity = {np.exp(smooth):7.3f} | {time.time()-t0:6.1f}s")
            txt = "".join(chars[i] for i in
                          net.sample(np.copy(h), int(inputs[0]), 220, rng))
            print("  ---- sample ----")
            for line in txt.split("\n"):
                print("   |" + line)
            print("  ----------------")
    print()

    # --- held-out evaluation ---------------------------------------------
    hh = np.zeros((args.hidden, 1))
    tot, cnt, correct = 0.0, 0, 0
    for s in range(0, len(test) - args.seq - 1, args.seq):
        inp = list(test[s:s + args.seq])
        tgt = list(test[s + 1:s + args.seq + 1])
        l, _, hh = net.loss_and_grads(inp, tgt, hh)
        tot += l
        cnt += args.seq
    heldout = tot / cnt
    print("-" * 72)
    print("(c) HELD-OUT PERFORMANCE")
    print("-" * 72)
    print(f"Cross-entropy   = {heldout:.4f} nats/character")
    print(f"Bits per char   = {heldout/np.log(2):.4f}")
    print(f"Perplexity      = {np.exp(heldout):.4f}   (uniform baseline = {V})")
    print(f"Final smoothed training loss = {smooth:.4f} nats/character")
    print()

    final_sample = "".join(chars[i] for i in
                           net.sample(np.zeros((args.hidden, 1)), stoi["T"], 600, rng))
    print("Final 600-character sample:")
    print("-" * 72)
    print(final_sample)
    print("-" * 72)
    print()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist_it, hist_loss, lw=1.4)
    ax[0].axhline(np.log(V), color="gray", ls=":", lw=1, label=r"$\log V$ (uniform)")
    ax[0].axhline(heldout, color="crimson", ls="--", lw=1,
                  label=f"held-out = {heldout:.3f}")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("nats per character")
    ax[0].set_title("Smoothed training loss (Adagrad + BPTT)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(hist_it, np.exp(hist_loss), lw=1.4, color="tab:orange")
    ax[1].axhline(V, color="gray", ls=":", lw=1, label=f"uniform = {V}")
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("perplexity")
    ax[1].set_yscale("log")
    ax[1].set_title("Perplexity")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    f1p = os.path.join(FIGDIR, "06_shakespeare_loss.pdf")
    fig.savefig(f1p)
    plt.close(fig)

    # singular values of W_hh: the exploding / vanishing gradient diagnostic
    sv = np.linalg.svd(net.Whh, compute_uv=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(sv, "o-", ms=3)
    ax[0].axhline(1.0, color="crimson", ls="--", lw=1, label=r"$\sigma=1$")
    ax[0].set_xlabel("index")
    ax[0].set_ylabel(r"singular value of $W_{hh}$")
    ax[0].set_title(rf"$\sigma_{{\max}}={sv.max():.3f}$: gradient growth per step")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    im = ax[1].imshow(net.Why, aspect="auto", cmap="RdBu_r")
    ax[1].set_xlabel("hidden unit")
    ax[1].set_ylabel("character index")
    ax[1].set_title(r"Output weights $W_{hy}$")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    f2p = os.path.join(FIGDIR, "06_shakespeare_weights.pdf")
    fig.savefig(f2p)
    plt.close(fig)

    with open(os.path.join(RESDIR, "06_shakespeare.json"), "w") as fh:
        json.dump({"vocab_size": V, "hidden": args.hidden, "seq": args.seq,
                   "iters": args.iters, "eta": args.eta, "parameters": int(n_par),
                   "grad_check_rel_err": rel,
                   "final_train_loss": float(smooth),
                   "heldout_loss": float(heldout),
                   "heldout_bpc": float(heldout / np.log(2)),
                   "heldout_perplexity": float(np.exp(heldout)),
                   "uniform_loss": float(np.log(V)),
                   "sigma_max_Whh": float(sv.max()),
                   "sample": final_sample,
                   "loss_curve": {"iter": hist_it, "loss": [float(v) for v in hist_loss]}},
                  fh, indent=2)
    with open(os.path.join(RESDIR, "06_sample.txt"), "w") as fh:
        fh.write(final_sample)
    print(f"Figure saved: {f1p}")
    print(f"Figure saved: {f2p}")
    print(f"Results saved: {os.path.join(RESDIR, '06_shakespeare.json')}")


if __name__ == "__main__":
    main()
