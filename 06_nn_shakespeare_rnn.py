"""
06 - Neural network: a character-level Recurrent Neural Network (RNN) written
     from scratch in NumPy and trained on the Tiny Shakespeare text.

The task: given the characters seen so far, predict the next character. After
training we can generate new text by feeding each prediction back in as the
next input.

======================================================================
1. REPRESENTING THE TEXT
======================================================================
There are V distinct characters in the file. Character number k is written as a
one-hot vector x with x[k] = 1 and every other entry 0.

======================================================================
2. FORWARD PASS (one time step)
======================================================================
    h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)      hidden state
    y_t = W_hy h_t + b_y                           scores for the next character
    p_t = softmax(y_t)                             probabilities

h_t is the network's memory: it depends on the current character AND on the
previous hidden state, which is what makes this a recurrent network.

Derivative of tanh (needed below):

    tanh'(z) = 1 - tanh(z)^2

======================================================================
3. LOSS
======================================================================
Over a sequence of T characters,

    L = - sum_t log p_t[correct next character]

which is the cross-entropy again, summed over the time steps.

======================================================================
4. BACKWARD PASS: BACKPROPAGATION THROUGH TIME
======================================================================
The SAME weight matrices are used at every time step, so a weight affects the
loss through every step. The chain rule then says the gradient is the SUM of
the contributions from all time steps. Going backwards from t = T to t = 1,
with dy_t = p_t - y_onehot_t (the usual softmax + cross-entropy result):

    dW_hy += dy_t h_t^T                 db_y += dy_t
    dh     = W_hy^T dy_t + dh_next      (gradient from this step + from the future)
    dr     = (1 - h_t^2) * dh           (through the tanh)
    dW_xh += dr x_t^T
    dW_hh += dr h_{t-1}^T
    db_h  += dr
    dh_next = W_hh^T dr                 (passed back to step t-1)

Because dh_next is multiplied by W_hh at every step, going back many steps
multiplies by W_hh over and over: the gradient can grow very large (exploding)
or shrink to nothing (vanishing). We handle the exploding case by CLIPPING
every gradient into [-5, 5] before the update.

======================================================================
5. UPDATE
======================================================================
Standard gradient descent on the clipped gradients:

    W <- W - eta * dW
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


# ======================================================================
# THE NETWORK
# ======================================================================
class CharRNN:
    def __init__(self, V, H, seed=0):
        rng = np.random.default_rng(seed)
        self.V, self.H = V, H
        self.W_xh = rng.standard_normal((H, V)) * 0.01   # input   -> hidden
        self.W_hh = rng.standard_normal((H, H)) * 0.01   # hidden  -> hidden
        self.W_hy = rng.standard_normal((V, H)) * 0.01   # hidden  -> output
        self.b_h = np.zeros((H, 1))
        self.b_y = np.zeros((V, 1))

    # ---------------- forward + BPTT over one short sequence ------------
    def loss_and_gradients(self, inputs, targets, h_prev):
        xs, hs, ps = {}, {-1: np.copy(h_prev)}, {}
        loss = 0.0

        # ---------- FORWARD: run through the sequence -------------------
        for t, char_index in enumerate(inputs):
            xs[t] = np.zeros((self.V, 1))
            xs[t][char_index] = 1.0                      # one-hot input
            hs[t] = np.tanh(self.W_xh @ xs[t]
                            + self.W_hh @ hs[t - 1]
                            + self.b_h)                  # hidden state
            scores = self.W_hy @ hs[t] + self.b_y        # scores
            scores = scores - scores.max()               # for a safe exp
            exp_scores = np.exp(scores)
            ps[t] = exp_scores / exp_scores.sum()        # softmax
            loss += -np.log(ps[t][targets[t], 0] + 1e-12)

        # ---------- BACKWARD: go back through the sequence ---------------
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        dh_next = np.zeros((self.H, 1))

        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1.0                        # softmax + CE: p - y
            dW_hy += dy @ hs[t].T
            db_y += dy
            dh = self.W_hy.T @ dy + dh_next              # from now + from later
            dr = (1.0 - hs[t] ** 2) * dh                 # through the tanh
            db_h += dr
            dW_xh += dr @ xs[t].T
            dW_hh += dr @ hs[t - 1].T
            dh_next = self.W_hh.T @ dr                   # pass to the step before

        # average over the T characters of the sequence, so that the loss and
        # the gradients are "per character" and do not depend on T
        T = len(inputs)
        grads = [g / T for g in (dW_xh, dW_hh, dW_hy, db_h, db_y)]
        return loss / T, grads, hs[len(inputs) - 1]

    # ---------------- gradient descent step -----------------------------
    def update(self, grads, eta, clip=5.0):
        """Clip each gradient into [-5, 5], then W <- W - eta * dW."""
        names = ["W_xh", "W_hh", "W_hy", "b_h", "b_y"]
        for name, g in zip(names, grads):
            g = np.clip(g, -clip, clip)
            setattr(self, name, getattr(self, name) - eta * g)

    # ---------------- generating text -----------------------------------
    def sample(self, h, first_index, n, rng):
        """Feed each predicted character back in as the next input."""
        x = np.zeros((self.V, 1))
        x[first_index] = 1.0
        out = []
        for _ in range(n):
            h = np.tanh(self.W_xh @ x + self.W_hh @ h + self.b_h)
            scores = self.W_hy @ h + self.b_y
            scores = scores - scores.max()
            p = np.exp(scores)
            p = (p / p.sum()).ravel()
            index = int(rng.choice(self.V, p=p))         # draw from the softmax
            x = np.zeros((self.V, 1))
            x[index] = 1.0
            out.append(index)
        return out


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--seq", type=int, default=25, help="characters per update")
    ap.add_argument("--eta", type=float, default=0.1, help="learning rate")
    ap.add_argument("--iters", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(RESDIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    text = load_text()
    chars = sorted(set(text))
    V = len(chars)
    char_to_index = {c: i for i, c in enumerate(chars)}
    data = np.array([char_to_index[c] for c in text], dtype=int)
    n_heldout = 100000
    train_data, heldout_data = data[:-n_heldout], data[-n_heldout:]

    print("=" * 72)
    print("06 | CHARACTER-LEVEL RNN FROM SCRATCH  -  Tiny Shakespeare")
    print("=" * 72)
    print(f"Text length     : {len(text):,} characters")
    print(f"Distinct chars V: {V}")
    print(f"Train / held out: {len(train_data):,} / {len(heldout_data):,}")
    print(f"Hidden units H  : {args.hidden}, sequence length T = {args.seq}")
    n_par = args.hidden * V + args.hidden ** 2 + V * args.hidden + args.hidden + V
    print(f"Parameters      : {n_par:,}")
    print(f"If the model guessed uniformly the loss would be "
          f"log V = {np.log(V):.4f} per character.")
    print()

    net = CharRNN(V, args.hidden, seed=args.seed)

    print("-" * 72)
    print(f"(a) TRAINING  (gradient descent, eta = {args.eta}, "
          "gradients clipped to [-5, 5])")
    print("-" * 72)
    pointer = 0
    h = np.zeros((args.hidden, 1))
    smooth_loss = np.log(V)                 # start from the uniform-guess value
    iters_recorded, losses_recorded = [], []
    t0 = time.time()

    for it in range(1, args.iters + 1):
        # take the next chunk of characters; wrap around at the end
        if pointer + args.seq + 1 >= len(train_data):
            pointer = 0
            h = np.zeros((args.hidden, 1))
        inputs = list(train_data[pointer:pointer + args.seq])
        targets = list(train_data[pointer + 1:pointer + args.seq + 1])

        loss, grads, h = net.loss_and_gradients(inputs, targets, h)
        net.update(grads, args.eta)
        pointer += args.seq

        # running average of the loss per character, so the curve is readable
        smooth_loss = 0.999 * smooth_loss + 0.001 * loss
        if it % 500 == 0:
            iters_recorded.append(it)
            losses_recorded.append(smooth_loss)
        if it % 10000 == 0 or it == 1000:
            print(f"  iteration {it:>7} | loss = {smooth_loss:.4f} per character "
                  f"| {time.time() - t0:6.1f}s")
            sample = "".join(chars[i] for i in
                             net.sample(np.copy(h), int(inputs[0]), 200, rng))
            print("  ---- sample ----")
            for line in sample.split("\n"):
                print("   |" + line)
            print("  ----------------")
    print()

    # ---------------- held-out evaluation ---------------------------------
    h_eval = np.zeros((args.hidden, 1))
    total, count = 0.0, 0
    for start in range(0, len(heldout_data) - args.seq - 1, args.seq):
        inputs = list(heldout_data[start:start + args.seq])
        targets = list(heldout_data[start + 1:start + args.seq + 1])
        loss, _, h_eval = net.loss_and_gradients(inputs, targets, h_eval)
        total += loss * args.seq
        count += args.seq
    heldout_loss = total / count

    print("-" * 72)
    print("(b) HELD-OUT PERFORMANCE")
    print("-" * 72)
    print(f"Cross-entropy = {heldout_loss:.4f} per character")
    print(f"Final training loss = {smooth_loss:.4f} per character")
    print(f"Uniform guessing would give {np.log(V):.4f}")
    print()

    final_sample = "".join(chars[i] for i in
                           net.sample(np.zeros((args.hidden, 1)),
                                      char_to_index["T"], 600, rng))
    print("Final 600-character sample:")
    print("-" * 72)
    print(final_sample)
    print("-" * 72)
    print()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(iters_recorded, losses_recorded, lw=1.4)
    ax.axhline(np.log(V), color="gray", ls=":", lw=1,
               label=rf"$\log V = {np.log(V):.2f}$ (uniform guessing)")
    ax.axhline(heldout_loss, color="crimson", ls="--", lw=1,
               label=f"held out = {heldout_loss:.3f}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss per character")
    ax.set_title("Training a character RNN")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f1p = os.path.join(FIGDIR, "06_shakespeare_loss.pdf")
    fig.savefig(f1p)
    plt.close(fig)

    with open(os.path.join(RESDIR, "06_shakespeare.json"), "w") as fh:
        json.dump({"vocab_size": V, "hidden": args.hidden, "seq": args.seq,
                   "iterations": args.iters, "eta": args.eta,
                   "parameters": int(n_par),
                   "final_train_loss": float(smooth_loss),
                   "heldout_loss": float(heldout_loss),
                   "uniform_loss": float(np.log(V)),
                   "sample": final_sample,
                   "loss_curve": {"iteration": iters_recorded,
                                  "loss": [float(v) for v in losses_recorded]}},
                  fh, indent=2)
    with open(os.path.join(RESDIR, "06_sample.txt"), "w") as fh:
        fh.write(final_sample)
    print(f"Figure saved: {f1p}")
    print(f"Results saved: {os.path.join(RESDIR, '06_shakespeare.json')}")


if __name__ == "__main__":
    main()
