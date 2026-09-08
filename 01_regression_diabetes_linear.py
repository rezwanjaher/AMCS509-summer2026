"""
01 - Regression: Linear Regression on the Diabetes dataset (scikit-learn).

Two ways of fitting the same model are compared:
  (a) the closed-form normal equation, and
  (b) standard gradient descent.
Both should give the same answer.

----------------------------------------------------------------------
THE MODEL
----------------------------------------------------------------------
    y_hat_i = w^T x_i + b

To avoid carrying b separately we add a column of ones to X:

    X_tilde = [1, X]          theta = [b, w]          y_hat = X_tilde theta

----------------------------------------------------------------------
THE LOSS: MEAN SQUARED ERROR
----------------------------------------------------------------------
    L(theta) = (1/N) sum_i (y_i - x_tilde_i^T theta)^2
             = (1/N) || y - X_tilde theta ||^2

----------------------------------------------------------------------
THE GRADIENT
----------------------------------------------------------------------
Expanding the square and differentiating with respect to theta,

    dL/dtheta = -(2/N) X_tilde^T (y - X_tilde theta)

which is the input matrix transposed, times the residual (y - y_hat).

----------------------------------------------------------------------
(a) CLOSED FORM: THE NORMAL EQUATION
----------------------------------------------------------------------
Setting dL/dtheta = 0:

    X_tilde^T X_tilde theta = X_tilde^T y
    theta = (X_tilde^T X_tilde)^{-1} X_tilde^T y

Geometrically this says the residual is perpendicular to every column of
X_tilde: the fitted values are the projection of y onto the column space.

----------------------------------------------------------------------
(b) GRADIENT DESCENT
----------------------------------------------------------------------
    theta <- theta - eta * dL/dtheta

repeated for a fixed number of iterations. It reaches the same solution as
the normal equation, just iteratively.

----------------------------------------------------------------------
METRICS
----------------------------------------------------------------------
    MSE = (1/n) sum (y - y_hat)^2
    MAE = (1/n) sum |y - y_hat|
    R^2 = 1 - SS_res / SS_tot     (1.0 = perfect, 0.0 = no better than the mean)
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
RESDIR = os.path.join(HERE, "results")


# ======================================================================
# THE MATHEMATICS
# ======================================================================
def add_bias_column(X):
    """X_tilde = [1, X], so that the bias b becomes theta[0]."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def predict(X, theta):
    """y_hat = X_tilde theta."""
    return add_bias_column(X) @ theta


def mse(y, y_hat):
    return float(np.mean((y - y_hat) ** 2))


def mae(y, y_hat):
    return float(np.mean(np.abs(y - y_hat)))


def r2_score(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def fit_normal_equation(X, y):
    """theta = (X^T X)^{-1} X^T y."""
    Xt = add_bias_column(X)
    return np.linalg.inv(Xt.T @ Xt) @ Xt.T @ y


def gradient(Xt, y, theta):
    """dL/dtheta = -(2/N) X^T (y - X theta)."""
    N = Xt.shape[0]
    residual = y - Xt @ theta
    return -(2.0 / N) * (Xt.T @ residual)


def fit_gradient_descent(X, y, eta, n_iter):
    """theta <- theta - eta * dL/dtheta, repeated n_iter times."""
    Xt = add_bias_column(X)
    theta = np.zeros(Xt.shape[1])
    history = [mse(y, Xt @ theta)]
    for _ in range(n_iter):
        theta = theta - eta * gradient(Xt, y, theta)
        history.append(mse(y, Xt @ theta))
    return theta, np.array(history)


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", type=float, default=0.9, help="learning rate")
    ap.add_argument("--iters", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(RESDIR, exist_ok=True)

    data = load_diabetes()
    X, y = data.data, data.target
    names = list(data.feature_names)

    print("=" * 72)
    print("01 | LINEAR REGRESSION  -  Diabetes dataset")
    print("=" * 72)
    print(f"Samples N = {X.shape[0]}, features p = {X.shape[1]}")
    print(f"Features: {', '.join(names)}")
    print("Target: disease progression one year after baseline")
    print(f"Target mean = {y.mean():.2f}, standard deviation = {y.std():.2f}")
    print()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=args.seed)
    print(f"Train / test split: {X_tr.shape[0]} / {X_te.shape[0]}")
    print()

    # ---------------- (a) closed form -------------------------------------
    print("-" * 72)
    print("(a) CLOSED FORM:  theta = (X^T X)^{-1} X^T y")
    print("-" * 72)
    theta_cf = fit_normal_equation(X_tr, y_tr)
    print(f"Intercept b = {theta_cf[0]:.4f}")
    for name, w in zip(names, theta_cf[1:]):
        print(f"   w[{name:>4s}] = {w:10.4f}")
    g = gradient(add_bias_column(X_tr), y_tr, theta_cf)
    print(f"\n||dL/dtheta|| at this solution = {np.linalg.norm(g):.3e}")
    print("(essentially zero, so we really are at the minimum)")
    print()

    # ---------------- (b) gradient descent --------------------------------
    print("-" * 72)
    print(f"(b) GRADIENT DESCENT  (eta = {args.eta}, iterations = {args.iters})")
    print("-" * 72)
    theta_gd, history = fit_gradient_descent(X_tr, y_tr, args.eta, args.iters)
    print(f"{'iteration':>12}{'training MSE':>16}")
    for k in [0, 10, 100, 1000, 10000, args.iters]:
        print(f"{k:>12}{history[k]:>16.4f}")
    print()
    print(f"Training MSE, gradient descent : {history[-1]:.6f}")
    print(f"Training MSE, closed form      : {mse(y_tr, predict(X_tr, theta_cf)):.6f}")
    print(f"Difference between the two theta vectors: "
          f"{np.linalg.norm(theta_gd - theta_cf):.6f}")
    print()
    print("The two methods agree. Gradient descent needs many iterations here")
    print("because the features have very different scales, which makes the")
    print("loss surface a long narrow valley.")
    print()

    # ---------------- evaluation ------------------------------------------
    y_hat_tr = predict(X_tr, theta_cf)
    y_hat_te = predict(X_te, theta_cf)
    metrics = {
        "train_mse": mse(y_tr, y_hat_tr), "train_mae": mae(y_tr, y_hat_tr),
        "train_r2": r2_score(y_tr, y_hat_tr),
        "test_mse": mse(y_te, y_hat_te), "test_mae": mae(y_te, y_hat_te),
        "test_r2": r2_score(y_te, y_hat_te),
        "test_rmse": float(np.sqrt(mse(y_te, y_hat_te))),
        "gd_final_train_mse": float(history[-1]),
        "theta_difference": float(np.linalg.norm(theta_gd - theta_cf)),
        "eta": args.eta, "iterations": args.iters,
    }
    print("-" * 72)
    print("(c) PERFORMANCE")
    print("-" * 72)
    print(f"{'':<8}{'MSE':>12}{'MAE':>12}{'R^2':>10}")
    print(f"{'train':<8}{metrics['train_mse']:>12.3f}"
          f"{metrics['train_mae']:>12.3f}{metrics['train_r2']:>10.4f}")
    print(f"{'test':<8}{metrics['test_mse']:>12.3f}"
          f"{metrics['test_mae']:>12.3f}{metrics['test_r2']:>10.4f}")
    print(f"test RMSE = {metrics['test_rmse']:.3f}")
    print()

    # ---------------- plots ------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history, lw=1.6)
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("training MSE")
    ax[0].set_title(rf"Gradient descent, $\eta = {args.eta}$")
    ax[0].set_yscale("log")
    ax[0].grid(alpha=0.3)
    ax[1].scatter(y_te, y_hat_te, s=18, alpha=0.75, edgecolor="none")
    lims = [min(y_te.min(), y_hat_te.min()), max(y_te.max(), y_hat_te.max())]
    ax[1].plot(lims, lims, "k--", lw=1)
    ax[1].set_xlabel("true $y$")
    ax[1].set_ylabel(r"predicted $\hat{y}$")
    ax[1].set_title(f"Test set, $R^2$ = {metrics['test_r2']:.3f}")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    f1 = os.path.join(FIGDIR, "01_diabetes_fit.pdf")
    fig.savefig(f1)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    residuals = y_te - y_hat_te
    ax[0].scatter(y_hat_te, residuals, s=18, alpha=0.75, edgecolor="none")
    ax[0].axhline(0, color="k", lw=1, ls="--")
    ax[0].set_xlabel(r"predicted $\hat{y}$")
    ax[0].set_ylabel(r"residual $y - \hat{y}$")
    ax[0].set_title("Residuals vs fitted values")
    ax[0].grid(alpha=0.3)
    order = np.argsort(theta_cf[1:])
    ax[1].barh(np.array(names)[order], theta_cf[1:][order])
    ax[1].set_xlabel(r"weight $w_j$")
    ax[1].set_title("Fitted coefficients")
    ax[1].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    f2 = os.path.join(FIGDIR, "01_diabetes_diagnostics.pdf")
    fig.savefig(f2)
    plt.close(fig)

    with open(os.path.join(RESDIR, "01_diabetes.json"), "w") as fh:
        json.dump({"metrics": metrics,
                   "coefficients": {n: float(w) for n, w in zip(names, theta_cf[1:])},
                   "intercept": float(theta_cf[0])}, fh, indent=2)
    print(f"Figures saved: {f1}")
    print(f"               {f2}")
    print(f"Results saved: {os.path.join(RESDIR, '01_diabetes.json')}")


if __name__ == "__main__":
    main()
