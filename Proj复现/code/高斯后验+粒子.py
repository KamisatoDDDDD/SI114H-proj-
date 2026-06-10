"""
Gaussian posterior particle-flow experiment.

Produces a 3 x 3 figure like the paper/Julia example:
columns: lambda = 0.01, 0.1, 1.0
rows:    error of E[theta], relative covariance error, error of E[cos(omega^T theta + b)]
curves:  Wasserstein GF, affine-invariant Wasserstein GF,
         Stein GF, affine-invariant Stein GF

Only requires: numpy, matplotlib

Default settings are chosen to run in reasonable time in pure Python.
For the closer Julia/paper setting, set N_ENS = 400 and N_REPEAT = 10,
but it will be much slower.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


Array = np.ndarray


@dataclass(frozen=True)
class GaussianTarget:
    """Target density rho(theta) = N(m_star, C_star)."""

    m_star: Array
    C_star: Array

    @property
    def inv_C_star(self) -> Array:
        return np.linalg.inv(self.C_star)

    def grad_logrho(self, theta: Array) -> Array:
        """Vectorized grad log rho for theta with shape (N, d)."""
        return -(theta - self.m_star) @ self.inv_C_star.T


def empirical_mean(theta: Array) -> Array:
    return theta.mean(axis=0)


def empirical_cov(theta: Array, jitter: float = 0.0) -> Array:
    """Unbiased empirical covariance, same denominator N-1 as the Julia code."""
    x = theta - theta.mean(axis=0, keepdims=True)
    cov = x.T @ x / max(theta.shape[0] - 1, 1)
    if jitter > 0:
        cov = cov + jitter * np.eye(theta.shape[1])
    return cov


def sqrtm_psd(C: Array, eps: float = 1e-12) -> Array:
    """Symmetric square root of a PSD 2x2 covariance matrix."""
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, eps)
    return (V * np.sqrt(w)) @ V.T


def gaussian_cos_expectation(m: Array, C: Array, omega: Array, b: Array) -> Array:
    """E_{N(m,C)}[cos(omega^T theta + b)]."""
    quad = np.einsum("ij,jk,ik->i", omega, C, omega)
    return np.exp(-0.5 * quad) * np.cos(omega @ m + b)


def particle_cos_expectation(theta: Array, omega: Array, b: Array) -> Array:
    """Particle empirical average: 1/N sum_j cos(omega^T theta_j + b)."""
    return np.cos(theta @ omega.T + b[None, :]).mean(axis=0)


def median_bandwidth(theta: Array) -> float:
    """Julia compute_h(theta): h = sqrt(0.5 * median(pairwise_dist^2) / log(N+1))."""
    diff = theta[:, None, :] - theta[None, :, :]
    sqdist = np.einsum("ijk,ijk->ij", diff, diff)
    h2 = 0.5 * np.median(sqdist) / np.log(theta.shape[0] + 1.0)
    return math.sqrt(max(h2, 1e-12))


def rbf_kernel_and_grad_sum(theta: Array, precond_cov: Array | None) -> Tuple[Array, Array]:
    """
    Return K and sum_j grad_{x_j} K(x_i, x_j) for all i.

    This follows the Julia kernel! convention:
      if no preconditioner: C = h^2 I, scale = sqrt((1 + 4 log(N+1)/d)^d)
      if preconditioner:    C = d * Cov, scale = sqrt((1 + 2/d)^d)
    """
    N, d = theta.shape
    diff = theta[:, None, :] - theta[None, :, :]  # (i, j, dim), x_i - x_j

    if precond_cov is None:
        h = median_bandwidth(theta)
        C_scalar = h * h
        sqdist = np.einsum("ijk,ijk->ij", diff, diff)
        power = -0.5 * sqdist / C_scalar
        scale = math.sqrt((1.0 + 4.0 * math.log(N + 1.0) / d) ** d)
        K = scale * np.exp(np.clip(power, -745.0, 50.0))
        # dK[i,j] = K[i,j] * C^{-1}(x_i - x_j)
        dK_sum = (K[:, :, None] * diff / C_scalar).sum(axis=1)
    else:
        M = d * precond_cov + 1e-10 * np.eye(d)
        Minv = np.linalg.inv(M)
        solved = np.einsum("ijk,kl->ijl", diff, Minv.T)
        sq_maha = np.einsum("ijk,ijk->ij", diff, solved)
        power = -0.5 * sq_maha
        scale = math.sqrt((1.0 + 2.0 / d) ** d)
        K = scale * np.exp(np.clip(power, -745.0, 50.0))
        dK_sum = (K[:, :, None] * solved).sum(axis=1)

    return K, dK_sum


def one_step(
    theta: Array,
    target: GaussianTarget,
    dt: float,
    method: str,
    affine_invariant: bool,
    rng: np.random.Generator,
) -> Array:
    """One Euler step for Wasserstein or Stein particle flow."""
    N, d = theta.shape
    grad = target.grad_logrho(theta)
    cov = empirical_cov(theta, jitter=1e-12)
    Prec = cov if affine_invariant else np.eye(d)

    if method == "Wasserstein":
        drift = grad @ Prec
        if affine_invariant:
            sqrt_prec = sqrtm_psd(Prec)
            noise = rng.normal(size=theta.shape) @ sqrt_prec.T
        else:
            noise = rng.normal(size=theta.shape)
        return theta + dt * drift + math.sqrt(2.0 * dt) * noise



    raise ValueError(f"Unknown method: {method}")


def compute_errors(
    theta: Array,
    target: GaussianTarget,
    omega: Array,
    b: Array,
    cos_ref: Array,
    cos_estimator: str = "gaussian",
) -> Array:
    """[mean_error, relative_cov_error, cos_error]."""
    m = empirical_mean(theta)
    C = empirical_cov(theta)

    mean_err = np.linalg.norm(m - target.m_star)
    cov_err = np.linalg.norm(C - target.C_star, ord="fro") / np.linalg.norm(target.C_star, ord="fro")

    if cos_estimator == "gaussian":
        cos_now = gaussian_cos_expectation(m, C, omega, b)
    elif cos_estimator == "particle":
        cos_now = particle_cos_expectation(theta, omega, b)
    else:
        raise ValueError("cos_estimator must be 'gaussian' or 'particle'")
    cos_err = np.linalg.norm(cos_now - cos_ref) / math.sqrt(len(b))

    return np.array([mean_err, cov_err, cos_err], dtype=float)


def simulate_single_run(
    theta0: Array,
    target: GaussianTarget,
    omega: Array,
    b: Array,
    dt: float,
    n_steps: int,
    method: str,
    affine_invariant: bool,
    rng: np.random.Generator,
    cos_estimator: str = "gaussian",
) -> Array:
    """Return errors with shape (n_steps + 1, 3)."""
    theta = theta0.copy()
    errors = np.empty((n_steps + 1, 3), dtype=float)

    # True reference under the exact target Gaussian.
    cos_ref = gaussian_cos_expectation(target.m_star, target.C_star, omega, b)

    errors[0] = compute_errors(theta, target, omega, b, cos_ref, cos_estimator)
    for k in range(1, n_steps + 1):
        theta = one_step(theta, target, dt, method, affine_invariant, rng)
        errors[k] = compute_errors(theta, target, omega, b, cos_ref, cos_estimator)
    return errors


def run_experiment(
    lambdas=(0.01, 0.1, 1.0),
    n_ens: int = 100,
    n_repeat: int = 3,
    dt: float = 0.005,
    n_steps: int = 3000,
    seed: int = 42,
    cos_estimator: str = "gaussian",
) -> Tuple[Array, Dict[float, Dict[str, Array]]]:
    """
    Run all lambda/method/preconditioner combinations.

    Returns:
      ts: time grid, shape (n_steps+1,)
      results[lambda][label]: errors, shape (n_repeat, n_steps+1, 3)
    """
    rng_master = np.random.default_rng(seed)
    ts = np.linspace(0.0, dt * n_steps, n_steps + 1)

    d = 2
    m0 = np.array([10.0, 10.0], dtype=float)
    C0 = np.array([[0.5, 0.0], [0.0, 2.0]], dtype=float)
    chol_C0 = np.linalg.cholesky(C0)

    omega = rng_master.normal(size=(20, d))
    b = rng_master.uniform(0.0, 2.0 * np.pi, size=20)

    configs = [
        ("Wasserstein", False, "Wasserstein GF"),
        ("Wasserstein", True, "Affine invariant Wasserstein GF")
    ]

    results: Dict[float, Dict[str, Array]] = {}

    for lam in lambdas:
        target = GaussianTarget(
            m_star=np.array([0.0, 0.0], dtype=float),
            C_star=np.array([[1.0, 0.0], [0.0, 1.0 / lam]], dtype=float),
        )
        results[lam] = {label: np.empty((n_repeat, n_steps + 1, 3), dtype=float) for _, _, label in configs}

        for r in range(n_repeat):
            theta0 = m0 + rng_master.normal(size=(n_ens, d)) @ chol_C0.T
            # Use independent stochastic streams per method while keeping the same theta0.
            run_seeds = rng_master.integers(0, 2**32 - 1, size=len(configs), dtype=np.uint32)
            for (method, affine, label), s in zip(configs, run_seeds):
                rng = np.random.default_rng(int(s))
                results[lam][label][r] = simulate_single_run(
                    theta0=theta0,
                    target=target,
                    omega=omega,
                    b=b,
                    dt=dt,
                    n_steps=n_steps,
                    method=method,
                    affine_invariant=affine,
                    rng=rng,
                    cos_estimator=cos_estimator,
                )
                print(f"done lambda={lam}, repeat={r + 1}/{n_repeat}, {label}")

    return ts, results


def plot_results(ts: Array, results: Dict[float, Dict[str, Array]], out_file: str) -> None:
    labels = [
        "Wasserstein GF",
        "Affine invariant Wasserstein GF"
    ]
    styles = {
        "Wasserstein GF": dict(color="C2", linestyle=":", marker="s"),
        "Affine invariant Wasserstein GF": dict(color="C1", linestyle="-", marker="s"),
    }
    row_ylabels = [
        r"Error of $\mathbb{E}[\theta]$",
        r"Rel. error of Cov$[\theta]$",
        r"Error of $\mathbb{E}[\cos(\omega^T\theta + b)]$",
    ]
    ylims = [(1.2e-3, 2.0e1), (1.2e-3, 1.0e1), (1.2e-3, 2.0e0)]

    lambdas = list(results.keys())
    fig, axes = plt.subplots(nrows=3, ncols=len(lambdas), sharex=True, sharey="row", figsize=(12, 10))
    if len(lambdas) == 1:
        axes = axes.reshape(3, 1)

    markevery = max(len(ts) // 10, 1)
    eps = 1e-12

    for col, lam in enumerate(lambdas):
        axes[0, col].set_title(rf"$\lambda = {lam:g}$")
        for label in labels:
            err = results[lam][label]  # (repeat, time, stat)
            mean = err.mean(axis=0)
            std = err.std(axis=0)
            for row in range(3):
                ax = axes[row, col]
                st = styles[label]
                lower = np.maximum(mean[:, row] - std[:, row], eps)
                upper = np.maximum(mean[:, row] + std[:, row], eps)
                ax.semilogy(
                    ts,
                    np.maximum(mean[:, row], eps),
                    label=label,
                    markevery=markevery,
                    fillstyle="none",
                    linewidth=1.3,
                    markersize=4,
                    **st,
                )
                ax.fill_between(ts, lower, upper, alpha=0.18, color=st["color"])
                ax.grid(True, which="both", alpha=0.45)
                ax.set_ylim(*ylims[row])
                ax.set_xlabel("t")
        for row in range(3):
            axes[row, 0].set_ylabel(row_ylabels[row])

    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4)
    fig.subplots_adjust(bottom=0.07, top=0.90, left=0.10, right=0.98, hspace=0.22, wspace=0.12)
    fig.savefig(out_file, dpi=220)
    print(f"saved figure to {out_file}")


def main() -> None:
    # Fast-ish default. To match the Julia notebook more closely, use:
    # N_ENS = 400; N_REPEAT = 10; N_STEPS = 3000
    N_ENS = 100
    N_REPEAT = 1
    DT = 0.005
    N_STEPS = 3000

    ts, results = run_experiment(
        lambdas=(0.01, 0.1, 1.0),
        n_ens=N_ENS,
        n_repeat=N_REPEAT,
        dt=DT,
        n_steps=N_STEPS,
        seed=42,
        # 'gaussian' matches your Julia notebook cell; 'particle' uses the raw particle average.
        cos_estimator="particle",
    )
    plot_results(ts, results, out_file="gaussian_particle_converge.png")


if __name__ == "__main__":
    main()
