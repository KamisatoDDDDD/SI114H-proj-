"""
Gaussian posterior particle-flow experiment with NLAW (Normalized Local Affine-Wasserstein).
Produced curves: Wasserstein GF, Affine-invariant Wasserstein GF, NLAW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

Array = np.ndarray


@dataclass(frozen=True)
class GaussianTarget:
    m_star: Array
    C_star: Array

    @property
    def inv_C_star(self) -> Array:
        return np.linalg.inv(self.C_star)

    def grad_logrho(self, theta: Array) -> Array:
        return -(theta - self.m_star) @ self.inv_C_star.T


def empirical_mean(theta: Array) -> Array:
    return theta.mean(axis=0)


def empirical_cov(theta: Array, jitter: float = 0.0) -> Array:
    x = theta - theta.mean(axis=0, keepdims=True)
    cov = x.T @ x / max(theta.shape[0] - 1, 1)
    if jitter > 0:
        cov = cov + jitter * np.eye(theta.shape[1])
    return cov


def sqrtm_psd(C: Array, eps: float = 1e-12) -> Array:
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, eps)
    return (V * np.sqrt(w)) @ V.T


def gaussian_cos_expectation(m: Array, C: Array, omega: Array, b: Array) -> Array:
    quad = np.einsum("ij,jk,ik->i", omega, C, omega)
    return np.exp(-0.5 * quad) * np.cos(omega @ m + b)


def particle_cos_expectation(theta: Array, omega: Array, b: Array) -> Array:
    return np.cos(theta @ omega.T + b[None, :]).mean(axis=0)


# ----------------------------------------------------------------------
# NLAW specific functions
# ----------------------------------------------------------------------
def nlaws_preconditioner_and_divergence(theta: Array, m: Array, C: Array,
                                        eta: float, delta: float) -> Tuple[Array, Array]:
    """
    Compute P_j and divergence term for each particle.
    Returns:
        P: array of shape (N, d, d)  (preconditioner for each particle)
        divP: array of shape (N, d)   (divergence term for each particle)
    """
    N, d = theta.shape
    invC = np.linalg.inv(C)
    # Precompute common quantities
    r = theta - m                           # (N, d)
    q = np.einsum('ij,ij->i', r, r @ invC)  # (N,)
    # Avoid division by zero
    q_clip = np.maximum(q, delta)
    # P_j = (1-eta)*C + eta * d * (r r^T) / (q + delta)
    # We'll compute per particle using outer product
    P = np.zeros((N, d, d))
    for i in range(N):
        r_i = r[i:i+1, :]                  # (1, d)
        outer = r_i.T @ r_i                # (d, d)
        P[i] = (1-eta) * C + eta * d * outer / (q_clip[i] + delta)
    # Divergence term: ∇·P = eta * d * ((d-1)*q + (d+1)*delta) / (q+delta)^2 * r
    # For d=2: factor = 2*eta * (q + 3*delta) / (q+delta)^2
    factor = eta * d * ((d-1)*q_clip + (d+1)*delta) / ( (q_clip + delta)**2 )
    divP = factor[:, np.newaxis] * r
    return P, divP


def nlaws_step(theta: Array, target: GaussianTarget, m: Array, C: Array,
               dt: float, eta: float, delta: float, rng: np.random.Generator) -> Array:
    """
    One Euler step for NLAW (affine-invariant with local rank-1 preconditioner).
    Assumes m, C are the current empirical mean/covariance (frozen for this step).
    """
    N, d = theta.shape
    grad = target.grad_logrho(theta)                     # (N, d)
    P, divP = nlaws_preconditioner_and_divergence(theta, m, C, eta, delta)

    # Drift: P * grad + divP
    drift = np.einsum('nij,nj->ni', P, grad) + divP

    # Diffusion: sqrt(2*dt * P)
    # For each particle, compute sqrt(P) using symmetric square root (could use Cholesky + rank-1 trick,
    # but for d=2 we simply compute matrix square root per particle).
    sqrt_P = np.zeros_like(P)
    for i in range(N):
        sqrt_P[i] = sqrtm_psd(P[i])
    noise = rng.normal(size=(N, d))
    noise_corr = np.einsum('nij,nj->ni', sqrt_P, noise)

    return theta + dt * drift + math.sqrt(2.0 * dt) * noise_corr


# ----------------------------------------------------------------------
# One-step dispatcher
# ----------------------------------------------------------------------
def one_step(
    theta: Array,
    target: GaussianTarget,
    dt: float,
    method: str,
    affine_invariant: bool,
    rng: np.random.Generator,
    eta: float = None,
    delta: float = None,
) -> Array:
    if method == "Wasserstein":
        grad = target.grad_logrho(theta)
        if affine_invariant:
            cov = empirical_cov(theta, jitter=1e-12)
            sqrt_prec = sqrtm_psd(cov)
            drift = grad @ cov
            noise = rng.normal(size=theta.shape) @ sqrt_prec.T
        else:
            drift = grad
            noise = rng.normal(size=theta.shape)
        return theta + dt * drift + math.sqrt(2.0 * dt) * noise
    elif method == "NLAW":
        # For NLAW, we need current empirical mean and covariance
        m = empirical_mean(theta)
        C = empirical_cov(theta, jitter=1e-12)
        return nlaws_step(theta, target, m, C, dt, eta, delta, rng)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_errors(
    theta: Array,
    target: GaussianTarget,
    omega: Array,
    b: Array,
    cos_ref: Array,
    cos_estimator: str = "gaussian",
) -> Array:
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
    eta: float = None,
    delta: float = None,
) -> Array:
    theta = theta0.copy()
    errors = np.empty((n_steps + 1, 3), dtype=float)

    cos_ref = gaussian_cos_expectation(target.m_star, target.C_star, omega, b)
    errors[0] = compute_errors(theta, target, omega, b, cos_ref, cos_estimator)

    for k in range(1, n_steps + 1):
        theta = one_step(theta, target, dt, method, affine_invariant, rng,
                         eta=eta, delta=delta)
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
    nlaws_eta: float = 0.25,
    nlaws_delta: float = 1e-3,
) -> Tuple[Array, Dict[float, Dict[str, Array]]]:
    rng_master = np.random.default_rng(seed)
    ts = np.linspace(0.0, dt * n_steps, n_steps + 1)

    d = 2
    m0 = np.array([10.0, 10.0], dtype=float)
    C0 = np.array([[0.5, 0.0], [0.0, 2.0]], dtype=float)
    chol_C0 = np.linalg.cholesky(C0)

    omega = rng_master.normal(size=(20, d))
    b = rng_master.uniform(0.0, 2.0 * np.pi, size=20)

    # Add NLAW configuration
    configs = [
        ("Wasserstein", False, "Wasserstein GF"),
        ("Wasserstein", True, "Affine invariant Wasserstein GF"),
        ("NLAW", False, "NLAW"),   # affine_invariant flag not used for NLAW; kept for compatibility
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
            # For each method, create an independent RNG
            for (method, affine, label) in configs:
                if label == "NLAW":
                    rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
                    results[lam][label][r] = simulate_single_run(
                        theta0=theta0, target=target, omega=omega, b=b, dt=dt, n_steps=n_steps,
                        method=method, affine_invariant=affine, rng=rng, cos_estimator=cos_estimator,
                        eta=nlaws_eta, delta=nlaws_delta
                    )
                else:
                    rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
                    results[lam][label][r] = simulate_single_run(
                        theta0=theta0, target=target, omega=omega, b=b, dt=dt, n_steps=n_steps,
                        method=method, affine_invariant=affine, rng=rng, cos_estimator=cos_estimator,
                        eta=None, delta=None
                    )
                print(f"done lambda={lam}, repeat={r+1}/{n_repeat}, {label}")

    return ts, results


def plot_results(ts: Array, results: Dict[float, Dict[str, Array]], out_file: str) -> None:
    labels = ["Wasserstein GF", "Affine invariant Wasserstein GF", "NLAW"]
    styles = {
        "Wasserstein GF": dict(color="C2", linestyle=":", marker="s"),
        "Affine invariant Wasserstein GF": dict(color="C1", linestyle="-", marker="s"),
        "NLAW": dict(color="C0", linestyle="-", marker="o"),
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
            err = results[lam][label]
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
    fig.legend(handles, leg_labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=len(labels))
    fig.subplots_adjust(bottom=0.07, top=0.90, left=0.10, right=0.98, hspace=0.22, wspace=0.12)
    fig.savefig(out_file, dpi=220)
    print(f"saved figure to {out_file}")


def main() -> None:
    N_ENS = 100
    N_REPEAT = 1
    DT = 0.005
    N_STEPS = 3000

    # Hyperparameters for NLAW
    ETA = 0.25
    DELTA = 1e-3

    ts, results = run_experiment(
        lambdas=(0.01, 0.1, 1.0),
        n_ens=N_ENS,
        n_repeat=N_REPEAT,
        dt=DT,
        n_steps=N_STEPS,
        seed=42,
        cos_estimator="particle",
        nlaws_eta=ETA,
        nlaws_delta=DELTA,
    )
    plot_results(ts, results, out_file="gaussian_particle_converge_with_nlaw.png")


if __name__ == "__main__":
    main()