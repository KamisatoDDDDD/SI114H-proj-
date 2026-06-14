import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular, sqrtm

# ----------------------------- 参数 -----------------------------
np.random.seed(42)
lambdas = [0.01, 0.1, 1.0]
m0 = np.array([10.0, 10.0])
C0 = np.diag([4.0, 4.0])
N_ens = 100
dt = 0.002
T = 15.0
n_steps = int(T / dt)
ts = np.linspace(0, T, n_steps + 1)

# 随机投影（20组）
n_proj = 20
omega = np.random.randn(n_proj, 2)
b = np.random.uniform(0, 2*np.pi, n_proj)

# ----------------------------- 目标分布 -----------------------------
def grad_log_posterior(theta, lam):
    θ1, θ2 = theta
    g1 = -(lam*θ1 - np.sqrt(lam)*θ2) / 10
    g2 = -(-np.sqrt(lam)*θ1 + θ2)/10 - θ2**3/5
    return np.array([g1, g2])

# 高精度参考值（附录 F）
def compute_ground_truth(lam, omega, b):
    y = np.linspace(-200, 200, 10**7)
    weight = np.exp(-y**4/20)
    Z = np.trapz(weight, y)
    E_θ2_sq = np.trapz(y**2 * weight, y) / Z
    E_θ1_sq = (E_θ2_sq + 10) / lam
    E_θ1_θ2 = E_θ2_sq / np.sqrt(lam)
    m_true = np.array([0.0, 0.0])
    C_true = np.array([[E_θ1_sq, E_θ1_θ2], [E_θ1_θ2, E_θ2_sq]])
    cos_ref = np.zeros(len(b))
    for i, (w, bb) in enumerate(zip(omega, b)):
        factor = np.exp(-5 * w[0]**2 / lam)
        integrand = lambda t: np.cos((w[0]/np.sqrt(lam) + w[1])*t + bb) * np.exp(-t**4/20)
        cos_ref[i] = factor * np.trapz(integrand(y), y) / Z
    return m_true, C_true, cos_ref

def cos_error_estimation(m, C, omega, b):
    return np.exp(-0.5 * np.diag(omega @ C @ omega.T)) * np.cos(omega @ m + b)

# ----------------------------- NLAW 辅助函数 -----------------------------
def sqrtm_psd(C, eps=1e-12):
    """Symmetric square root of a PSD matrix."""
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, eps)
    return (V * np.sqrt(w)) @ V.T

def nlaws_preconditioner_and_divergence(theta, m, C, eta, delta):
    """
    Compute P_j and divergence term for each particle.
    Returns:
        P: array of shape (N, d, d)
        divP: array of shape (N, d)
    """
    N, d = theta.shape
    invC = np.linalg.inv(C)
    r = theta - m
    q = np.einsum('ij,ij->i', r, r @ invC)
    q_clip = np.maximum(q, delta)
    # Preconditioner: (1-eta)*C + eta * d * (r r^T) / (q+delta)
    P = np.zeros((N, d, d))
    for i in range(N):
        outer = np.outer(r[i], r[i])
        P[i] = (1-eta) * C + eta * d * outer / (q_clip[i] + delta)
    # Divergence: eta * d * ((d-1)*q + (d+1)*delta) / (q+delta)^2 * r
    factor = eta * d * ((d-1)*q_clip + (d+1)*delta) / ((q_clip + delta)**2)
    divP = factor[:, np.newaxis] * r
    return P, divP

def nlaws_step(theta, lam, m, C, dt, eta, delta, rng):
    """One Euler step for NLAW."""
    N, d = theta.shape
    grad = np.array([grad_log_posterior(th, lam) for th in theta])
    P, divP = nlaws_preconditioner_and_divergence(theta, m, C, eta, delta)
    drift = np.einsum('nij,nj->ni', P, grad) + divP
    # Compute sqrt(P) for each particle
    sqrt_P = np.array([sqrtm_psd(P[i]) for i in range(N)])
    noise = rng.normal(size=(N, d))
    noise_corr = np.einsum('nij,nj->ni', sqrt_P, noise)
    return theta + dt * drift + np.sqrt(2*dt) * noise_corr

# ----------------------------- 模拟（单次，支持三种方法） -----------------------------
def simulate(lam, method, eta=None, delta=None):
    """
    method: 'standard', 'affine', 'nlaws'
    """
    theta = np.random.multivariate_normal(m0, C0, N_ens)
    m_oo, C_oo, cos_ref = compute_ground_truth(lam, omega, b)
    mean_err = np.zeros(n_steps+1)
    cov_err  = np.zeros(n_steps+1)
    proj_err = np.zeros(n_steps+1)

    for step in range(n_steps+1):
        if step > 0:
            if method == 'standard':
                grad = np.array([grad_log_posterior(th, lam) for th in theta])
                drift = grad
                noise = np.sqrt(2*dt) * np.random.randn(N_ens, 2)
                theta = theta + dt * drift + noise
            elif method == 'affine':
                grad = np.array([grad_log_posterior(th, lam) for th in theta])
                C_emp = np.cov(theta, rowvar=False) + 1e-8 * np.eye(2)
                L = cholesky(C_emp)
                drift = grad @ C_emp
                noise = np.sqrt(2*dt) * (np.random.randn(N_ens, 2) @ L.T)
                theta = theta + dt * drift + noise
            elif method == 'nlaws':
                m_cur = theta.mean(axis=0)
                C_cur = np.cov(theta, rowvar=False) + 1e-8 * np.eye(2)
                theta = nlaws_step(theta, lam, m_cur, C_cur, dt, eta, delta, np.random.default_rng())
            else:
                raise ValueError(f"Unknown method: {method}")
        # 记录误差
        m_cur = theta.mean(axis=0)
        C_cur = np.cov(theta, rowvar=False)
        mean_err[step] = np.linalg.norm(m_cur - m_oo)
        cov_err[step]  = np.linalg.norm(C_cur - C_oo, ord='fro') / np.linalg.norm(C_oo, ord='fro')
        curr_cos = cos_error_estimation(m_cur, C_cur, omega, b)
        proj_err[step] = np.linalg.norm(curr_cos - cos_ref) / np.sqrt(n_proj)
    return mean_err, cov_err, proj_err

# ----------------------------- 主循环（绘图） -----------------------------
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
# NLAW 超参数（与之前高斯实验一致）
ETA = 0.25
DELTA = 1e-3

for i, lam in enumerate(lambdas):
    # 运行三种方法
    mean_std, cov_std, proj_std = simulate(lam, method='standard')
    mean_aff, cov_aff, proj_aff = simulate(lam, method='affine')
    mean_nlaw, cov_nlaw, proj_nlaw = simulate(lam, method='nlaws', eta=ETA, delta=DELTA)

    markevery = n_steps // 10

    # 标准 Wasserstein（蓝色虚线）
    axes[0, i].semilogy(ts, mean_std, 'b--s', markevery=markevery, fillstyle='none',
                        label='Wasserstein GF' if i==0 else "")
    axes[1, i].semilogy(ts, cov_std,  'b--s', markevery=markevery, fillstyle='none')
    axes[2, i].semilogy(ts, proj_std, 'b--s', markevery=markevery, fillstyle='none')

    # 仿射不变 Wasserstein（红色实线）
    axes[0, i].semilogy(ts, mean_aff, 'r-s', markevery=markevery, fillstyle='none',
                        label='Affine invariant Wasserstein GF' if i==0 else "")
    axes[1, i].semilogy(ts, cov_aff,  'r-s', markevery=markevery, fillstyle='none')
    axes[2, i].semilogy(ts, proj_aff, 'r-s', markevery=markevery, fillstyle='none')

    # NLAW（绿色实线，圆形标记）
    axes[0, i].semilogy(ts, mean_nlaw, 'g-o', markevery=markevery, fillstyle='none',
                        label='NLAW' if i==0 else "")
    axes[1, i].semilogy(ts, cov_nlaw,  'g-o', markevery=markevery, fillstyle='none')
    axes[2, i].semilogy(ts, proj_nlaw, 'g-o', markevery=markevery, fillstyle='none')

    axes[0, i].set_title(f'λ = {lam}')
    axes[0, i].set_ylim([1e-4, 1e3])
    axes[1, i].set_ylim([1e-4, 1e3])
    axes[2, i].set_ylim([1e-4, 1e3])
    for j in range(3):
        axes[j, i].grid(True, linestyle='--', alpha=0.5)
        axes[j, i].set_xlabel('Time t')

axes[0,0].set_ylabel(r'Error of $\mathbb{E}[\theta]$')
axes[1,0].set_ylabel(r'Rel. error of Cov$[\theta]$')
axes[2,0].set_ylabel(r'Error of $\mathbb{E}[\cos(\omega^T\theta+b)]$')
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=3)
plt.tight_layout()
plt.savefig('Logconcave_particle_converge_with_nlaw.png', dpi=300)
plt.show()