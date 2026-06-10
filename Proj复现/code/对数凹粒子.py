import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular

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

# ----------------------------- 模拟（单次） -----------------------------
def simulate(lam, preconditioner):
    theta = np.random.multivariate_normal(m0, C0, N_ens)
    m_oo, C_oo, cos_ref = compute_ground_truth(lam, omega, b)

    mean_err = np.zeros(n_steps+1)
    cov_err  = np.zeros(n_steps+1)
    proj_err = np.zeros(n_steps+1)

    for step in range(n_steps+1):
        if step > 0:
            # 计算梯度
            grad = np.array([grad_log_posterior(th, lam) for th in theta])
            if preconditioner:
                C_emp = np.cov(theta, rowvar=False) + 1e-8 * np.eye(2)
                L = cholesky(C_emp)
                drift = grad @ C_emp
                noise = np.sqrt(2*dt) * (np.random.randn(N_ens, 2) @ L.T)
            else:
                drift = grad
                noise = np.sqrt(2*dt) * np.random.randn(N_ens, 2)
            theta = theta + dt * drift + noise

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
for i, lam in enumerate(lambdas):
    mean_std, cov_std, proj_std = simulate(lam, preconditioner=False)
    mean_aff, cov_aff, proj_aff = simulate(lam, preconditioner=True)
    markevery = n_steps // 10

    axes[0, i].semilogy(ts, mean_std, 'b--s', markevery=markevery, fillstyle='none',
                        label='Wasserstein GF' if i==0 else "")
    axes[0, i].semilogy(ts, mean_aff, 'r-s', markevery=markevery, fillstyle='none',
                        label='Affine invariant Wasserstein GF' if i==0 else "")
    axes[1, i].semilogy(ts, cov_std,  'b--s', markevery=markevery, fillstyle='none')
    axes[1, i].semilogy(ts, cov_aff,  'r-s', markevery=markevery, fillstyle='none')
    axes[2, i].semilogy(ts, proj_std, 'b--s', markevery=markevery, fillstyle='none')
    axes[2, i].semilogy(ts, proj_aff, 'r-s', markevery=markevery, fillstyle='none')

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
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=2)
plt.tight_layout()
plt.savefig('Logconcave_particle_converge_paper2.png', dpi=300)
plt.show()