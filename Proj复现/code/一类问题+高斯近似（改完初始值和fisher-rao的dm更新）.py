import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- 辅助函数 -----------------------------
def cos_error_estimation_particle(m, C, omega, b):
    """高斯分布的 E[cos(ωᵀθ+b)] 解析式"""
    val = np.exp(-0.5 * np.einsum('ij,ij->i', omega @ C, omega)) * np.cos(omega @ m + b)
    return val

# ----------------------------- 高斯近似方法 -----------------------------
def residual(method_type, m_oo, C_oo, m, C):
    if method_type == "gradient_descent":          # 欧几里得 GF (4.32)
        dm = -np.linalg.solve(C_oo, m - m_oo)
        dC = 0.5 * np.linalg.inv(C) - 0.5 * np.linalg.inv(C_oo)
    elif method_type == "natural_gradient_descent":  # Fisher-Rao GF (4.18) 按照用户原代码
        dm = - C @ np.linalg.solve(C_oo, m - m_oo)
        dC = C - C @ np.linalg.solve(C_oo, C)
    elif method_type == "wasserstein_gradient_descent":  # Wasserstein GF (4.24)
        dm = -np.linalg.solve(C_oo, m - m_oo)
        I = np.eye(len(m))
        dC = 2*I - np.linalg.solve(C_oo, C) - C @ np.linalg.inv(C_oo)
    else:
        raise ValueError(f"Unknown method_type: {method_type}")
    dC = (dC + dC.T) / 2
    return dm, dC

def continuous_dynamics(method_type, m_oo, C_oo, m0, C0, dt, n_steps):
    d = len(m0)
    m_hist = np.zeros((n_steps+1, d))
    C_hist = np.zeros((n_steps+1, d, d))
    m_hist[0] = m0
    C_hist[0] = C0
    for i in range(n_steps):
        dm, dC = residual(method_type, m_oo, C_oo, m_hist[i], C_hist[i])
        m_hist[i+1] = m_hist[i] + dm * dt
        C_hist[i+1] = C_hist[i] + dC * dt
        C_hist[i+1] = (C_hist[i+1] + C_hist[i+1].T) / 2
        C_hist[i+1] += 1e-12 * np.eye(d)
    return m_hist, C_hist

# ----------------------------- 主程序 -----------------------------
np.random.seed(42)
N_θ = 2
# 注意：原用户 Julia 代码中 m0=[1,1], C0=4I，与论文不同，但为了复现其代码，保留
m0 = np.array([10.0, 10.0])
C0 = np.diag([0.5, 2.0])
# 若需按论文设置，取消下一行注释
# m0, C0 = np.array([10.0,10.0]), np.diag([0.5,2.0])

ϵs = [0.01, 0.1, 1.0]
dt = 5e-3
n_steps = 3000
ts = np.linspace(0, dt*n_steps, n_steps+1)

n_proj = 20
omega = np.random.randn(n_proj, N_θ)
b = np.random.uniform(0, 2*np.pi, n_proj)

fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey='row')

for idx_ϵ, ϵ in enumerate(ϵs):
    m_oo = np.array([0.0, 0.0])
    C_oo = np.diag([1.0, 1.0/ϵ])

    # 计算三种高斯近似方法
    m_gd, C_gd = continuous_dynamics("gradient_descent", m_oo, C_oo, m0, C0, dt, n_steps)
    m_wgd, C_wgd = continuous_dynamics("wasserstein_gradient_descent", m_oo, C_oo, m0, C0, dt, n_steps)
    m_ngd, C_ngd = continuous_dynamics("natural_gradient_descent", m_oo, C_oo, m0, C0, dt, n_steps)

    cos_ref = cos_error_estimation_particle(m_oo, C_oo, omega, b)

    # 计算误差
    def compute_errors(m_hist, C_hist):
        mean_err = np.linalg.norm(m_hist - m_oo, axis=1)
        cov_err = np.array([np.linalg.norm(C - C_oo, ord='fro') / np.linalg.norm(C_oo, ord='fro') for C in C_hist])
        proj_err = np.zeros(len(ts))
        for i in range(len(ts)):
            proj_err[i] = np.linalg.norm(cos_ref - cos_error_estimation_particle(m_hist[i], C_hist[i], omega, b)) / np.sqrt(n_proj)
        return mean_err, cov_err, proj_err

    err_gd = compute_errors(m_gd, C_gd)
    err_wgd = compute_errors(m_wgd, C_wgd)
    err_ngd = compute_errors(m_ngd, C_ngd)

    markevery = n_steps // 10
    # 第一行：均值误差
    axes[0, idx_ϵ].semilogy(ts, err_gd[0], '-*', markevery=markevery, fillstyle='none', label="Gaussian approximate GF")
    axes[0, idx_ϵ].semilogy(ts, err_wgd[0], '-s', markevery=markevery, fillstyle='none', label="Gaussian approximate Wasserstein GF")
    axes[0, idx_ϵ].semilogy(ts, err_ngd[0], '-o', markevery=markevery, fillstyle='none', label="Gaussian approximate Fisher-Rao GF")
    # 第二行：协方差误差
    axes[1, idx_ϵ].semilogy(ts, err_gd[1], '-*', markevery=markevery, fillstyle='none')
    axes[1, idx_ϵ].semilogy(ts, err_wgd[1], '-s', markevery=markevery, fillstyle='none')
    axes[1, idx_ϵ].semilogy(ts, err_ngd[1], '-o', markevery=markevery, fillstyle='none')
    # 第三行：投影误差
    axes[2, idx_ϵ].semilogy(ts, err_gd[2], '-*', markevery=markevery, fillstyle='none')
    axes[2, idx_ϵ].semilogy(ts, err_wgd[2], '-s', markevery=markevery, fillstyle='none')
    axes[2, idx_ϵ].semilogy(ts, err_ngd[2], '-o', markevery=markevery, fillstyle='none')

    axes[0, idx_ϵ].set_title(f"λ = {ϵ}")
    axes[0, idx_ϵ].set_ylim([1.2e-4, 2e1])
    axes[1, idx_ϵ].set_ylim([1.2e-4, 5e0])
    axes[2, idx_ϵ].set_ylim([1.2e-4, 2e0])
    for j in range(3):
        axes[j, idx_ϵ].grid(True)

handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=3)
axes[0,0].set_ylabel(r"Error of $\mathbb{E}[\theta]$")
axes[1,0].set_ylabel(r"Rel. error of Cov$[\theta]$")
axes[2,0].set_ylabel(r"Error of $\mathbb{E}[\cos(\omega^T\theta+b)]$")
fig.tight_layout()
fig.savefig("Gaussian_gd_converge.png", dpi=300)
plt.show()