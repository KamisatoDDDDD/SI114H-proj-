import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, inv

# ========================= 1. 目标分布定义 =========================
def log_concave_potential(theta, lam):
    """负对数能量 Φ_R(θ)"""
    θ1, θ2 = theta
    return ((np.sqrt(lam)*θ1 - θ2)**2 + θ2**4) / 20

def grad_log_posterior(theta, lam):
    """∇ log ρ_post = -∇Φ_R"""
    θ1, θ2 = theta
    g1 = -(lam*θ1 - np.sqrt(lam)*θ2) / 10
    g2 = -(-np.sqrt(lam)*θ1 + θ2)/10 - θ2**3/5
    return np.array([g1, g2])

def hessian_log_posterior(theta, lam):
    """∇∇ log ρ_post = -∇∇Φ_R"""
    θ1, θ2 = theta
    H11 = -lam/10
    H12 = np.sqrt(lam)/10
    H22 = -1/10 - 3*θ2**2/5
    return np.array([[H11, H12], [H12, H22]])

# ========================= 2. Unscented Transform（完全复现 Julia 参数）=========================
def unscented_transform_julia(m, C, func):
    """
    严格复现 NGD.jl 中 UnscentedTransform 的参数：
    κ = 0.0, β = 2.0, α = min(sqrt(4/(n+κ)), 1.0)
    """
    n = len(m)
    κ = 0.0
    β = 2.0
    α = min(np.sqrt(4/(n + κ)), 1.0)   # 对于 n=2 → α=1.0
    λ = α**2 * (n + κ) - n             # α=1, κ=0 → λ=0
    # 权重
    Wm = np.full(2*n+1, 1/(2*(n+λ)))
    Wc = np.full(2*n+1, 1/(2*(n+λ)))
    Wm[0] = λ / (n+λ)
    Wc[0] = λ / (n+λ) + 1 - α**2 + β
    # 保证 C 正定、对称
    C = (C + C.T) / 2
    C += 1e-12 * np.eye(n)
    sqrtC = sqrtm((n+λ) * C).real
    # sigma 点
    sigma_points = [m]
    for i in range(n):
        sigma_points.append(m + sqrtC[:, i])
        sigma_points.append(m - sqrtC[:, i])
    sigma_points = np.array(sigma_points)
    # 计算函数值
    vals = np.array([func(sp) for sp in sigma_points])
    # 适配不同输出维度（标量、向量、矩阵）
    w_reshaped = Wm.reshape(-1, *([1]*(vals.ndim-1)))
    expected = np.sum(w_reshaped * vals, axis=0)
    return expected

# ========================= 3. 真值计算（高精度积分）=========================
def compute_ground_truth(lam, omega, b):
    """完全复现 Julia 中的 compute_Eref"""
    y = np.linspace(-200, 200, 10**7)
    weight = np.exp(-y**4 / 20)
    Z = np.trapz(weight, y)
    E_θ2_sq = np.trapz(y**2 * weight, y) / Z
    E_θ1_sq = (E_θ2_sq + 10) / lam
    E_θ1_θ2 = E_θ2_sq / np.sqrt(lam)
    m_true = np.array([0.0, 0.0])
    C_true = np.array([[E_θ1_sq, E_θ1_θ2], [E_θ1_θ2, E_θ2_sq]])
    cos_ref = np.zeros(len(b))
    for i, (w, bb) in enumerate(zip(omega, b)):
        factor = np.exp(-5 * w[0]**2 / lam)
        integrand = lambda t: np.cos((w[0]/np.sqrt(lam) + w[1]) * t + bb) * np.exp(-t**4/20)
        cos_ref[i] = factor * np.trapz(integrand(y), y) / Z
    return m_true, C_true, cos_ref

# ========================= 4. 三种高斯近似方法的 ODE 求解 =========================
def run_gaussian_approximation(lam, m0, C0, dt, n_steps, method):
    """
    严格复现 NGD.jl 中 second‑order 情况下的离散化：
    - Gradient_descent (Euclidean):  显式欧拉
    - Wasserstein:                   M = I - Δt*(E[∇²Φ] - inv(C_old)), C_new = M C_old M
    - Fisher‑Rao:                    C_new = inv( inv(C_old) + Δt*(E[∇²Φ] - inv(C_old)) )
    其中 ∇²Φ = -∇∇logρ, 因此 E[∇²Φ] = -EH (EH = E[∇∇logρ]).
    """
    n = len(m0)
    m = m0.copy()
    C = C0.copy()
    m_hist = [m.copy()]
    C_hist = [C.copy()]

    for _ in range(n_steps):
        # 计算期望 (UT)
        Eg = unscented_transform_julia(m, C, lambda th: grad_log_posterior(th, lam))
        EH = unscented_transform_julia(m, C, lambda th: hessian_log_posterior(th, lam))

        if method == 'Euclidean':        # Gradient_descent
            dm = Eg
            invC = inv(C)
            dC = 0.5 * invC + 0.5 * EH
            m = m + dm * dt
            C = C + dC * dt

        elif method == 'Wasserstein':
            dm = Eg
            invC = inv(C)
            # 注意: E[∇²Φ] = -EH
            M = np.eye(n) - dt * (-EH - invC)   # 化简为 I + dt*(EH + invC)
            C = M @ C @ M
            m = m + dm * dt

        elif method == 'FisherRao':
            dm = C @ Eg
            invC = inv(C)
            # 注意: E[∇²Φ] = -EH
            invC_new = invC + dt * (-EH - invC)   # = invC - dt*EH - dt*invC
            C = inv(invC_new)
            m = m + dm * dt

        else:
            raise ValueError('Unknown method')

        # 强制对称并加小正则
        C = (C + C.T) / 2
        C += 1e-12 * np.eye(n)
        m_hist.append(m.copy())
        C_hist.append(C.copy())

    return np.array(m_hist), np.array(C_hist)

# ========================= 5. 主程序（完全复现 Julia 实验） =========================
def main():
    np.random.seed(42)
    lambdas = [0.01, 0.1, 1.0]

    # 严格使用 Julia 代码中的初始条件
    m0 = np.array([10.0, 10.0])
    C0 = np.diag([4.0, 4.0])

    dt = 0.005
    n_steps = 3000
    t = np.arange(0, (n_steps + 1) * dt, dt)

    # 随机投影（20组）
    n_proj = 20
    omega = np.random.randn(n_proj, 2)
    b = np.random.uniform(0, 2 * np.pi, n_proj)

    # 计算真值
    ground_truth = {}
    for lam in lambdas:
        ground_truth[lam] = compute_ground_truth(lam, omega, b)

    # 绘图
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    methods = ['Euclidean', 'Wasserstein', 'FisherRao']
    colors = {'Euclidean': 'blue', 'Wasserstein': 'green', 'FisherRao': 'red'}
    markers = {'Euclidean': '*', 'Wasserstein': 's', 'FisherRao': 'o'}

    for i, lam in enumerate(lambdas):
        m_true, C_true, cos_ref = ground_truth[lam]
        for method in methods:
            m_hist, C_hist = run_gaussian_approximation(lam, m0, C0, dt, n_steps, method)

            # 误差计算
            mean_err = np.linalg.norm(m_hist - m_true, axis=1)
            cov_err = np.array([np.linalg.norm(C - C_true, ord='fro') / np.linalg.norm(C_true, ord='fro') for C in C_hist])
            proj_err = np.zeros(len(t))
            for j, (mm, CC) in enumerate(zip(m_hist, C_hist)):
                curr_cos = np.array([np.exp(-0.5 * w @ CC @ w) * np.cos(w @ mm + bb) for w, bb in zip(omega, b)])
                proj_err[j] = np.linalg.norm(curr_cos - cos_ref) / np.sqrt(n_proj)

            markevery = n_steps // 10
            axes[0, i].semilogy(t, mean_err, color=colors[method], marker=markers[method],
                                markevery=markevery, fillstyle='none', linewidth=1.5,
                                label=method if i == 0 else "")
            axes[1, i].semilogy(t, cov_err, color=colors[method], marker=markers[method],
                                markevery=markevery, fillstyle='none', linewidth=1.5)
            axes[2, i].semilogy(t, proj_err, color=colors[method], marker=markers[method],
                                markevery=markevery, fillstyle='none', linewidth=1.5)

        # 子图样式
        axes[0, i].set_title(f'λ = {lam}')
        axes[0, i].set_ylim([1.2e-4, 2e1])
        axes[1, i].set_ylim([1.2e-4, 2e0])
        axes[2, i].set_ylim([1.2e-4, 2e0])
        for j in range(3):
            axes[j, i].grid(True, linestyle='--', alpha=0.5)
            axes[j, i].set_xlabel('Time t')

    # 整体标签和图例
    axes[0, 0].set_ylabel(r'Error of $\mathbb{E}[\theta]$')
    axes[1, 0].set_ylabel(r'Rel. error of Cov$[\theta]$')
    axes[2, 0].set_ylabel(r'Error of $\mathbb{E}[\cos(\omega^T\theta+b)]$')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=3)
    plt.tight_layout()
    plt.savefig('Logconcave_gd_converge_perfect.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()