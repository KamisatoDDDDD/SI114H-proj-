import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

# ==================== 参数设置 ====================
lambdas = [0.01, 0.1, 1.0]
m0 = np.array([10.0, 10.0])
C0 = np.diag([0.5, 2.0])
m_star = np.array([0.0, 0.0])
T = 15.0
t = np.linspace(0, T, 300)

# 随机投影（20组，固定种子保证可重复）
np.random.seed(42)
n_proj = 20
omegas = [np.random.randn(2) for _ in range(n_proj)]
bs = [np.random.uniform(0, 2*np.pi) for _ in range(n_proj)]

# ==================== 解析解函数 ====================
def solve_euclidean(C_star, m0, C0, t):
    """欧几里得 GF (4.32) 解析解"""
    invC_star = np.linalg.inv(C_star)
    invC0 = np.linalg.inv(C0)
    m_t = np.zeros((len(t), 2))
    C_t = np.zeros((len(t), 2, 2))
    for i, ti in enumerate(t):
        # 均值：对角矩阵指数
        exp_diag = np.exp(-np.diag(invC_star) * ti)
        m_t[i] = exp_diag * m0
        # 协方差
        invCt = invC0 + (invC_star - invC0) * (1 - np.exp(-ti))
        C_t[i] = np.linalg.inv(invCt)
    return m_t, C_t

def solve_wasserstein(C_star, m0, C0, t):
    """高斯近似 Wasserstein GF (4.24) 解析解"""
    invC_star = np.linalg.inv(C_star)
    m_t = np.zeros((len(t), 2))
    C_t = np.zeros((len(t), 2, 2))
    for i, ti in enumerate(t):
        exp_diag = np.exp(-np.diag(invC_star) * ti)
        m_t[i] = exp_diag * m0
        exp2_diag = np.exp(-2 * np.diag(invC_star) * ti)
        C_t[i] = C_star + np.diag(exp2_diag) @ (C0 - C_star)
    return m_t, C_t

def solve_fisher_rao(C_star, m0, C0, t):
    """高斯近似 Fisher-Rao GF (4.18) 解析解（协方差解析，均值数值积分）"""
    invC_star = np.linalg.inv(C_star)
    invC0 = np.linalg.inv(C0)
    C_t = np.zeros((len(t), 2, 2))
    # 协方差解析
    for i, ti in enumerate(t):
        invCt = invC_star + np.exp(-ti) * (invC0 - invC_star)
        C_t[i] = np.linalg.inv(invCt)
    # 均值通过数值积分：dm/dt = -C(t) @ invC_star @ m
    # 由于所有矩阵对角，可逐分量计算
    m_t = np.zeros((len(t), 2))
    for d in range(2):
        # a(s) = C(s)[d,d] * invC_star[d,d]
        a = C_t[:, d, d] * invC_star[d, d]
        integral = cumtrapz(a, t, initial=0)
        m_t[:, d] = m0[d] * np.exp(-integral)
    return m_t, C_t

# ==================== 误差计算 ====================
def mean_l2_error(m, m_true):
    return np.linalg.norm(m - m_true)

def cov_frobenius_relative(C, C_true):
    return np.linalg.norm(C - C_true, ord='fro') / np.linalg.norm(C_true, ord='fro')

def projection_error(m, C, m_true, C_true, omegas, bs):
    """高斯分布的 E[cos(ωᵀθ+b)] 解析公式"""
    err = 0.0
    for omega, b in zip(omegas, bs):
        curr = np.exp(-0.5 * omega @ C @ omega) * np.cos(omega @ m + b)
        true = np.exp(-0.5 * omega @ C_true @ omega) * np.cos(omega @ m_true + b)
        err += abs(curr - true)
    return err / len(omegas)

# ==================== 运行所有方法 ====================
methods = {
    'Euclidean': solve_euclidean,
    'Wasserstein': solve_wasserstein,
    'Fisher-Rao': solve_fisher_rao
}

results = {}
for lam in lambdas:
    C_star = np.diag([1.0, lam])
    results[lam] = {}
    for name, solver in methods.items():
        m_t, C_t = solver(C_star, m0, C0, t)
        results[lam][name] = (t, m_t, C_t)

# ==================== 收集误差范围（统一纵坐标） ====================
all_errors = {'mean': [], 'cov': [], 'proj': []}
for lam in lambdas:
    C_star = np.diag([1.0, lam])
    for name in methods:
        t, m_t, C_t = results[lam][name]
        mean_err = [mean_l2_error(m, m_star) for m in m_t]
        cov_err = [cov_frobenius_relative(C, C_star) for C in C_t]
        proj_err = [projection_error(m_t[i], C_t[i], m_star, C_star, omegas, bs) for i in range(len(t))]
        all_errors['mean'].extend(mean_err)
        all_errors['cov'].extend(cov_err)
        all_errors['proj'].extend(proj_err)

ylims = {}
for key in ['mean', 'cov', 'proj']:
    min_val = max(min(all_errors[key]), 1e-16)
    max_val = max(all_errors[key])
    ylims[key] = (min_val, max_val * 1.2)

# ==================== 绘图（3行×3列，空心标记） ====================
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
row_titles = ['Error of E[θ] (L2)', 'Error of Cov[θ] (rel. Frobenius)', 'Error of E[cos(ωᵀθ+b)]']
col_titles = [f'λ = {lam}' for lam in lambdas]

style = {
    'Euclidean':   {'color': 'blue',  'marker': 'o', 'label': 'Euclidean'},
    'Wasserstein': {'color': 'green', 'marker': 's', 'label': 'Wasserstein'},
    'Fisher-Rao':  {'color': 'red',   'marker': '^', 'label': 'Fisher-Rao'}
}
line_styles = ['-', '--', ':']   # 均值、协方差、投影

for i, lam in enumerate(lambdas):
    C_star = np.diag([1.0, lam])
    for name, sty in style.items():
        t, m_t, C_t = results[lam][name]
        # 均值误差（实线）
        mean_err = [mean_l2_error(m, m_star) for m in m_t]
        axes[0, i].semilogy(t, mean_err, color=sty['color'], ls=line_styles[0],
                            marker=sty['marker'], markevery=30, linewidth=1.5,
                            markerfacecolor='none', markeredgecolor=sty['color'],
                            label=sty['label'] if i==0 else "")
        # 协方差误差（虚线）
        cov_err = [cov_frobenius_relative(C, C_star) for C in C_t]
        axes[1, i].semilogy(t, cov_err, color=sty['color'], ls=line_styles[1],
                            marker=sty['marker'], markevery=30, linewidth=1.5,
                            markerfacecolor='none', markeredgecolor=sty['color'])
        # 投影误差（点线）
        proj_err = [projection_error(m_t[j], C_t[j], m_star, C_star, omegas, bs) for j in range(len(t))]
        axes[2, i].semilogy(t, proj_err, color=sty['color'], ls=line_styles[2],
                            marker=sty['marker'], markevery=30, linewidth=1.5,
                            markerfacecolor='none', markeredgecolor=sty['color'])

# 统一纵坐标、图例、标签
for i, lam in enumerate(lambdas):
    axes[0, i].set_title(col_titles[i])
    for j in range(3):
        axes[j, i].grid(True, linestyle='--', alpha=0.5)
        axes[j, i].set_xlabel('Time t')
        axes[j, i].set_ylim(ylims[['mean','cov','proj'][j]])
    axes[0, i].legend(loc='upper right', fontsize=8)

for j, title in enumerate(row_titles):
    axes[j, 0].set_ylabel(title, fontsize=10, rotation=90, labelpad=20)

plt.tight_layout()
plt.savefig('fig2_correct.png', dpi=300)
plt.show()