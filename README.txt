Numerical Reproduction of Gradient Flow Samplers: From Gaussian to Log-concave Targets
====================================================================================

This repository contains the code and results for the SI114H group project.
We reproduce the main numerical experiments from Chen et al. (2023) on Gaussian and log-concave posteriors, and propose a new particle method: Normalized Local Affine-Wasserstein (NLAW).

Project Structure
-----------------
.
├── code/
│   ├── 一类问题+高斯近似.py               # Gaussian posterior – Gaussian approximations
│   ├── 二类问题+高斯近似.py               # Log-concave posterior – Gaussian approximations
│   ├── 加上新方法之后的高斯后验+粒子.py   # Gaussian posterior – particle methods (Wasserstein, AI-Wasserstein, NLAW)
│   ├── 加上新方法之后的对数凹粒子.py     # Log-concave posterior – particle methods (Wasserstein, AI-Wasserstein, NLAW)
├── figures/
│   ├── Gaussian_gd_converge.png
│   ├── gaussian_particle_converge_with_nlaw.png
│   ├── Logconcave_gd_converge_perfect.png
│   ├── Logconcave_particle_converge_with_nlaw.png
├── 2302.11024v7.pdf  #The original paper
├── Numerical Reproduction of Gradient Flow Samplers: From Gaussian to Log‑concave Targets.pdf # Our report
└── README.txt

Requirements
------------
- Python 3.9+
- numpy
- scipy
- matplotlib

Install with: pip install numpy scipy matplotlib

Running the Experiments
-----------------------
Each Python file in code/ is self-contained. Run it to generate the corresponding figure.

1. Gaussian posterior – Gaussian approximations
   python code/一类问题+高斯近似.py
   Output: Gaussian_gd_converge.png

2. Gaussian posterior – particle methods (including NLAW)
   python code/加上新方法之后的高斯后验+粒子.py
   Output: gaussian_particle_converge_with_nlaw.png

3. Log-concave posterior – Gaussian approximations
   python code/二类问题+高斯近似.py
   Output: Logconcave_gd_converge_perfect.png

4. Log-concave posterior – particle methods (including NLAW)
   python code/加上新方法之后的对数凹粒子.py
   Output: Logconcave_particle_converge_with_nlaw.png

All scripts fix the random seed to 42, ensuring reproducibility.

Results
-------
The generated figures correspond to the following paper figures:
- Gaussian_gd_converge.png                 -> Paper Fig. 1(a) (Gaussian posterior, Gaussian approximations)
- gaussian_particle_converge_with_nlaw.png -> Paper Fig. 1(b) (Gaussian posterior, particle methods) + NLAW
- Logconcave_gd_converge_perfect.png       -> Paper Fig. 2(a) (Log-concave posterior, Gaussian approximations)
- Logconcave_particle_converge_with_nlaw.png -> Paper Fig. 2(b) (Log-concave posterior, particle methods) + NLAW

Citation
--------
If you use this code, please cite the original paper:
Chen et al., "Gradient flows for sampling: Mean-field models, gaussian approximations and affine invariance", arXiv:2302.11024, 2023.

Author Contributions
--------------------
- Xindi Ping: 40% of paper writing, implementation of all codes.
- Xinying Hu: 30% of paper writing, implementation of all codes.
- Jingyi Ma: 30% of paper writing, implementation of all codes.
-------
For educational purposes only. The original paper and its code are property of their respective authors.