# plot_cca_curves_all_days_stdshade_coloralpha.py
import os, numpy as np
import matplotlib.pyplot as plt

# ===== 配置 =====
NPZ   = "data/Fig4e.npz"
OUT   = "figs/Fig4e.pdf"
TOPK  = 36   # 前K个canonical modes

# ===== 自定义颜色（线 + 阴影）=====
# RGBA 写法：最后一个数字是透明度(0~1)
C_AL  = (218/255, 53/255, 126/255, 0.7)   # 深粉色折线
C_UN  = (239/255, 134/255, 134/255, 1.0)    # 橙红折线
S_AL  = (218/255, 53/255, 126/255, 0.20)  # 对齐阴影：浅粉+透明
S_UN  = (239/255, 134/255, 134/255, 0.20)   # 未对齐阴影：浅橙+透明

# ===== 读取 =====
data = np.load(NPZ)
R_un_mat = np.asarray(data["r_unaligned_mat"])   # (num_days, K_MAX)
R_al_mat = np.asarray(data["r_aligned_mat"])
K_MAX     = int(np.asarray(data["K_MAX"]))
day_labels= np.asarray(data["day_labels"])
num_days  = R_un_mat.shape[0]

K = min(TOPK, K_MAX)
modes = np.arange(1, K+1)

# ===== 均值 ± 标准差 =====
mean_un = np.nanmean(R_un_mat[:, :K], axis=0)
std_un  = np.nanstd (R_un_mat[:, :K], axis=0)
mean_al = np.nanmean(R_al_mat[:, :K], axis=0)
std_al  = np.nanstd (R_al_mat[:, :K], axis=0)

lower_un = np.clip(mean_un - std_un, 0.0, 1.0)
upper_un = np.clip(mean_un + std_un, 0.0, 1.0)
lower_al = np.clip(mean_al - std_al, 0.0, 1.0)
upper_al = np.clip(mean_al + std_al, 0.0, 1.0)

# ===== 画图 =====
plt.figure(figsize=(6.8, 3.8), dpi=160)

# Unaligned 曲线 + 阴影
plt.plot(modes, mean_un, "--o", lw=2, ms=4, color=C_UN, label=f"Unaligned (n={num_days})")
plt.fill_between(modes, lower_un, upper_un, color=S_UN)

# Aligned 曲线 + 阴影
plt.plot(modes, mean_al, "-o", lw=2, ms=4, color=C_AL, label="CCA (Aligned)")
plt.fill_between(modes, lower_al, upper_al, color=S_AL)

plt.xlabel("Neural mode")
plt.ylabel("Correlation")
plt.ylim(0, 1.0)
plt.xlim(1, K)
plt.legend(frameon=False, loc="best")
plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")

