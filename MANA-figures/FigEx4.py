import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# ========= 基本路径（按需修改） =========
CSV_PATH      = "data/FigEx4.csv"
SPECTRA_NPZ   = "data/FigEx4.npz"
OUT_DIR       = "figs/"
os.makedirs(OUT_DIR, exist_ok=True)

# 聚合范围
S_SHOW_K = 10   # 只展示前10个维度（若 npz 里 K < 10，会自动取最小值）

# ========== 画图风格（matplotlib）==========
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["savefig.dpi"] = 160
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.25


# ========== 图1：跨天准确率（仅 raw/polar）==========
def plot_accuracy(csv_path, out_png, day_min, day_max, max_xticks=15):

    per_day_acc_raw, per_day_acc_polar = {}, {}

    # === 读取 CSV ===
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                d = int(row["test_day"])
            except Exception:
                continue
            if not (day_min <= d <= day_max):
                continue

            try:
                ar = float(row["acc_raw"])
                ap = float(row["acc_polar"])
            except Exception:
                continue
            per_day_acc_raw.setdefault(d, []).append(ar)
            per_day_acc_polar.setdefault(d, []).append(ap)

    days = sorted(set(per_day_acc_raw.keys()) | set(per_day_acc_polar.keys()))

    # === 聚合：中位数 + IQR ===
    def _agg_map(per_day_dict):
        med, lo, hi = [], [], []
        for d in days:
            xs = np.array(per_day_dict.get(d, []), dtype=float)
            if xs.size == 0:
                med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            else:
                med.append(np.nanmedian(xs))
                lo.append(np.nanpercentile(xs, 25))
                hi.append(np.nanpercentile(xs, 75))
        return np.array(med), np.array(lo), np.array(hi)

    med_raw, lo_raw, hi_raw = _agg_map(per_day_acc_raw)
    med_pol, lo_pol, hi_pol = _agg_map(per_day_acc_polar)

    # === 处理 NaN（避免 fill_between 出错）===
    def _fill_nan(a):
        if np.all(np.isnan(a)):  # 全 NaN
            return np.zeros_like(a)
        isnan = np.isnan(a)
        if np.any(isnan):
            a[isnan] = np.interp(np.flatnonzero(isnan), np.flatnonzero(~isnan), a[~isnan])
        return a

    med_raw, lo_raw, hi_raw = map(_fill_nan, [med_raw, lo_raw, hi_raw])
    med_pol, lo_pol, hi_pol = map(_fill_nan, [med_pol, lo_pol, hi_pol])

    # === 等间距横轴 ===
    x = np.arange(1, len(days) + 1)

    fig, ax = plt.subplots(figsize=(17, 5))
    ax.set_title(f"Cross-day Accuracy ({day_min} ≤ Day ≤ {day_max})")
    ax.set_xlabel("Test Day (equal spacing)")
    ax.set_ylabel("Accuracy")
    ax.axhline(1.0 / 8.0, linestyle=":", linewidth=1.5, color="gray", label="Chance (1/8)")

    # === Raw (TR-PCA) → unaligned ===
    ax.plot(
        x, med_raw,
        color="#f2c4a5",       # 折线颜色（unaligned）
        marker="o",            # 圆点
        linewidth=2,
        label="Raw (TR-PCA)"
    )
    ax.fill_between(
        x, lo_raw, hi_raw,
        color="#ff7f0e",       # 阴影颜色（unaligned）
        alpha=0.1              # 10% 不透明度
    )

    # === Polar (CCA→R) → aligned ===
    ax.plot(
        x, med_pol,
        color="#e9c4f2",       # 折线颜色（aligned）
        marker="o",            # 圆点
        linewidth=2,
        label="Polar (CCA→R)"
    )
    ax.fill_between(
        x, lo_pol, hi_pol,
        color="#f3e7f6",       # 阴影颜色（aligned）
        alpha=1.0              # 100% 不透明度
    )

    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best")

    # === 设置刻度标签 ===
    ax.set_xlim(0.5, len(days) + 0.5)
    if len(days) <= max_xticks:
        ax.set_xticks(x)
        ax.set_xticklabels(days)
    else:
        step = int(np.ceil(len(days) / max_xticks))
        keep_idx = np.arange(0, len(days), step)
        ax.set_xticks(x[keep_idx])
        ax.set_xticklabels([days[i] for i in keep_idx])

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# ========== 图2：CCA谱(S1..S_K) + Raw未对齐线(逐维r) ==========
def plot_spectrum_with_raw_from_npz(npz_path, out_png, day_min, day_max, s_show_k=S_SHOW_K):

    data = np.load(npz_path)

    days_all = data["days"].astype(int)           # [n_days]
    S_top    = np.asarray(data["S_top"],  dtype=float)   # [n_seeds, n_days, K]
    raw_top  = np.asarray(data["raw_top"], dtype=float)  # [n_seeds, n_days, K]

    # 过滤 Day 范围
    mask = (days_all >= day_min) & (days_all <= day_max)

    days_sel = days_all[mask]
    S_sel    = S_top[:, mask, :]   # [n_seeds, n_sel_days, K]
    raw_sel  = raw_top[:, mask, :] # [n_seeds, n_sel_days, K]

    # K（总维数）可能 >= S_SHOW_K，这里取最小值
    K_total = S_sel.shape[-1]
    k = min(s_show_k, K_total)

    # reshape 到 [N, K]，N = n_seeds * n_sel_days
    S_flat   = S_sel[..., :k].reshape(-1, k)    # [N, k]
    raw_flat = raw_sel[..., :k].reshape(-1, k)  # [N, k]

    # 用 nan 中位数 / 百分位，忽略 NaN
    S_med   = np.nanmedian(S_flat, axis=0)
    S_q25   = np.nanpercentile(S_flat, 25, axis=0)
    S_q75   = np.nanpercentile(S_flat, 75, axis=0)

    raw_med = np.nanmedian(raw_flat, axis=0)
    raw_q25 = np.nanpercentile(raw_flat, 25, axis=0)
    raw_q75 = np.nanpercentile(raw_flat, 75, axis=0)

    idx = np.arange(1, k + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"CCA Spectrum vs Raw (Top {k}) — {day_min} ≤ Day ≤ {day_max}")
    ax.set_xlabel("Dimension index")
    ax.set_ylabel("Correlation")

    # CCA 对齐谱 → aligned
    ax.plot(
        idx, S_med,
        color="#e9c4f2",       # 折线颜色（aligned）
        marker="o",            # 圆点
        linewidth=2,
        label="CCA (aligned)"
    )
    ax.fill_between(
        idx, S_q25, S_q75,
        color="#f3e7f6",       # 阴影颜色（aligned）
        alpha=1.0              # 100% 不透明度
    )

    # Raw 未对齐谱 → unaligned
    ax.plot(
        idx, raw_med,
        color="#f2c4a5",       # 折线颜色（unaligned）
        linestyle="--",        # 用虚线区分
        marker="o",            # 圆点
        linewidth=1.8,
        label="Raw (unaligned)"
    )
    ax.fill_between(
        idx, raw_q25, raw_q75,
        color="#ff7f0e",       # 阴影颜色（unaligned）
        alpha=0.1              # 10% 不透明度
    )

    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# ================== 主流程 ==================
def main():

    for phase in [1, 2]:

        if phase == 1:
            DAY_MIN  = 15   # 起始天
            DAY_MAX  = 37  # 终止天
            end_1 = 'b'
            end_2 = 'd'
        elif phase == 2:
            DAY_MIN  = 50   # 起始天
            DAY_MAX  = 119  # 终止天
            end_1 = 'c'
            end_2 = 'e'


        # 图①：跨天 acc_raw vs acc_polar
        out1 = os.path.join(OUT_DIR, f"FigEx4{end_1}.pdf")
        plot_accuracy(CSV_PATH, out1, day_min=DAY_MIN, day_max=DAY_MAX)

        # 图②：CCA谱 + Raw 未对齐线（从合并 npz 中读取）
        out2 = os.path.join(OUT_DIR, f"FigEx4{end_2}.pdf")
        plot_spectrum_with_raw_from_npz(SPECTRA_NPZ, out2, day_min=DAY_MIN, day_max=DAY_MAX, s_show_k=S_SHOW_K)


if __name__ == "__main__":
    main()
