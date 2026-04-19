import matplotlib.pyplot as plt
import numpy as np

# ---------------- Data ----------------
N = np.array([6, 8, 10, 12, 14, 16, 18, 20])
x = np.arange(len(N))

train = np.array([76.2, 80.5, 82.6, 83.1, 83.4, 83.6, 83.8, 84.1])
test  = np.array([73.1, 76.8, 79.3, 79.6, 79.8, 80.1, 80.3, 80.5])

# forward Δ（整体前移）
delta_train = np.array([4.3, 2.1, 0.5, 0.3, 0.2, 0.2, 0.3])
delta_test  = np.array([3.7, 2.5, 0.3, 0.2, 0.3, 0.2, 0.2])

# ---------------- Style ----------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11
})

fig, ax = plt.subplots(figsize=(7.4, 4.6))

# ---------------- Colors ----------------
color_train_fill = "#EAF2FF"
color_test_fill  = "#E8FBF5"

color_train_edge = "#4C72B0"
color_test_edge  = "#2A9D8F"

color_train_delta = "#1F4E79"
color_test_delta  = "#1B7F5C"

# ---------------- Bars ----------------
width = 0.34

bars1 = ax.bar(x - width/2, train, width,
               color=color_train_fill,
               edgecolor=color_train_edge,
               linewidth=1.0,
               label='Train WA')

bars2 = ax.bar(x + width/2, test, width,
               color=color_test_fill,
               edgecolor=color_test_edge,
               linewidth=1.0,
               label='Test WA')

# ---------------- Axis ----------------
ax.set_ylabel("Weighted Accuracy (WA, %)")
ax.set_xlabel("Population Size (N)")
ax.set_xticks(x)
ax.set_xticklabels(N)
ax.set_ylim(72, 85)

ax.grid(axis='y', linestyle='--', alpha=0.4)

# ---------------- WA 数值 ----------------
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.15,
            f"{bar.get_height():.1f}",
            ha='center', fontsize=9)

for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.15,
            f"{bar.get_height():.1f}",
            ha='center', fontsize=9)

# ---------------- Δ 标注（最终版） ----------------
for i in range(len(delta_train)):  # 到 N=18

    # ---- Train Δ（左侧）----
    xt = x[i] - width/2
    yt = train[i]

    ax.text(xt - 0.08, yt + 0.7,
            f"+{delta_train[i]:.1f}",
            ha='right',
            fontsize=9,
            color=color_train_delta)

    ax.text(xt - 0.2, yt + 0.32,
            "↑",
            ha='right',
            fontsize=9,
            color=color_train_delta)

    # ---- Test Δ（右侧）----
    xt2 = x[i] + width/2
    yt2 = test[i]

    ax.text(xt2 + 0.08, yt2 + 0.7,
            f"+{delta_test[i]:.1f}",
            ha='left',
            fontsize=9,
            color=color_test_delta)

    ax.text(xt2 + 0.2, yt2 + 0.32,
            "↑",
            ha='left',
            fontsize=9,
            color=color_test_delta)

# ---------------- Legend ----------------
ax.legend(loc='lower right', frameon=True)

# ---------------- Title ----------------
plt.title("Impact of Population Size on Weighted Accuracy", pad=10)

plt.tight_layout()

# ---------------- Save ----------------
plt.savefig("population_final_perfect.pdf", dpi=600, bbox_inches='tight')
plt.savefig("population_final_perfect.png", dpi=600, bbox_inches='tight')

plt.show()