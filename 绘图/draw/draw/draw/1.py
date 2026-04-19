import matplotlib.pyplot as plt
import numpy as np

# Data
N = np.array([6, 8, 10, 12, 14, 16, 18, 20])

train = np.array([76.2, 80.5, 82.6, 83.1, 83.4, 83.6, 83.8, 84.1])
test  = np.array([73.1, 76.8, 79.3, 79.6, 79.8, 80.1, 80.3, 80.5])

delta_train = np.array([0, 4.3, 2.1, 0.5, 0.3, 0.2, 0.2, 0.3])
delta_test  = np.array([0, 3.7, 2.5, 0.3, 0.2, 0.3, 0.2, 0.2])

# ---- Layout ----
x = np.arange(len(N))
width = 0.32

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11
})

fig, ax1 = plt.subplots(figsize=(6.5, 4.5))

# ---- Colors (match your reference style) ----
color_train_bar = "#B0B0B0"   # gray
color_test_bar  = "#4C72B0"   # blue
color_train_line = "#7F7F7F"  # dark gray
color_test_line  = "#1F77B4"  # strong blue

# ---- Bar: WA ----
bars1 = ax1.bar(x - width/2, train, width,
                color=color_train_bar,
                hatch='////',
                edgecolor='black',
                linewidth=0.6,
                label='Train WA')

bars2 = ax1.bar(x + width/2, test, width,
                color=color_test_bar,
                hatch='....',
                edgecolor='black',
                linewidth=0.6,
                label='Test WA')

ax1.set_ylabel("Weighted Accuracy (WA, %)")
ax1.set_xlabel("Population Size (N)")
ax1.set_xticks(x)
ax1.set_xticklabels(N)
ax1.set_ylim(70, 86)

ax1.grid(axis='y', linestyle='--', alpha=0.5)

# ---- Line: Δ ----
ax2 = ax1.twinx()

ax2.plot(x, delta_train,
         marker='o',
         linestyle='--',
         linewidth=1.8,
         color=color_train_line,
         label='Δ Train')

ax2.plot(x, delta_test,
         marker='s',
         linestyle='-',
         linewidth=2.0,
         color=color_test_line,
         label='Δ Test')

ax2.set_ylabel("Δ (%)")
ax2.set_ylim(0, 5)

# ---- Highlight Elbow Point ----
elbow_idx = 2  # N=10
ax1.scatter(x[elbow_idx] + width/2, test[elbow_idx],
            color='red', zorder=5)

ax1.annotate(
    "Elbow (N=10)",
    xy=(x[elbow_idx] + width/2, test[elbow_idx]),
    xytext=(3.2, 77),
    arrowprops=dict(arrowstyle="->", lw=1.2),
    fontsize=10
)

# ---- Legend (merge) ----
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='lower right', frameon=True)

plt.tight_layout()

# ---- Save ----
plt.savefig("population_bar_line.pdf", dpi=600, bbox_inches='tight')
plt.savefig("population_bar_line.png", dpi=600, bbox_inches='tight')

plt.show()