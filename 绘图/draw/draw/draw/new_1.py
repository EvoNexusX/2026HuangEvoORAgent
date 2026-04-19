import matplotlib.pyplot as plt
import numpy as np


# Core data
N = np.array([6, 8, 10, 12, 14, 16, 18, 20])
train = np.array([76.2, 80.5, 82.6, 83.1, 83.4, 83.6, 83.8, 84.1])
test = np.array([73.1, 76.8, 79.3, 79.6, 79.8, 80.1, 80.3, 80.5])

test_std = np.array([2.86, 2.10, 1.35, 1.05, 0.92, 0.84, 0.78, 0.72])
train_std = np.array([2.24, 1.65, 1.20, 1.00, 0.90, 0.82, 0.76, 0.70])


# Publication-style figure settings
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "legend.frameon": False,
    }
)

fig, ax = plt.subplots(figsize=(6.6, 4.6))

# Colors: orange for train, blue for test
train_line = "#D55E00"
train_fill = "#F6C8A5"
test_line = "#0072B2"
test_fill = "#AED7F2"

# Standard deviation bands (light)
ax.fill_between(N, train - train_std, train + train_std, color=train_fill, alpha=0.35, linewidth=0, zorder=1)
ax.fill_between(N, test - test_std, test + test_std, color=test_fill, alpha=0.35, linewidth=0, zorder=1)

# Mean lines (dark)
ax.plot(N, train, color=train_line, marker="o", markersize=4.8, linewidth=2.1, label="Train WA", zorder=3)
ax.plot(N, test, color=test_line, marker="s", markersize=4.8, linewidth=2.1, label="Test WA", zorder=3)

# Numeric labels: train above points, test below points.
for x_val, y_val in zip(N, train):
    # Fine-tune the first train label to avoid overlap with nearby test label.
    if x_val == 6 and abs(y_val - 76.2) < 1e-9:
        ax.text(
            x_val + 0.3,
            y_val + 0.6,
            f"{y_val:.1f}",
            ha="right",
            va="bottom",
            fontsize=9,
            color=train_line,
        )
    else:
        ax.text(x_val, y_val + 0.4, f"{y_val:.1f}", ha="center", va="bottom", fontsize=9, color=train_line)

for x_val, y_val in zip(N, test):
    # Fine-tune the second test label to avoid overlap with nearby train label.
    if x_val == 8 and abs(y_val - 76.8) < 1e-9:
        ax.text(
            x_val - 0.1,
            y_val - 0.5,
            f"{y_val:.1f}",
            ha="left",
            va="top",
            fontsize=9,
            color=test_line,
        )
    else:
        ax.text(x_val, y_val - 0.5, f"{y_val:.1f}", ha="center", va="top", fontsize=9, color=test_line)

ax.set_xlabel("Population Size (N)")
ax.set_ylabel("Weighted Accuracy (WA, %)")
ax.set_xticks(N)
ax.set_xlim(5.5, 20.5)
ax.set_ylim(70, 86)

# Subtle grid commonly accepted in paper figures
ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
ax.minorticks_on()

# Clean top/right spines for journal-style clarity
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(loc="lower right")

plt.tight_layout()

plt.savefig("population_line_std.pdf", dpi=600, bbox_inches="tight")
plt.savefig("population_line_std.png", dpi=600, bbox_inches="tight")

plt.show()
