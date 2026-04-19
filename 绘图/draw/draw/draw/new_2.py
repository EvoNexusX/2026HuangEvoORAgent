import matplotlib.pyplot as plt
import numpy as np


T = np.arange(16)
train = np.array([
    59.5, 62.8, 66.5, 70.8, 73.2, 76.1, 79.3, 81.5,
    82.6, 82.6, 83.5, 84.2, 84.8, 85.1, 85.5, 85.5,
])
test = np.array([
    60.6, 63.7, 66.9, 69.2, 71.4, 74.3, 76.5, 78.8,
    79.3, 79.3, 79.1, 79.2, 78.9, 78.7, 78.5, 78.1,
])

train_std = np.array([
    3.20, 2.75, 2.43, 2.20, 2.01, 1.80, 1.45, 1.31,
    1.10, 1.05, 1.00, 0.95, 0.90, 0.85, 0.80, 0.78,
])
test_std = np.array([
    3.40, 2.98, 2.78, 2.38, 2.12, 1.95, 1.58, 1.41,
    1.25, 1.20, 1.15, 1.08, 1.02, 0.96, 0.90, 0.86,
])


# Publication-style figure settings (aligned with new_1.py)
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

fig, ax = plt.subplots(figsize=(10.8, 5.0))

# Colors: orange for train, blue for test
train_line = "#D55E00"
train_fill = "#F6C8A5"
test_line = "#0072B2"
test_fill = "#AED7F2"

# Standard deviation bands (light)
ax.fill_between(T, train - train_std, train + train_std, color=train_fill, alpha=0.28, linewidth=0, zorder=1)
ax.fill_between(T, test - test_std, test + test_std, color=test_fill, alpha=0.28, linewidth=0, zorder=1)

# Mean lines (dark)
ax.plot(
    T,
    train,
    color=train_line,
    marker="o",
    markersize=4.8,
    markeredgecolor="white",
    markeredgewidth=0.7,
    linewidth=2.2,
    label="Train WA",
    zorder=3,
)
ax.plot(
    T,
    test,
    color=test_line,
    marker="s",
    markersize=4.8,
    markeredgecolor="white",
    markeredgewidth=0.7,
    linewidth=2.2,
    label="Test WA",
    zorder=3,
)

# Numeric labels with adaptive anti-overlap offsets.
for i, x_val in enumerate(T):
    train_y = train[i]
    test_y = test[i]

    if i < 3:
        train_dy = -0.75
        test_dy = 0.75
        train_va = "top"
        test_va = "bottom"
    else:
        train_dy = 0.75
        test_dy = -0.75
        train_va = "bottom"
        test_va = "top"

    ax.text(
        x_val,
        train_y + train_dy,
        f"{train_y:.1f}",
        ha="center",
        va=train_va,
        fontsize=8.4,
        color=train_line,
    )
    ax.text(
        x_val,
        test_y + test_dy,
        f"{test_y:.1f}",
        ha="center",
        va=test_va,
        fontsize=8.4,
        color=test_line,
    )

ax.set_xlabel("Generation (T)")
ax.set_ylabel("Weighted Accuracy (WA, %)")
ax.set_xticks(T)
ax.set_xlim(-0.4, 15.4)

# Auto y-limits from means and uncertainty bands to avoid clipping after data updates.
y_lower = min(np.min(train - train_std), np.min(test - test_std))
y_upper = max(np.max(train + train_std), np.max(test + test_std))
ax.set_ylim(np.floor(y_lower - 0.8), np.ceil(y_upper + 0.6))

# Subtle grid commonly accepted in paper figures
ax.set_axisbelow(True)
ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
ax.minorticks_on()

# Clean top/right spines for journal-style clarity
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(loc="lower right")

plt.tight_layout()

plt.savefig("generation_line_std.pdf", dpi=600, bbox_inches="tight")
plt.savefig("generation_line_std.png", dpi=600, bbox_inches="tight")

plt.show()
