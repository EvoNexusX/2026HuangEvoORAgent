import matplotlib.pyplot as plt
import numpy as np

# ---------------- Data ----------------
labels = ['0', '1', '2', '3', '4', '5', '6', '7',
          '8', '9', '10', '11', '12', '13', '14', '15']

x = np.arange(len(labels))

train = np.array([71.5, 71.8, 73.5, 77.8, 79.2, 80.1, 81.3, 81.9,
                  82.6, 82.6, 83.5, 84.2, 84.8, 85.1, 85.5, 85.5])

test  = np.array([70.6, 70.7, 71.9, 76.2, 76.4, 75.3, 77.5, 78.8,
                  79.3, 79.3, 79.1, 79.2, 78.9, 78.7, 78.5, 78.1])

# forward Δ（自动计算）
delta_train = np.diff(train)
delta_test  = np.diff(test)

# ---------------- Style ----------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11
})

# ---------------- Figure ----------------
fig, ax = plt.subplots(figsize=(12.5, 6))  # 更宽更舒展

# ---------------- Colors ----------------
color_train_fill = "#EAF2FF"
color_test_fill  = "#E8FBF5"

color_train_edge = "#4C72B0"
color_test_edge  = "#2A9D8F"

color_train_delta = "#1F4E79"
color_test_delta  = "#1B7F5C"
color_negative = "#B22222"

# ---------------- Bars ----------------
width = 0.30

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
ax.set_xlabel("Generation (T)")
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_ylim(69, 87)

ax.grid(axis='y', linestyle='--', alpha=0.4)

# 留白优化（关键）
ax.margins(x=0.02)

# ---------------- WA 数值 ----------------
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.12,
            f"{bar.get_height():.1f}",
            ha='center', fontsize=8)

for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2+0.05,
            bar.get_height() + 0.12,
            f"{bar.get_height():.1f}",
            ha='center', fontsize=8)

# ---------------- Δ 标注（最终优化版） ----------------
for i in range(len(delta_train)):

    # ---- Train Δ ----
    xt = x[i] - width/2
    yt = train[i]
    dt = delta_train[i]

    if dt > 0:
        arrow = "↑"
        color = color_train_delta
    elif dt < 0:
        arrow = "↓"
        color = color_negative
    else:
        arrow = "——"
        color = color_train_delta

    ax.text(xt + 0.02, yt + 1.1,
            f"{dt:+.1f}",
            ha='right',
            fontsize=8,
            color=color)

    if arrow:
        if arrow == "——":
            ax.text(xt, yt + 0.9,
                arrow,
                ha='right',
                fontsize=8,
                color=color)
        else:
            ax.text(xt - 0.08, yt + 0.65,
                arrow,
                ha='right',
                fontsize=8,
                color=color)
        

    # ---- Test Δ ----
    xt2 = x[i] + width/2
    yt2 = test[i]
    dt2 = delta_test[i]

    if dt2 > 0:
        arrow2 = "↑"
        color2 = color_test_delta
    elif dt2 < 0:
        arrow2 = "↓"
        color2 = color_negative
    else:
        arrow2 = "——"
        color2 = color_test_delta

    ax.text(xt2 -0.02 , yt2 + 1.1,
            f"{dt2:+.1f}",
            ha='left',
            fontsize=8,
            color=color2)

    if arrow2:
         if arrow2 == "——":
            ax.text(xt, yt + 0.9,
                arrow,
                ha='right',
                fontsize=8,
                color=color)
         else:
            ax.text(xt2 + 0.08, yt2 + 0.65,
                arrow2,
                ha='left',
                fontsize=8,
                color=color2)
       

# ---------------- Legend ----------------
ax.legend(loc='lower right', frameon=True)

# ---------------- Title ----------------
plt.title("Performance Scaling across Generations (N=10)", pad=10)

# ---------------- Save ----------------
plt.savefig("generation_scaling_final.pdf", dpi=600, bbox_inches='tight')
plt.savefig("generation_scaling_final.png", dpi=600, bbox_inches='tight')

plt.show()