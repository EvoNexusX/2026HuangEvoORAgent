import matplotlib.pyplot as plt
# ======================
# Data
# ======================
generations = list(range(9))

data = {
    "01": [(71.5,None),(71.8,"Cross"),(73.5,"Cross"),(77.8,"Cross"),(79.2,"Cross"),
           (80.1,"Cross"),(81.3,"Cross"),(81.9,"Cross"),(82.6,"Cross")],

    "02": [(69.2,None),(71.5,"Elite"),(71.8,"Elite"),(74.1,"Cross"),(77.8,"Elite"),
           (79.2,"Elite"),(80.1,"Elite"),(81.3,"Elite"),(80.5,"Cross")],

    "03": [(65.1,None),(69.2,"Elite"),(71.5,"Elite"),(73.5,"Elite"),(75.1,"Cross"),
           (77.8,"Elite"),(79.2,"Elite"),(80.1,"Elite"),(81.9,"Elite")],

    "04": [(60.5,None),(68.5,"Cross"),(70.2,"Mut-L"),(71.8,"Elite"),(74.1,"Elite"),
           (76.5,"Cross"),(78.2,"Cross"),(79.8,"Cross"),(81.3,"Elite")],

    "05": [(56.4,None),(65.2,"Cross"),(69.8,"Cross"),(70.5,"Mut-L"),(72.5,"Mut-L"),
           (74.8,"Mut-L"),(76.1,"Mut-L"),(77.5,"Mut-L"),(78.4,"Mut-L")],

    "06": [(52.2,None),(61.8,"Mut-L"),(66.2,"Cross"),(68.5,"Cross"),(69.5,"Cross"),
           (71.1,"Cross"),(72.4,"Cross"),(74.2,"Cross"),(75.2,"Cross")],

    "07": [(49.3,None),(56.6,"Mut-S"),(62.6,"Mut-S"),(64.2,"Mut-S"),(66.2,"Mut-S"),
           (70.2,"Mut-S"),(70.5,"Mut-S"),(71.2,"Mut-S"),(73.1,"Mut-S")],

    "08": [(46.1,None),(53.1,"Mut-L"),(57.1,"Mut-L"),(60.1,"Cross"),(62.1,"Cross"),
           (65.5,"Mut-L"),(68.4,"Cross"),(69.5,"Cross"),(71.8,"Cross")],

    "09": [(43.8,None),(51.3,"Cross"),(54.2,"Mut-S"),(58.1,"Mut-L"),(60.8,"Mut-L"),
           (63.4,"Cross"),(64.8,"Mut-L"),(66.8,"Mut-L"),(68.2,"Mut-L")],

    "10": [(39.4,None),(50.2,"Mut-S"),(42.1,"Cross"),(45.2,"Mut-S"),(59.4,"Mut-S"),
           (60.2,"Mut-S"),(61.2,"Mut-S"),(60.5,"Mut-S"),(65.3,"Mut-S")]
}

value = []
# ======================
# Piecewise scaling
# ======================
def transform_y(y):
       if y < 60:
              return y * 0.5
       elif y <= 70:
              return 30 + (y - 60) * 0.8
       else:
              return 38 + (y - 70) * 1.2

# ======================
# Colors
# ======================
colors = {
          "Direct Init": "#000000",
          "KB Guided Init": "#CC79A7",
          "Elite": "#0072B2",
       "Cross": "#E69F00",
       "Mut-L": "#009E73",
       "Mut-S": "#D55E00"
}

DIRECT_INIT = "Direct Init"
KB_GUIDED_INIT = "KB Guided Init"

kb_guided_init_ids = {"01", "02", "04", "05", "07"}

# ======================
# Plot
# ======================
plt.figure(figsize=(10,10))

for ind, traj in data.items():
    for g in range(len(traj)):
        score, op = traj[g]
        y = transform_y(score)

        # ---------- G0 ----------
        if g == 0:
                     init_color = colors[KB_GUIDED_INIT] if ind in kb_guided_init_ids else colors[DIRECT_INIT]
                     plt.scatter(g, y, s=85, color=init_color,
                                 edgecolor="black", linewidth=0.4,
                                 zorder=5)

        else:
            prev_score = traj[g][0]
            prev_y = transform_y(prev_score)

            if op == "Elite":
                # 点
                plt.scatter(g, y, s=90,
                            color=colors["Elite"],
                            edgecolor="black",
                            linewidth=0.4,
                            zorder=5)

                            # 从上一代指向当前代（水平箭头）
                plt.annotate("",
                                                  xy=(g, prev_y),       # 目标（当前）
                                                  xytext=(g-1, prev_y), # 起点（左）
                             arrowprops=dict(
                                 arrowstyle="->",
                                                         color="red",
                                 lw=1.8,
                                 alpha=0.9
                             ),
                             zorder=3)

            else:
                plt.scatter(g, y, s=75,
                            color=colors.get(op, "#999999"),
                                                 edgecolor="black",
                                                 linewidth=0.4,
                            alpha=0.9,
                            zorder=4)

        # ---------- label（右侧+更小字号）----------

        if op == "Elite":
            if (score == 71.5 or score==77.8 or score==79.2 or score==81.3) and (score not in value) :  # G1的Elite点，特殊处理标签位置
                value.append(score)
                plt.text(g, y-0.6,
                 f"{score:.1f}",
                             ha='center', va='top',
                 fontsize=9)
            elif (score == 71.8 or score==80.1) and (score not in value):  # G6的Elite点，特殊处理标签位置
                value.append(score)
                plt.text(g, y+0.3,
                 f"{score:.1f}",
                             ha='center', va='bottom',
                 fontsize=9)
            else:
                plt.text(g + 0.1, y,
                 f"{score:.1f}",
                 ha='left', va='center',
                 fontsize=9)
        else:
            if score == 50.2 :  # G6的Elite点，特殊处理标签位置
                plt.text(g, y-0.6,
                 f"{score:.1f}",
                             ha='center', va='top',
                 fontsize=9)
            else:
                plt.text(g - 0.1, y,
                 f"{score:.1f}",
                 ha='right', va='center',
                 fontsize=9)

# ======================
# Axis
# ======================
plt.xlabel("Generation", fontsize=13)
plt.ylabel("Train WA (%)", fontsize=13)

plt.xticks(range(9))

yticks_real = [40, 50, 60, 70, 80]
yticks_trans = [transform_y(y) for y in yticks_real]
plt.yticks(yticks_trans, yticks_real)

plt.grid(alpha=0.2)

# ======================
# Legend
# ======================
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0],[0], marker='o', color='w', label='Direct Initialization (G0)',
           markerfacecolor=colors[DIRECT_INIT], markeredgecolor='black', markersize=7),

    Line2D([0],[0], marker='o', color='w', label='Knowledge-Base Guided Initialization (G0)',
           markerfacecolor=colors[KB_GUIDED_INIT], markeredgecolor='black', markersize=7),

    Line2D([0],[0], marker='o', color='w', label='Elitism',
           markerfacecolor=colors["Elite"], markeredgecolor='black', markersize=7),

    Line2D([0],[0], marker='o', color='w', label='Recombination',
           markerfacecolor=colors["Cross"], markeredgecolor='black', markersize=7),

    Line2D([0],[0], marker='o', color='w', label='Knowledge-Base Guided Mutation',
           markerfacecolor=colors["Mut-L"], markeredgecolor='black', markersize=7),

    Line2D([0],[0], marker='o', color='w', label='Direct Mutation',
           markerfacecolor=colors["Mut-S"], markeredgecolor='black', markersize=7),

    Line2D([0],[0], color='red', lw=2.0,
           label='Elite Inheritance (→)')
]

plt.legend(handles=legend_elements, frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig("figure3.png", format="png", dpi=300, bbox_inches="tight")
plt.savefig("figure3.pdf", format="pdf", bbox_inches="tight")
plt.show()