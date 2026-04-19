import matplotlib.pyplot as plt
import numpy as np

# Publication-style settings inspired by the reference figure
plt.rcParams.update(
    {
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'font.family': 'serif',
        'font.serif': ['Times New Roman'] + plt.rcParams['font.serif'],
        'mathtext.fontset': 'stix',
        'axes.linewidth': 1.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'legend.frameon': False,
        'font.size': 11,
    }
)


# Experimental data from the table
x = np.array([0, 2, 4, 6, 8, 10, 12, 15])

LABEL_INDUSTRY = 'IndustryOR'
LABEL_COMPLEX = 'ComplexLP'
LABEL_EASY = 'EasyLP'
LABEL_NL4OPT = 'NL4OPT'
LABEL_BWOR = 'BWOR'
LABEL_TRAIN = 'Train WA'
LABEL_TEST = 'Test WA'

data = {
    LABEL_INDUSTRY: {
        'mean': np.array([36.15, 44.50, 51.20, 56.40, 60.32, 60.10, 59.50, 59.00]),
        'std': np.array([3.60, 2.80, 2.10, 1.40, 0.94, 0.85, 0.75, 0.65]),
    },
    LABEL_COMPLEX: {
        'mean': np.array([45.18, 48.20, 50.30, 51.90, 53.08, 52.80, 52.20, 51.80]),
        'std': np.array([3.20, 2.20, 1.60, 1.10, 0.85, 0.75, 0.68, 0.60]),
    },
    LABEL_EASY: {
        'mean': np.array([78.02, 79.50, 80.80, 81.90, 82.98, 82.70, 82.50, 82.20]),
        'std': np.array([0.65, 0.58, 0.52, 0.48, 0.42, 0.38, 0.35, 0.30]),
    },
    LABEL_NL4OPT: {
        'mean': np.array([67.10, 70.80, 73.50, 75.80, 77.55, 77.10, 76.70, 76.40]),
        'std': np.array([2.50, 1.80, 1.20, 0.85, 0.54, 0.48, 0.45, 0.40]),
    },
    LABEL_BWOR: {
        'mean': np.array([49.73, 62.40, 71.80, 79.20, 84.15, 83.60, 83.10, 82.90]),
        'std': np.array([4.20, 3.10, 2.20, 1.50, 1.02, 0.90, 0.85, 0.80]),
    },
    LABEL_TRAIN: {
        'mean': np.array([49.60, 59.66, 66.36, 72.46, 75.76, 76.66, 77.96, 78.66]),
        'std': np.array([3.80, 3.30, 2.50, 1.85, 1.45, 1.25, 1.05, 0.85]),
    },
    LABEL_TEST: {
        'mean': np.array([51.50, 60.00, 64.71, 69.22, 72.46, 72.08, 71.59, 71.25]),
        'std': np.array([3.40, 2.80, 2.10, 1.55, 1.25, 1.15, 1.05, 0.86]),
    },
}


# Visual encoding
labels = list(data.keys())
colors = [
    '#0072B2',
    '#D55E00',
    '#009E73',
    '#CC79A7',
    '#E69F00',
    '#56B4E9',
    '#5B5B5B',
]
fills = ['#AED7F2', '#F6C8A5', '#BFE6D1', '#E9C6E6', '#F8E1A8', '#C8E7F5', '#D6D6D6']
markers = ['o', 's', '^', 'D', 'v', 'P', '*']


def get_vertical_sign(label, default_sign, x_value):
    if label == LABEL_BWOR:
        return -1 if x_value < 10 else 1
    if label == LABEL_EASY:
        return 1 if x_value < 10 else -1
    return default_sign


def apply_base_rules(label, idx, total_count, x_val, y_val, default_sign):
    sign = get_vertical_sign(label, default_sign, x_val)
    x_offset = 0
    y_offset = 6 * sign
    ha = 'center'
    va = 'bottom' if sign > 0 else 'top'

    if label == LABEL_NL4OPT and idx not in (0, total_count - 1):
        y_offset = 6
        va = 'bottom'

    if idx == 0:
        x_offset = -8
        y_offset = 0
        ha = 'right'
        va = 'center'
    elif idx == total_count - 1:
        x_offset = 8
        y_offset = 0
        ha = 'left'
        va = 'center'

    if label == LABEL_INDUSTRY and y_val >= 51.2:
        x_offset = 0
        y_offset = 6
        ha = 'center'
        va = 'bottom'

    if label == LABEL_COMPLEX and y_val >= 50.3:
        x_offset = 0
        y_offset = -6
        ha = 'center'
        va = 'top'

    return x_offset, y_offset, ha, va


def apply_manual_overrides(label, x_val):
    point_key = (label, int(round(float(x_val))))
    manual_point_rules = {
        (LABEL_TRAIN, 0): (0, -6, 'center', 'top'),
        (LABEL_EASY, 8): (0, -6, 'center', 'top'),
        (LABEL_BWOR, 8): (0, 6, 'center', 'bottom'),
        (LABEL_BWOR, 15): (0, 6, 'center', 'bottom'),
        (LABEL_NL4OPT, 6): (0, -6, 'center', 'top'),
        (LABEL_BWOR, 2): (0, 6, 'center', 'bottom'),
        (LABEL_TRAIN, 4): (-6, -4, 'right', 'bottom'),
        (LABEL_EASY, 0): (0, 6, 'center', 'bottom'),
        (LABEL_NL4OPT, 12): (0, -6, 'center', 'top'),
        (LABEL_TEST, 2): (6, +1, 'left', 'top'),
        (LABEL_TRAIN, 12): (0, 1.5, 'center', 'bottom'),
    }

    if point_key in manual_point_rules:
        return manual_point_rules[point_key]
    return None


def annotate_all_points(axis, xs, mean_values, color, label, default_sign):
    for idx, (x_val, y_val) in enumerate(zip(xs, mean_values)):
        x_offset, y_offset, ha, va = apply_base_rules(
            label, idx, len(xs), x_val, y_val, default_sign
        )

        manual = apply_manual_overrides(label, x_val)
        if manual is not None:
            x_offset, y_offset, ha, va = manual

        axis.annotate(
            f'{y_val:.1f}',
            xy=(x_val, y_val),
            xytext=(x_offset, y_offset),
            textcoords='offset points',
            ha=ha,
            va=va,
            fontsize=9,
            color=color,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.62, 'pad': 0.18},
            zorder=5,
        )


# Plot
fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=600)

default_sign_map = {
    LABEL_INDUSTRY: -1,
    LABEL_COMPLEX: 1,
    LABEL_NL4OPT: -1,
    LABEL_TRAIN: -1,
    LABEL_TEST: -1,
}

for i, key in enumerate(labels):
    mean = data[key]['mean']
    std = data[key]['std']
    color = colors[i]
    marker = markers[i]
    fill = fills[i]

    ax.fill_between(x, mean - std, mean + std, color=fill, alpha=0.35, linewidth=0, zorder=1)
    ax.plot(
        x,
        mean,
        label=key,
        color=color,
        marker=marker,
        markersize=4.8,
        linewidth=2.1,
        markeredgecolor='black',
        markeredgewidth=0.5,
        zorder=3,
    )
    annotate_all_points(ax, x, mean, color, key, default_sign_map.get(key, 1))


# Axes, ticks, and layout
ax.set_xlabel('Generation (T)', fontsize=12.5)
ax.set_ylabel('Performance (%)', fontsize=12.5)

ax.set_xticks(x)
ax.set_xlim(-1.2, 15.5)
ax.set_ylim(30, 88)

ax.tick_params(axis='both', which='major', labelsize=10.5, length=4, width=1.0)
ax.minorticks_on()
ax.tick_params(axis='x', which='minor', bottom=True, top=False, length=2, direction='in')
ax.tick_params(axis='y', which='minor', left=True, right=False, length=2, direction='in')
ax.tick_params(axis='x', which='major', bottom=True, top=False, direction='in')
ax.tick_params(axis='y', which='major', left=True, right=False, direction='in')

ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.35, color='gray', zorder=0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend = ax.legend(
    loc='lower right',
    ncol=4,
    fontsize=9.5,
    frameon=False,
    handlelength=2.0,
    columnspacing=1.1,
    handletextpad=0.4,
)

for line in legend.get_lines():
    line.set_linewidth(2.1)


plt.tight_layout()
plt.savefig('convergence_scaling_journal.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.savefig('convergence_scaling_journal.png', format='png', dpi=600, bbox_inches='tight')
plt.show()
