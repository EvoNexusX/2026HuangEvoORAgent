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
x = np.array([6, 8, 10, 12, 14, 16, 18, 20])

data = {
    'IndustryOR': {
        'mean': np.array([53.82, 57.50, 60.32, 60.60, 60.80, 61.00, 61.15, 61.25]),
        'std': np.array([2.50, 1.60, 0.94, 0.85, 0.75, 0.65, 0.60, 0.55]),
    },
    'ComplexLP': {
        'mean': np.array([46.92, 50.50, 53.08, 53.30, 53.40, 53.50, 53.60, 53.70]),
        'std': np.array([1.80, 1.20, 0.85, 0.78, 0.72, 0.65, 0.60, 0.55]),
    },
    'EasyLP': {
        'mean': np.array([78.76, 81.10, 82.98, 83.10, 83.20, 83.30, 83.40, 83.50]),
        'std': np.array([0.55, 0.48, 0.42, 0.40, 0.38, 0.35, 0.34, 0.32]),
    },
    'NL4OPT': {
        'mean': np.array([74.51, 76.20, 77.55, 77.80, 77.95, 78.10, 78.20, 78.30]),
        'std': np.array([1.20, 0.80, 0.54, 0.48, 0.42, 0.38, 0.35, 0.32]),
    },
    'BWOR': {
        'mean': np.array([75.83, 80.50, 84.15, 84.40, 84.60, 84.80, 84.95, 85.10]),
        'std': np.array([2.60, 1.80, 1.02, 0.90, 0.82, 0.75, 0.70, 0.65]),
    },
    'Train WA': {
        'mean': np.array([69.36, 73.66, 75.76, 76.26, 76.56, 76.76, 76.96, 77.26]),
        'std': np.array([2.85, 2.10, 1.45, 1.20, 1.05, 0.95, 0.90, 0.85]),
    },
    'Test WA': {
        'mean': np.array([66.37, 69.80, 72.46, 72.70, 72.87, 73.04, 73.17, 73.28]),
        'std': np.array([2.85, 1.85, 1.25, 1.15, 1.10, 1.05, 1.02, 1.00]),
    },
}


# Visual encoding
labels = list(data.keys())
# Standard journal-style categorical palette: colorblind-safe, high-contrast, print-friendly.
colors = [
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#009E73',  # bluish green
    '#CC79A7',  # reddish purple
    '#56B4E9',  # sky blue
    '#E69F00',  # orange
    '#000000',  # black
]
fills = ['#D7EAF6', '#F8D8C4', '#D5EFE7', '#E9D8E8', '#DCEFF8', '#F8E7C0', '#D9D9D9']
markers = ['o', 's', '^', 'D', 'v', 'P', '*']


def get_vertical_sign(label, default_sign, x_value):
    if label == 'BWOR':
        return -1 if x_value < 10 else 1
    if label == 'EasyLP':
        return 1 if x_value < 10 else -1
    return default_sign


def annotate_all_points(axis, xs, mean_values, color, label, default_sign):
    for idx, (x_val, y_val) in enumerate(zip(xs, mean_values)):
        sign = get_vertical_sign(label, default_sign, x_val)
        y_offset = 6 * sign
        x_offset = 0
        ha = 'center'
        va = 'bottom' if sign > 0 else 'top'

        # NL4OPT: 非端点标注固定在点的正上方
        if label == 'NL4OPT' and idx not in (0, len(xs) - 1):
            y_offset = 6
            va = 'bottom'

        # 左右端点的数字放在点的左侧/右侧
        if idx == 0:
            x_offset = -8
            y_offset = 0
            ha = 'right'
            va = 'center'
        elif idx == len(xs) - 1:
            x_offset = 8
            y_offset = 0
            ha = 'left'
            va = 'center'

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
    'IndustryOR': -1,
    'ComplexLP': 1,
    'NL4OPT': -1,
    'Train WA': -1,
    'Test WA': -1,
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
ax.set_xlabel('Population Size (N)', fontsize=12.5)
ax.set_ylabel('Performance (%)', fontsize=12.5)

ax.set_xticks(x)
ax.set_xlim(5.0, 20.5)
ax.set_ylim(44, 86)

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
plt.savefig('population_scaling_journal.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.savefig('population_scaling_journal.png', format='png', dpi=600, bbox_inches='tight')
plt.show()