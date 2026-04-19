import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

# ========================= 节点定义 =========================
AGENT_START = "Agent Start"
LOGGING_INITIALIZED = "Logging Initialized"
CONFIGURATION_READY = "Configuration Ready"
DATASET_PATH_EXTRACTED = "Dataset Path Extracted"
QUESTION_LOADED = "Question Loaded"
QUESTION_FORMATTED = "Question Formatted"
CONTEXT_PARSED = "Context Parsed"
DOMAIN_IDENTIFIED = "Domain Identified"
PREPROCESSING_COMPLETED = "Preprocessing Completed"

SETS_DRAFTED = "Sets Drafted"
SETS_DEFINED = "Sets Defined"
SETS_RAW_OUTPUT = "Sets Raw Output"
SETS_VERIFIED = "Sets Verified"
SETS_CHECKED = "Sets Checked"
SETS_REFINED = "Sets Refined"

PARAMETERS_DEFINED = "Parameters Defined"
PARAMETERS_DRAFTED = "Parameters Drafted"
PARAMETERS_RAW_OUTPUT = "Parameters Raw Output"
PARAMETERS_BOUNDED = "Parameters Bounded"
PARAMETERS_CHECKED = "Parameters Checked"
PARAMETERS_REFINED = "Parameters Refined"

VARIABLES_DEFINED = "Variables Defined"
VARIABLES_DRAFTED = "Variables Drafted"
VARIABLES_RAW_OUTPUT = "Variables Raw Output"
VARIABLES_TYPED = "Variables Typed"
VARIABLES_BOUNDED = "Variables Bounded"
VARIABLES_VERIFIED = "Variables Verified"
VARIABLES_CHECKED = "Variables Checked"
VARIABLES_REFINED = "Variables Refined"

OBJECTIVE_DEFINED = "Objective Defined"
OBJECTIVE_DRAFTED = "Objective Drafted"
OBJECTIVE_RAW_OUTPUT = "Objective Raw Output"
OBJECTIVE_LINEARIZED = "Objective Linearized"
OBJECTIVE_CHECKED = "Objective Checked"
OBJECTIVE_REFINED = "Objective Refined"

CONSTRAINTS_DEFINED = "Constraints Defined"
CONSTRAINTS_DRAFTED = "Constraints Drafted"
CONSTRAINTS_RAW_OUTPUT = "Constraints Raw Output"
CONSTRAINTS_LOGIC = "Constraints Logic"
CONSTRAINTS_BOUNDED = "Constraints Bounded"
CONSTRAINTS_VERIFIED = "Constraints Verified"
CONSTRAINTS_CHECKED = "Constraints Checked"
CONSTRAINTS_REFINED = "Constraints Refined"

MODEL_GENERATION_INITIATED = "Model Generation Initiated"
MODEL_DRAFTED = "Model Drafted"
MODEL_RAW_OUTPUT = "Model Raw Output"
SYNTAX_CHECKED = "Syntax Checked"
MODEL_CLEANED = "Model Cleaned"
KNOWLEDGE_BASE_QUERIED = "Knowledge Base Queried"
KNOWLEDGE_BASE_INTEGRATED = "Knowledge Base Integrated"
FINAL_REVIEW = "Final Review"
STAGE_ONE_OUTPUT_GENERATED = "Stage One Output Generated"
TEXT_CONTEXT_READY = "Text Context Ready"
PHASE_ONE_COMPLETED = "Phase One Completed"


# ========================= 曲率控制 =========================
def get_rad_by_parallel_count(num_edges, edge_idx):
    presets = {
        1: [0.0],
        2: [-0.16, 0.16],
        3: [-0.26, 0.0, 0.26],
        4: [-0.34, -0.12, 0.12, 0.34],
        5: [-0.42, -0.22, 0.0, 0.22, 0.42],
        6: [-0.48, -0.30, -0.12, 0.12, 0.30, 0.48],
    }
    if num_edges in presets:
        return presets[num_edges][edge_idx]

    center = (num_edges - 1) / 2.0
    spread = 0.56
    return (edge_idx - center) * (spread / max(center, 1.0))


def get_t_by_parallel_count(num_edges, edge_idx, eid_num, backward_edge):
    if num_edges == 1:
        t_base = 0.53 if backward_edge else 0.47
    else:
        center = (num_edges - 1) / 2.0
        t_base = 0.50 + (edge_idx - center) * 0.07

    jitter = ((eid_num % 5) - 2) * 0.015
    t_base += jitter
    return max(0.18, min(0.82, t_base))


# ========================= 关键：画边（已修复） =========================
def draw_phase_edge(ax, graph, pos, u, v, eid, data):
    etype = data["etype"]
    is_macro = data["is_macro"]

    color_map = {"code": "#34a853", "prompt": "#ea4335", "tool": "#4285f4"}
    color = color_map.get(etype, "gray")

    edge_idx = list(graph[u][v]).index(eid)
    num_edges = len(graph[u][v])
    backward_edge = pos[u][0] > pos[v][0]

    rad = get_rad_by_parallel_count(num_edges, edge_idx)
    if backward_edge:
        rad = 0.28 if abs(rad) < 1e-9 else rad * 1.35

    lw = 1.35 if is_macro else 1.0
    alpha = 0.85 if is_macro else 0.6

    # ===== 用 FancyArrowPatch 画边 =====
    patch = mpatches.FancyArrowPatch(
        posA=pos[u],
        posB=pos[v],
        arrowstyle='-|>',
        connectionstyle=f'arc3,rad={rad}',
        mutation_scale=10,
        linewidth=lw,
        color=color,
        alpha=alpha,
    )
    ax.add_patch(patch)

    eid_num = int(eid[1:])
    t = get_t_by_parallel_count(num_edges, edge_idx, eid_num, backward_edge)

    # ===== 在数据坐标系中近似弧线上点，避免把文字放到画布像素坐标里 =====
    ux, uy = pos[u]
    vx, vy = pos[v]
    dx = vx - ux
    dy = vy - uy
    mid_x = (ux + vx) / 2.0
    mid_y = (uy + vy) / 2.0
    ctrl_x = mid_x - dy * rad
    ctrl_y = mid_y + dx * rad
    one_minus_t = 1.0 - t
    label_x = (one_minus_t ** 2) * ux + 2 * one_minus_t * t * ctrl_x + (t ** 2) * vx
    label_y = (one_minus_t ** 2) * uy + 2 * one_minus_t * t * ctrl_y + (t ** 2) * vy

    # ===== 法线偏移（避免压线）=====
    dx = 2 * one_minus_t * (ctrl_x - ux) + 2 * t * (vx - ctrl_x)
    dy = 2 * one_minus_t * (ctrl_y - uy) + 2 * t * (vy - ctrl_y)

    length = math.hypot(dx, dy) + 1e-6
    nx_off = -dy / length
    ny_off = dx / length

    offset_scale = 0.08
    label_x += nx_off * offset_scale
    label_y += ny_off * offset_scale

    ax.text(
        label_x,
        label_y,
        eid,
        fontsize=5.4,
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none"},
        zorder=10,
        clip_on=False,
    )


# ========================= 渲染 =========================
def render_phase_graph(phase_name, phase_edges, output_base):
    G = nx.MultiDiGraph()
    ordered_nodes = []

    def register_node(node):
        if node not in ordered_nodes:
            ordered_nodes.append(node)

    for u, v, eid, etype, is_macro in phase_edges:
        G.add_edge(u, v, key=eid, etype=etype, is_macro=is_macro)
        register_node(u)
        register_node(v)

    pos = {}
    for i, node in enumerate(ordered_nodes):
        pos[node] = (i * 1.5, math.sin(i * 0.8))

    plt.figure(figsize=(max(16, len(ordered_nodes)), 10))

    nx.draw_networkx_nodes(
        G, pos,
        node_size=900,
        node_color="#e8f0fe",
        edgecolors="#1a73e8"
    )

    ax = plt.gca()

    for node, (x, y) in pos.items():
        ax.text(x, y - 0.15, node, ha='center', fontsize=8)

    edges_sorted = sorted(G.edges(keys=True, data=True), key=lambda x: int(x[2][1:]))

    for u, v, eid, data in edges_sorted:
        draw_phase_edge(ax, G, pos, u, v, eid, data)

    plt.title(phase_name)
    plt.axis("off")
    plt.savefig(output_base + ".png", dpi=300, bbox_inches='tight')
    plt.close()


# ========================= 主函数 =========================
def draw():
    edges = [
        (AGENT_START, LOGGING_INITIALIZED, "e1", "tool", True),
        (LOGGING_INITIALIZED, CONFIGURATION_READY, "e2", "code", False),
        (CONFIGURATION_READY, DATASET_PATH_EXTRACTED, "e3", "code", False),
        (DATASET_PATH_EXTRACTED, QUESTION_LOADED, "e4", "tool", True),
        (QUESTION_LOADED, QUESTION_FORMATTED, "e5", "code", False),
    ]

    render_phase_graph("demo", edges, "output_graph")


if __name__ == "__main__":
    draw()