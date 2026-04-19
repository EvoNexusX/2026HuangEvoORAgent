import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

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


def has_label_overlap(candidate_x, candidate_y, used_label_positions, min_dist):
    return any(
        (candidate_x - px) ** 2 + (candidate_y - py) ** 2 < min_dist ** 2
        for px, py in used_label_positions
    )


def select_label_position_on_arc(arc_point, t_base, used_label_positions, min_dist):
    # All candidates are sampled on the arc/line itself, so labels always lie on edges.
    local_deltas = [0.0, 0.04, -0.04, 0.08, -0.08, 0.12, -0.12, 0.16, -0.16, 0.20, -0.20]
    t_candidates = [max(0.18, min(0.82, t_base + delta)) for delta in local_deltas]
    t_candidates += [0.18 + i * 0.04 for i in range(17)]

    # Keep unique ordering while preserving order preference.
    unique_candidates = []
    for t_value in t_candidates:
        if t_value not in unique_candidates:
            unique_candidates.append(t_value)

    best_xy = arc_point(max(0.18, min(0.82, t_base)))
    best_score = float("-inf")
    min_dist_sq = min_dist ** 2

    for t_value in unique_candidates:
        candidate_x, candidate_y = arc_point(t_value)
        if not used_label_positions:
            nearest_sq = float("inf")
        else:
            nearest_sq = min(
                (candidate_x - px) ** 2 + (candidate_y - py) ** 2
                for px, py in used_label_positions
            )

        overlap_penalty = 0 if nearest_sq >= min_dist_sq else -1_000_000
        # Prefer larger separation first, then keep close to preferred t_base.
        score = overlap_penalty + nearest_sq - abs(t_value - t_base) * 0.25
        if score > best_score:
            best_score = score
            best_xy = (candidate_x, candidate_y)

    return best_xy


def draw_phase_edge(ax, graph, pos, u, v, eid, data, used_label_positions):
    etype = data["etype"]
    is_macro = data["is_macro"]

    color_map = {"code": "#34a853", "prompt": "#ea4335", "tool": "#4285f4"}
    color = color_map.get(etype, "gray")

    edge_idx = list(graph[u][v]).index(eid)
    num_edges = len(graph[u][v])
    rad = 0.0 if num_edges == 1 else (edge_idx - (num_edges - 1) / 2.0) * 0.12
    rad = 0.35 + (edge_idx * 0.1) if pos[u][0] > pos[v][0] else rad

    lw = 1.35 if is_macro else 1.0
    alpha = 0.85 if is_macro else 0.6

    ax.annotate(
        "",
        xy=pos[v],
        xycoords="data",
        xytext=pos[u],
        textcoords="data",
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "shrinkA": 12,
            "shrinkB": 12,
            "connectionstyle": f"arc3,rad={rad}",
            "linewidth": lw,
            "alpha": alpha,
        },
    )

    # Label anchor: place text on the same curved trajectory used by arc3,rad.
    eid_num = int(eid[1:])
    ux, uy = pos[u]
    vx, vy = pos[v]

    # Direction-aware + ID-based jitter keeps labels on their own arcs and avoids visual merging.
    direction_bias = -0.07 if ux > vx else 0.07
    id_jitter = ((eid_num % 7) - 3) * 0.012
    t_base = 0.5 + (edge_idx - (num_edges - 1) / 2.0) * 0.08 + direction_bias + id_jitter
    t_base = max(0.22, min(0.78, t_base))

    def arc_point(t_value):
        one_minus_t = 1.0 - t_value
        dx, dy = vx - ux, vy - uy
        mid_x, mid_y = (ux + vx) / 2.0, (uy + vy) / 2.0
        # arc3,rad control-point approximation in data coordinates.
        ctrl_x = mid_x - dy * rad
        ctrl_y = mid_y + dx * rad
        bx = (one_minus_t ** 2) * ux + 2 * one_minus_t * t_value * ctrl_x + (t_value ** 2) * vx
        by = (one_minus_t ** 2) * uy + 2 * one_minus_t * t_value * ctrl_y + (t_value ** 2) * vy
        return bx, by

    min_dist = 0.26
    label_x, label_y = select_label_position_on_arc(arc_point, t_base, used_label_positions, min_dist)

    used_label_positions.append((label_x, label_y))

    ax.text(
        label_x,
        label_y,
        eid,
        fontsize=5.4,
        color="black",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "#d9d9d9", "alpha": 0.88},
        zorder=12,
        clip_on=False,
    )


def render_phase_graph(phase_name, phase_edges, output_base):
    G = nx.MultiDiGraph()
    ordered_nodes = []

    def register_node(node_name):
        if node_name not in ordered_nodes:
            ordered_nodes.append(node_name)

    for u, v, eid, etype, is_macro in phase_edges:
        G.add_edge(u, v, key=eid, etype=etype, is_macro=is_macro)
        register_node(u)
        register_node(v)

    x_step = 1.45
    lane_pattern = [0.0, 0.9, -0.9, 1.55, -1.55, 0.45, -0.45]
    vertical_bias = 0.22
    pos = {}
    for index, node in enumerate(ordered_nodes):
        lane_y = lane_pattern[index % len(lane_pattern)]
        wave_y = 0.22 * math.sin(index * 0.85)
        pos[node] = (index * x_step, lane_y + wave_y + vertical_bias)

    figure_width = max(16, len(ordered_nodes) * 1.18)
    figure_height = figure_width * 0.75
    plt.figure(figsize=(figure_width, figure_height))

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=980,
        node_color="#e8f0fe",
        edgecolors="#1a73e8",
        linewidths=1.35,
    )

    ax = plt.gca()
    for node_name, (x_pos, y_pos) in pos.items():
        ax.text(
            x_pos,
            y_pos - 0.12,
            node_name,
            fontsize=7.8,
            fontweight="bold",
            ha="center",
            va="top",
            color="#1f1f1f",
        )

    used_label_positions = []
    edges_sorted = sorted(G.edges(keys=True, data=True), key=lambda edge: int(edge[2][1:]))
    for u, v, eid, data in edges_sorted:
        draw_phase_edge(ax, G, pos, u, v, eid, data, used_label_positions)

    legend_elements = [
        mpatches.Patch(color="#34a853", label="Code Verification (Micro)"),
        mpatches.Patch(color="#ea4335", label="Prompt Generation (Micro)"),
        mpatches.Patch(color="#4285f4", label="Tool Invocation (Micro)"),
    ]
    plt.legend(handles=legend_elements, loc="lower left", fontsize=10, frameon=False)
    plt.title(phase_name, fontsize=15, fontweight="bold", pad=12)
    plt.axis("off")
    plt.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.16)
    pdf_path = f"{output_base}.pdf"
    png_path = f"{output_base}.png"
    try:
        plt.savefig(pdf_path, format="pdf", dpi=500, bbox_inches="tight")
        plt.savefig(png_path, format="png", dpi=500, bbox_inches="tight")
    except PermissionError:
        pdf_path = f"{output_base}_nonlinear.pdf"
        png_path = f"{output_base}_nonlinear.png"
        plt.savefig(pdf_path, format="pdf", dpi=500, bbox_inches="tight")
        plt.savefig(png_path, format="png", dpi=500, bbox_inches="tight")
    plt.close()
    print(f"Graph generation complete. Saved as {pdf_path} and {png_path}")

def draw_fully_expanded_aoe():
    # 125 Edges - Fully spelled out node names, expanded logic
    edge_data = [
        # --- Stage: System Initialization ---
        (AGENT_START, LOGGING_INITIALIZED, "e1", "tool", True),
        (AGENT_START, LOGGING_INITIALIZED, "e2", "code", False),
        (AGENT_START, LOGGING_INITIALIZED, "e3", "code", False),
        (LOGGING_INITIALIZED, CONFIGURATION_READY, "e4", "code", True),
        (LOGGING_INITIALIZED, CONFIGURATION_READY, "e5", "code", False),
        (CONFIGURATION_READY, DATASET_PATH_EXTRACTED, "e6", "code", True),
        (CONFIGURATION_READY, DATASET_PATH_EXTRACTED, "e7", "code", False),
        (DATASET_PATH_EXTRACTED, QUESTION_LOADED, "e8", "tool", True),
        
        # --- Stage: Context Parsing ---
        (QUESTION_LOADED, QUESTION_FORMATTED, "e9", "code", True),
        (QUESTION_LOADED, QUESTION_FORMATTED, "e10", "code", False),
        (QUESTION_FORMATTED, CONTEXT_PARSED, "e11", "prompt", False),
        (QUESTION_FORMATTED, CONTEXT_PARSED, "e12", "prompt", False),
        (CONTEXT_PARSED, DOMAIN_IDENTIFIED, "e13", "tool", False),
        (CONTEXT_PARSED, DOMAIN_IDENTIFIED, "e14", "code", False),
        (DOMAIN_IDENTIFIED, PREPROCESSING_COMPLETED, "e15", "prompt", False),
        (DOMAIN_IDENTIFIED, PREPROCESSING_COMPLETED, "e16", "tool", False),
        (PREPROCESSING_COMPLETED, SETS_DRAFTED, "e17", "prompt", False),
        
        # --- Stage: Sets ---
        (QUESTION_FORMATTED, SETS_DEFINED, "e18", "prompt", True), # MACRO
        (PREPROCESSING_COMPLETED, SETS_DRAFTED, "e19", "prompt", False),
        (SETS_DRAFTED, SETS_RAW_OUTPUT, "e20", "tool", False),
        (SETS_DRAFTED, SETS_RAW_OUTPUT, "e21", "tool", False),
        (SETS_RAW_OUTPUT, SETS_VERIFIED, "e22", "code", False),
        (SETS_RAW_OUTPUT, SETS_VERIFIED, "e23", "prompt", False),
        (SETS_VERIFIED, SETS_CHECKED, "e24", "tool", False),
        (SETS_VERIFIED, SETS_CHECKED, "e25", "code", False),
        (SETS_CHECKED, SETS_REFINED, "e26", "prompt", False),
        (SETS_CHECKED, SETS_REFINED, "e27", "code", False),
        (SETS_REFINED, SETS_DEFINED, "e28", "tool", False),
        (SETS_REFINED, SETS_DEFINED, "e29", "code", False),
        (SETS_CHECKED, SETS_DRAFTED, "e30", "code", False), # BACKTRACK
        
        # --- Stage: Parameters ---
        (SETS_DEFINED, PARAMETERS_DEFINED, "e31", "prompt", True), # MACRO
        (SETS_DEFINED, PARAMETERS_DRAFTED, "e32", "prompt", False),
        (SETS_DEFINED, PARAMETERS_DRAFTED, "e33", "prompt", False),
        (SETS_DEFINED, PARAMETERS_DRAFTED, "e34", "prompt", False),
        (PARAMETERS_DRAFTED, PARAMETERS_RAW_OUTPUT, "e35", "tool", False),
        (PARAMETERS_DRAFTED, PARAMETERS_RAW_OUTPUT, "e36", "tool", False),
        (PARAMETERS_RAW_OUTPUT, PARAMETERS_BOUNDED, "e37", "code", False),
        (PARAMETERS_RAW_OUTPUT, PARAMETERS_BOUNDED, "e38", "prompt", False),
        (PARAMETERS_RAW_OUTPUT, PARAMETERS_BOUNDED, "e39", "prompt", False),
        (PARAMETERS_BOUNDED, PARAMETERS_CHECKED, "e40", "tool", False),
        (PARAMETERS_BOUNDED, PARAMETERS_CHECKED, "e41", "code", False),
        (PARAMETERS_CHECKED, PARAMETERS_REFINED, "e42", "prompt", False),
        (PARAMETERS_CHECKED, PARAMETERS_REFINED, "e43", "prompt", False),
        (PARAMETERS_REFINED, PARAMETERS_DEFINED, "e44", "tool", False),
        (PARAMETERS_REFINED, PARAMETERS_DEFINED, "e45", "code", False),
        (PARAMETERS_CHECKED, PARAMETERS_DRAFTED, "e46", "code", False), # BACKTRACK
        
        # --- Stage: Variables ---
        (PARAMETERS_DEFINED, VARIABLES_DEFINED, "e47", "prompt", True), # MACRO
        (PARAMETERS_DEFINED, VARIABLES_DRAFTED, "e48", "prompt", False),
        (PARAMETERS_DEFINED, VARIABLES_DRAFTED, "e49", "prompt", False),
        (PARAMETERS_DEFINED, VARIABLES_DRAFTED, "e50", "prompt", False),
        (VARIABLES_DRAFTED, VARIABLES_RAW_OUTPUT, "e51", "tool", False),
        (VARIABLES_DRAFTED, VARIABLES_RAW_OUTPUT, "e52", "tool", False),
        (VARIABLES_RAW_OUTPUT, VARIABLES_TYPED, "e53", "prompt", False),
        (VARIABLES_RAW_OUTPUT, VARIABLES_TYPED, "e54", "code", False),
        (VARIABLES_RAW_OUTPUT, VARIABLES_TYPED, "e55", "prompt", False),
        (VARIABLES_TYPED, VARIABLES_BOUNDED, "e56", "tool", False),
        (VARIABLES_TYPED, VARIABLES_BOUNDED, "e57", "code", False),
        (VARIABLES_BOUNDED, VARIABLES_VERIFIED, "e58", "prompt", False),
        (VARIABLES_BOUNDED, VARIABLES_VERIFIED, "e59", "prompt", False),
        (VARIABLES_VERIFIED, VARIABLES_CHECKED, "e60", "tool", False),
        (VARIABLES_VERIFIED, VARIABLES_CHECKED, "e61", "code", False),
        (VARIABLES_CHECKED, VARIABLES_REFINED, "e62", "code", False),
        (VARIABLES_CHECKED, VARIABLES_REFINED, "e63", "prompt", False),
        (VARIABLES_REFINED, VARIABLES_DEFINED, "e64", "tool", False),
        (VARIABLES_REFINED, VARIABLES_DEFINED, "e65", "code", False),
        (VARIABLES_VERIFIED, VARIABLES_DRAFTED, "e66", "code", False), # BACKTRACK
        
        # --- Stage: Objective ---
        (VARIABLES_DEFINED, OBJECTIVE_DEFINED, "e67", "prompt", True), # MACRO
        (VARIABLES_DEFINED, OBJECTIVE_DRAFTED, "e68", "prompt", False),
        (VARIABLES_DEFINED, OBJECTIVE_DRAFTED, "e69", "prompt", False),
        (VARIABLES_DEFINED, OBJECTIVE_DRAFTED, "e70", "prompt", False),
        (OBJECTIVE_DRAFTED, OBJECTIVE_RAW_OUTPUT, "e71", "tool", False),
        (OBJECTIVE_DRAFTED, OBJECTIVE_RAW_OUTPUT, "e72", "tool", False),
        (OBJECTIVE_RAW_OUTPUT, OBJECTIVE_LINEARIZED, "e73", "prompt", False),
        (OBJECTIVE_RAW_OUTPUT, OBJECTIVE_LINEARIZED, "e74", "code", False),
        (OBJECTIVE_LINEARIZED, OBJECTIVE_CHECKED, "e75", "tool", False),
        (OBJECTIVE_LINEARIZED, OBJECTIVE_CHECKED, "e76", "code", False),
        (OBJECTIVE_CHECKED, OBJECTIVE_REFINED, "e77", "prompt", False),
        (OBJECTIVE_CHECKED, OBJECTIVE_REFINED, "e78", "prompt", False),
        (OBJECTIVE_REFINED, OBJECTIVE_DEFINED, "e79", "tool", False),
        (OBJECTIVE_REFINED, OBJECTIVE_DEFINED, "e80", "code", False),
        (OBJECTIVE_CHECKED, OBJECTIVE_DRAFTED, "e81", "code", False), # BACKTRACK
        
        # --- Stage: Constraints ---
        (OBJECTIVE_DEFINED, CONSTRAINTS_DEFINED, "e82", "prompt", True), # MACRO
        (OBJECTIVE_DEFINED, CONSTRAINTS_DRAFTED, "e83", "prompt", False),
        (OBJECTIVE_DEFINED, CONSTRAINTS_DRAFTED, "e84", "prompt", False),
        (OBJECTIVE_DEFINED, CONSTRAINTS_DRAFTED, "e85", "prompt", False),
        (CONSTRAINTS_DRAFTED, CONSTRAINTS_RAW_OUTPUT, "e86", "tool", False),
        (CONSTRAINTS_DRAFTED, CONSTRAINTS_RAW_OUTPUT, "e87", "tool", False),
        (CONSTRAINTS_RAW_OUTPUT, CONSTRAINTS_LOGIC, "e88", "prompt", False),
        (CONSTRAINTS_RAW_OUTPUT, CONSTRAINTS_LOGIC, "e89", "prompt", False),
        (CONSTRAINTS_RAW_OUTPUT, CONSTRAINTS_LOGIC, "e90", "code", False),
        (CONSTRAINTS_LOGIC, CONSTRAINTS_BOUNDED, "e91", "tool", False),
        (CONSTRAINTS_LOGIC, CONSTRAINTS_BOUNDED, "e92", "code", False),
        (CONSTRAINTS_BOUNDED, CONSTRAINTS_VERIFIED, "e93", "prompt", False),
        (CONSTRAINTS_BOUNDED, CONSTRAINTS_VERIFIED, "e94", "prompt", False),
        (CONSTRAINTS_VERIFIED, CONSTRAINTS_CHECKED, "e95", "tool", False),
        (CONSTRAINTS_VERIFIED, CONSTRAINTS_CHECKED, "e96", "code", False),
        (CONSTRAINTS_CHECKED, CONSTRAINTS_REFINED, "e97", "code", False),
        (CONSTRAINTS_CHECKED, CONSTRAINTS_REFINED, "e98", "prompt", False),
        (CONSTRAINTS_REFINED, CONSTRAINTS_DEFINED, "e99", "tool", False),
        (CONSTRAINTS_REFINED, CONSTRAINTS_DEFINED, "e100", "code", False),
        (CONSTRAINTS_VERIFIED, CONSTRAINTS_DRAFTED, "e101", "code", False), # BACKTRACK
        
        # --- Stage: Synthesis & Verification ---
        (CONSTRAINTS_DEFINED, MODEL_GENERATION_INITIATED, "e102", "tool", True), # MACRO
        (CONSTRAINTS_DEFINED, MODEL_DRAFTED, "e103", "prompt", False),
        (CONSTRAINTS_DEFINED, MODEL_DRAFTED, "e104", "prompt", False),
        (MODEL_DRAFTED, MODEL_RAW_OUTPUT, "e105", "tool", False),
        (MODEL_RAW_OUTPUT, SYNTAX_CHECKED, "e106", "code", False),
        (MODEL_RAW_OUTPUT, SYNTAX_CHECKED, "e107", "prompt", False),
        (SYNTAX_CHECKED, MODEL_CLEANED, "e108", "tool", False),
        (SYNTAX_CHECKED, MODEL_CLEANED, "e109", "code", False),
        (MODEL_CLEANED, KNOWLEDGE_BASE_QUERIED, "e110", "prompt", False),
        (KNOWLEDGE_BASE_QUERIED, KNOWLEDGE_BASE_INTEGRATED, "e111", "tool", False),
        (KNOWLEDGE_BASE_QUERIED, KNOWLEDGE_BASE_INTEGRATED, "e112", "code", False),
        (KNOWLEDGE_BASE_INTEGRATED, FINAL_REVIEW, "e113", "prompt", False),
        (KNOWLEDGE_BASE_INTEGRATED, FINAL_REVIEW, "e114", "code", False),
        (FINAL_REVIEW, STAGE_ONE_OUTPUT_GENERATED, "e115", "tool", False),
        (FINAL_REVIEW, STAGE_ONE_OUTPUT_GENERATED, "e116", "tool", False),
        (STAGE_ONE_OUTPUT_GENERATED, TEXT_CONTEXT_READY, "e117", "code", False),
        (STAGE_ONE_OUTPUT_GENERATED, TEXT_CONTEXT_READY, "e118", "tool", False),
        (TEXT_CONTEXT_READY, PHASE_ONE_COMPLETED, "e119", "code", True),
        (TEXT_CONTEXT_READY, PHASE_ONE_COMPLETED, "e120", "code", False),
        (SYNTAX_CHECKED, MODEL_DRAFTED, "e121", "code", False), # BACKTRACK
        (FINAL_REVIEW, MODEL_DRAFTED, "e122", "code", False), # BACKTRACK
        (TEXT_CONTEXT_READY, PHASE_ONE_COMPLETED, "e123", "prompt", False),
        (TEXT_CONTEXT_READY, PHASE_ONE_COMPLETED, "e124", "tool", False),
        (MODEL_GENERATION_INITIATED, TEXT_CONTEXT_READY, "e125", "code", True) # MACRO
    ]

    phase_1_edges = [edge for edge in edge_data if 1 <= int(edge[2][1:]) <= 30]
    phase_2_edges = [edge for edge in edge_data if 31 <= int(edge[2][1:]) <= 81]
    phase_3_edges = [edge for edge in edge_data if 82 <= int(edge[2][1:]) <= 125]

    render_phase_graph(
        "phase1",
        phase_1_edges,
        "AOE_Phase_1",
    )
    render_phase_graph(
        "phase2",
        phase_2_edges,
        "AOE_Phase_2",
    )
    render_phase_graph(
        "phase3",
        phase_3_edges,
        "AOE_Phase_3",
    )

if __name__ == "__main__":
    draw_fully_expanded_aoe()