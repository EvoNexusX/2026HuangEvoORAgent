import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import random

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


def pick_t_on_trajectory(eid_num, num_edges, edge_idx):
    # Deterministic random t per edge to keep figure reproducible.
    rng = random.Random(eid_num * 9973 + num_edges * 131 + edge_idx)
    return rng.uniform(0.28, 0.72)


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

    patch = mpatches.FancyArrowPatch(
        posA=pos[u],
        posB=pos[v],
        arrowstyle='-|>',
        connectionstyle=f"arc3,rad={rad}",
        mutation_scale=10,
        linewidth=lw,
        color=color,
        alpha=alpha,
    )
    ax.add_patch(patch)
    eid_num = int(eid[1:])
    t_base = pick_t_on_trajectory(eid_num, num_edges, edge_idx)
    return u, v, rad, eid, t_base


def place_edge_label(ax, pos, u, v, rad, eid, t_base):
    # Sample from Arc3 connection centerline in display coordinates.
    p0_disp = ax.transData.transform(pos[u])
    p1_disp = ax.transData.transform(pos[v])
    centerline_path = mpatches.ConnectionStyle.Arc3(rad=rad).connect(p0_disp, p1_disp)
    vertices_disp = centerline_path.interpolated(320).vertices
    idx = int(max(0, min(len(vertices_disp) - 1, round(t_base * (len(vertices_disp) - 1)))))
    label_disp_x, label_disp_y = vertices_disp[idx]
    label_x, label_y = ax.transData.inverted().transform((label_disp_x, label_disp_y))

    ax.text(
        label_x,
        label_y,
        eid,
        fontsize=5.4,
        color="black",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 1.0},
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

    edges_sorted = sorted(G.edges(keys=True, data=True), key=lambda edge: int(edge[2][1:]))
    edge_label_specs = []
    for u, v, eid, data in edges_sorted:
        edge_label_specs.append(draw_phase_edge(ax, G, pos, u, v, eid, data))

    # Finalize transforms, then place labels on the finalized rendered paths.
    ax.figure.canvas.draw()
    for u, v, rad, eid, t_base in edge_label_specs:
        place_edge_label(ax, pos, u, v, rad, eid, t_base)

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
    # 162 Edges - updated experimental trajectory
    edge_data = [
        # --- Phase 1 (1-55) ---
        ("Agent Start", "Logging Initialized", "e1", "tool", False),
        ("Agent Start", "Logging Initialized", "e2", "code", False),
        ("Agent Start", "Logging Initialized", "e3", "tool", False),
        ("Agent Start", "Configuration Ready", "e4", "code", False),
        ("Agent Start", "Configuration Ready", "e5", "code", False),
        ("Logging Initialized", "Configuration Ready", "e6", "code", False),
        ("Logging Initialized", "Configuration Ready", "e7", "code", False),
        ("Logging Initialized", "Configuration Ready", "e8", "tool", False),
        ("Configuration Ready", "Dataset Path Extracted", "e9", "code", False),
        ("Configuration Ready", "Dataset Path Extracted", "e10", "code", False),
        ("Configuration Ready", "Dataset Path Extracted", "e11", "tool", False),
        ("Dataset Path Extracted", "Question Loaded", "e12", "tool", False),
        ("Dataset Path Extracted", "Question Loaded", "e13", "tool", False),
        ("Dataset Path Extracted", "Question Loaded", "e14", "code", False),
        ("Question Loaded", "Question Formatted", "e15", "code", False),
        ("Question Loaded", "Question Formatted", "e16", "code", False),
        ("Question Loaded", "Question Formatted", "e17", "tool", False),
        ("Question Formatted", "Context Parsed", "e18", "prompt", False),
        ("Question Formatted", "Context Parsed", "e19", "prompt", False),
        ("Question Formatted", "Context Parsed", "e20", "code", False),
        ("Context Parsed", "Domain Identified", "e21", "tool", False),
        ("Context Parsed", "Domain Identified", "e22", "prompt", False),
        ("Context Parsed", "Domain Identified", "e23", "code", False),
        ("Domain Identified", "Sets Defined", "e24", "prompt", False),
        ("Domain Identified", "Sets Defined", "e25", "prompt", False),
        ("Domain Identified", "Sets Defined", "e26", "tool", False),
        ("Question Formatted", "Sets Defined", "e27", "prompt", False),
        ("Question Formatted", "Sets Defined", "e28", "tool", False),
        ("Sets Defined", "Parameters Defined", "e29", "prompt", False),
        ("Sets Defined", "Parameters Defined", "e30", "prompt", False),
        ("Sets Defined", "Parameters Defined", "e31", "code", False),
        ("Parameters Defined", "Variables Defined", "e32", "prompt", False),
        ("Parameters Defined", "Variables Defined", "e33", "prompt", False),
        ("Parameters Defined", "Variables Defined", "e34", "tool", False),
        ("Variables Defined", "Objective Defined", "e35", "prompt", False),
        ("Variables Defined", "Objective Defined", "e36", "prompt", False),
        ("Variables Defined", "Objective Defined", "e37", "code", False),
        ("Objective Defined", "Constraints Defined", "e38", "prompt", False),
        ("Objective Defined", "Constraints Defined", "e39", "prompt", False),
        ("Objective Defined", "Constraints Defined", "e40", "code", False),
        ("Constraints Defined", "Model Drafted", "e41", "prompt", False),
        ("Constraints Defined", "Model Drafted", "e42", "prompt", False),
        ("Constraints Defined", "Model Drafted", "e43", "tool", False),
        ("Model Drafted", "Syntax Checked", "e44", "code", False),
        ("Model Drafted", "Syntax Checked", "e45", "prompt", False),
        ("Model Drafted", "Syntax Checked", "e46", "tool", False),
        ("Syntax Checked", "Final Review", "e47", "prompt", False),
        ("Syntax Checked", "Final Review", "e48", "tool", False),
        ("Syntax Checked", "Final Review", "e49", "code", False),
        ("Final Review", "Phase One Completed", "e50", "tool", False),
        ("Final Review", "Phase One Completed", "e51", "code", False),
        ("Syntax Checked", "Phase One Completed", "e52", "code", False),
        ("Syntax Checked", "Phase One Completed", "e53", "tool", False),
        ("Model Drafted", "Phase One Completed", "e54", "code", False),
        ("Constraints Defined", "Phase One Completed", "e55", "code", False),

        # --- Phase 2 (56-103) ---
        ("Phase One Completed", "Properties Parsed", "e56", "prompt", False),
        ("Phase One Completed", "Properties Parsed", "e57", "prompt", False),
        ("Phase One Completed", "Properties Parsed", "e58", "tool", False),
        ("Phase One Completed", "Limits Evaluated", "e59", "code", False),
        ("Phase One Completed", "Route Decided", "e60", "prompt", False),
        ("Properties Parsed", "Limits Evaluated", "e61", "prompt", False),
        ("Properties Parsed", "Limits Evaluated", "e62", "prompt", False),
        ("Properties Parsed", "Limits Evaluated", "e63", "tool", False),
        ("Limits Evaluated", "Route Decided", "e64", "prompt", False),
        ("Limits Evaluated", "Route Decided", "e65", "prompt", False),
        ("Limits Evaluated", "Route Decided", "e66", "code", False),
        ("Route Decided", "Raw Route Output", "e67", "tool", False),
        ("Route Decided", "Raw Route Output", "e68", "tool", False),
        ("Route Decided", "Raw Route Output", "e69", "tool", False),
        ("Route Decided", "Raw Route Output", "e70", "prompt", False),
        ("Raw Route Output", "Route Extracted", "e71", "code", False),
        ("Raw Route Output", "Route Extracted", "e72", "code", False),
        ("Raw Route Output", "Route Extracted", "e73", "code", False),
        ("Raw Route Output", "Route Extracted", "e74", "tool", False),
        ("Route Extracted", "Algorithm Structured", "e75", "prompt", False),
        ("Route Extracted", "Algorithm Structured", "e76", "prompt", False),
        ("Route Extracted", "Algorithm Structured", "e77", "code", False),
        ("Route Extracted", "Algorithm Structured", "e78", "tool", False),
        ("Algorithm Structured", "Strategy Verified", "e79", "prompt", False),
        ("Algorithm Structured", "Strategy Verified", "e80", "prompt", False),
        ("Algorithm Structured", "Strategy Verified", "e81", "code", False),
        ("Algorithm Structured", "Strategy Verified", "e82", "tool", False),
        ("Strategy Verified", "Raw Phase Two Output", "e83", "tool", False),
        ("Strategy Verified", "Raw Phase Two Output", "e84", "tool", False),
        ("Strategy Verified", "Raw Phase Two Output", "e85", "prompt", False),
        ("Strategy Verified", "Raw Phase Two Output", "e86", "prompt", False),
        ("Raw Phase Two Output", "Strategy Verified", "e87", "code", False),
        ("Raw Phase Two Output", "Strategy Verified", "e88", "code", False),
        ("Raw Phase Two Output", "Strategy Verified", "e89", "code", False),
        ("Raw Phase Two Output", "Strategy Verified", "e90", "tool", False),
        ("Strategy Verified", "Phase Two Completed", "e91", "tool", False),
        ("Strategy Verified", "Phase Two Completed", "e92", "code", False),
        ("Strategy Verified", "Phase Two Completed", "e93", "prompt", False),
        ("Strategy Verified", "Phase Two Completed", "e94", "code", False),
        ("Raw Phase Two Output", "Phase Two Completed", "e95", "code", False),
        ("Raw Phase Two Output", "Phase Two Completed", "e96", "tool", False),
        ("Raw Phase Two Output", "Phase Two Completed", "e97", "code", False),
        ("Route Extracted", "Phase Two Completed", "e98", "code", False),
        ("Route Extracted", "Phase Two Completed", "e99", "code", False),
        ("Route Decided", "Phase Two Completed", "e100", "code", False),
        ("Route Decided", "Phase Two Completed", "e101", "tool", False),
        ("Limits Evaluated", "Phase Two Completed", "e102", "code", False),
        ("Limits Evaluated", "Phase Two Completed", "e103", "code", False),

        # --- Phase 3 (104-162) ---
        ("Phase Two Completed", "Final Text Assembled", "e104", "code", False),
        ("Phase Two Completed", "Final Text Assembled", "e105", "code", False),
        ("Phase Two Completed", "Final Text Assembled", "e106", "tool", False),
        ("Phase Two Completed", "Code Generation Triggered", "e107", "prompt", False),
        ("Phase Two Completed", "Code Generation Triggered", "e108", "tool", False),
        ("Final Text Assembled", "Code Generation Triggered", "e109", "prompt", False),
        ("Final Text Assembled", "Code Generation Triggered", "e110", "prompt", False),
        ("Final Text Assembled", "Code Generation Triggered", "e111", "code", False),
        ("Code Generation Triggered", "Raw Code Generated", "e112", "tool", False),
        ("Code Generation Triggered", "Raw Code Generated", "e113", "tool", False),
        ("Code Generation Triggered", "Raw Code Generated", "e114", "tool", False),
        ("Code Generation Triggered", "Raw Code Generated", "e115", "prompt", False),
        ("Raw Code Generated", "Code Parsed", "e116", "code", False),
        ("Raw Code Generated", "Code Parsed", "e117", "code", False),
        ("Raw Code Generated", "Code Parsed", "e118", "code", False),
        ("Raw Code Generated", "Code Parsed", "e119", "tool", False),
        ("Code Parsed", "Code Saved", "e120", "tool", False),
        ("Code Parsed", "Code Saved", "e121", "prompt", False),
        ("Code Parsed", "Code Saved", "e122", "code", False),
        ("Code Parsed", "Code Saved", "e123", "tool", False),
        ("Code Saved", "Execution Output", "e124", "tool", False),
        ("Code Saved", "Execution Output", "e125", "code", False),
        ("Code Saved", "Execution Output", "e126", "code", False),
        ("Code Saved", "Execution Output", "e127", "tool", False),
        ("Execution Output", "Objective Extracted", "e128", "tool", False),
        ("Execution Output", "Objective Extracted", "e129", "code", False),
        ("Execution Output", "Objective Extracted", "e130", "prompt", False),
        ("Execution Output", "Objective Extracted", "e131", "tool", False),
        ("Objective Extracted", "Type Checked", "e132", "code", False),
        ("Objective Extracted", "Type Checked", "e133", "code", False),
        ("Objective Extracted", "Type Checked", "e134", "prompt", False),
        ("Objective Extracted", "Type Checked", "e135", "tool", False),
        ("Type Checked", "Status Checked", "e136", "code", False),
        ("Type Checked", "Status Checked", "e137", "code", False),
        ("Type Checked", "Status Checked", "e138", "prompt", False),
        ("Type Checked", "Status Checked", "e139", "tool", False),
        ("Status Checked", "Reflection Synthesized", "e140", "prompt", False),
        ("Status Checked", "Reflection Synthesized", "e141", "prompt", False),
        ("Status Checked", "Reflection Synthesized", "e142", "code", False),
        ("Status Checked", "Reflection Synthesized", "e143", "tool", False),
        ("Reflection Synthesized", "Corrected Result", "e144", "tool", False),
        ("Reflection Synthesized", "Corrected Result", "e145", "prompt", False),
        ("Reflection Synthesized", "Corrected Result", "e146", "tool", False),
        ("Reflection Synthesized", "Corrected Result", "e147", "prompt", False),
        ("Corrected Result", "Code Updated", "e148", "code", False),
        ("Corrected Result", "Code Updated", "e149", "code", False),
        ("Corrected Result", "Code Updated", "e150", "tool", False),
        ("Corrected Result", "Code Updated", "e151", "code", False),
        ("Code Updated", "Execution Output", "e152", "code", False),
        ("Code Updated", "Execution Output", "e153", "code", False),
        ("Code Updated", "Execution Output", "e154", "prompt", False),
        ("Code Updated", "Execution Output", "e155", "tool", False),
        ("Status Checked", "Benchmarking Done", "e156", "code", False),
        ("Status Checked", "Benchmarking Done", "e157", "tool", False),
        ("Process Terminated", "Benchmarking Done", "e158", "code", False),
        ("Process Terminated", "Benchmarking Done", "e159", "tool", False),
        ("Execution Output", "Benchmarking Done", "e160", "code", False),
        ("Execution Output", "Benchmarking Done", "e161", "code", False),
        ("Type Checked", "Benchmarking Done", "e162", "code", False),
    ]

    phase_1_edges = [edge for edge in edge_data if 1 <= int(edge[2][1:]) <= 55]
    phase_2_edges = [edge for edge in edge_data if 56 <= int(edge[2][1:]) <= 103]
    phase_3_edges = [edge for edge in edge_data if 104 <= int(edge[2][1:]) <= 162]

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