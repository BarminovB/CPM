"""
IPMA ICB 4.0/4.1 Compliant CPM/PDM Project Scheduler
=====================================================
A complete project network diagram and scheduling application implementing
the Precedence Diagramming Method (PDM) with Activity-on-Node format.

Supports all four precedence relationships:
- FS (Finish-to-Start) - default
- SS (Start-to-Start)
- FF (Finish-to-Finish)
- SF (Start-to-Finish)

Plus positive and negative lags (leads).

Author: IPMA-Compliant Scheduler
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import re
import base64
import io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, List, Dict, Any

from cpm.engine import PDMScheduler
from cpm.models import Relationship

STATUS_OPTIONS = [
    "Not Started",
    "Planned",
    "In Progress",
    "Blocked",
    "Done",
]

RISK_OPTIONS = [
    "Low",
    "Medium",
    "High",
]

THEMES = {
    "Warm Clay": {
        "bg": "#f7f5f2",
        "surface": "#ffffff",
        "surface2": "#fbfaf7",
        "ink": "#1f2328",
        "muted": "#5f6c7b",
        "accent": "#ff6b4a",
        "accent2": "#2c7a7b",
        "border": "#e2ded7",
        "critical": "#ff6b4a",
        "critical_soft": "#ffd2c5",
        "noncritical": "#2c7a7b",
        "node_crit": "#ffc2b3",
        "node_noncrit": "#d7eef0",
        "edge_fs": "#2563eb",
        "edge_ss": "#2c7a7b",
        "edge_ff": "#f59e0b",
        "edge_sf": "#7c3aed",
        "graph_edge": "#c7bfb4",
        "highlight": "#ffe1d6",
        "sidebar_ink": "#f2f2f2",
        "status": {
            "Not Started": "#e9e6e1",
            "Planned": "#e0ecff",
            "In Progress": "#ffe6a7",
            "Blocked": "#ffd6d6",
            "Done": "#d9f5e8",
        },
        "risk": {
            "Low": "#d9f5e8",
            "Medium": "#fff3c4",
            "High": "#ffe0e0",
        },
    },
    "Nordic Blue": {
        "bg": "#f3f6fb",
        "surface": "#ffffff",
        "surface2": "#f2f7ff",
        "ink": "#1c2433",
        "muted": "#5b6b7f",
        "accent": "#3b82f6",
        "accent2": "#0f766e",
        "border": "#dbe3f2",
        "critical": "#f97316",
        "critical_soft": "#ffe2d1",
        "noncritical": "#0f766e",
        "node_crit": "#ffd6c7",
        "node_noncrit": "#d9f0ff",
        "edge_fs": "#3b82f6",
        "edge_ss": "#0f766e",
        "edge_ff": "#f59e0b",
        "edge_sf": "#8b5cf6",
        "graph_edge": "#c7d3e6",
        "highlight": "#ffe7d6",
        "sidebar_ink": "#f2f2f2",
        "status": {
            "Not Started": "#e8eef6",
            "Planned": "#dbeafe",
            "In Progress": "#fef3c7",
            "Blocked": "#fee2e2",
            "Done": "#dcfce7",
        },
        "risk": {
            "Low": "#dcfce7",
            "Medium": "#fef3c7",
            "High": "#fee2e2",
        },
    },
    "Studio Sage": {
        "bg": "#f4f7f3",
        "surface": "#ffffff",
        "surface2": "#f0f4ef",
        "ink": "#1f2a24",
        "muted": "#5f6f66",
        "accent": "#10b981",
        "accent2": "#2563eb",
        "border": "#dfe7de",
        "critical": "#f97316",
        "critical_soft": "#ffe3c2",
        "noncritical": "#10b981",
        "node_crit": "#ffd8b8",
        "node_noncrit": "#dff5ea",
        "edge_fs": "#2563eb",
        "edge_ss": "#10b981",
        "edge_ff": "#f59e0b",
        "edge_sf": "#7c3aed",
        "graph_edge": "#c5d0c6",
        "highlight": "#ffe6d0",
        "sidebar_ink": "#f2f2f2",
        "status": {
            "Not Started": "#e6ebe7",
            "Planned": "#d9f0ff",
            "In Progress": "#fff3c4",
            "Blocked": "#ffdede",
            "Done": "#d9f5e8",
        },
        "risk": {
            "Low": "#d9f5e8",
            "Medium": "#fff3c4",
            "High": "#ffe0e0",
        },
    },
}

ACTIVE_THEME = THEMES["Warm Clay"]
STATUS_COLORS = ACTIVE_THEME["status"]
RISK_COLORS = ACTIVE_THEME["risk"]


def set_active_theme(theme_name: str) -> None:
    global ACTIVE_THEME, STATUS_COLORS, RISK_COLORS
    ACTIVE_THEME = THEMES.get(theme_name, THEMES["Warm Clay"])
    STATUS_COLORS = ACTIVE_THEME["status"]
    RISK_COLORS = ACTIVE_THEME["risk"]


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_network_diagram(scheduler: PDMScheduler) -> plt.Figure:
    """
    Create a network diagram visualization using NetworkX and Matplotlib.

    Nodes represent activities with scheduling information.
    Edges represent precedence relationships with type and lag labels.
    Critical activities are highlighted in red.
    """
    if not scheduler.activities:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No activities to display', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes
    for act_id, act in scheduler.activities.items():
        G.add_node(act_id, activity=act)

    # Add edges with relationship info
    for act_id, act in scheduler.activities.items():
        for pred_rel in act.predecessors:
            lag_str = f"+{pred_rel.lag}" if pred_rel.lag >= 0 else str(pred_rel.lag)
            label = f"{pred_rel.relation_type}({lag_str})"
            G.add_edge(pred_rel.predecessor_id, act_id, label=label,
                      rel_type=pred_rel.relation_type, lag=pred_rel.lag)

    # Create figure with dynamic size based on number of nodes
    # Base size 12x8, add more height/width for complex graphs
    num_nodes = len(G.nodes())
    dynamic_width = max(14, int(num_nodes * 0.8))
    dynamic_height = max(10, int(num_nodes * 0.5))
    
    fig, ax = plt.subplots(figsize=(dynamic_width, dynamic_height))

    # Calculate layout
    try:
        # Try to use graphviz layout if available
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=LR')
    except:
        # Fallback to spring layout with adjustments for left-to-right flow
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

        # Adjust positions to flow left-to-right based on ES values
        if any(scheduler.activities[n].es is not None for n in G.nodes()):
            for node in G.nodes():
                act = scheduler.activities[node]
                if act.es is not None:
                    # Increased horizontal spacing: 2 -> 3
                    pos[node] = (act.es * 3, pos[node][1])

    # Draw edges with different styles based on relationship type
    edge_colors = {
        'FS': ACTIVE_THEME["edge_fs"],
        'SS': ACTIVE_THEME["edge_ss"],
        'FF': ACTIVE_THEME["edge_ff"],
        'SF': ACTIVE_THEME["edge_sf"],
    }
    edge_styles = {'FS': 'solid', 'SS': 'dashed', 'FF': 'dotted', 'SF': 'dashdot'}

    for edge in G.edges(data=True):
        rel_type = edge[2].get('rel_type', 'FS')
        color = edge_colors.get(rel_type, ACTIVE_THEME["edge_fs"])
        style = edge_styles.get(rel_type, 'solid')

        nx.draw_networkx_edges(G, pos, edgelist=[(edge[0], edge[1])],
                               edge_color=color, style=style,
                               arrows=True, arrowsize=20,
                               connectionstyle="arc3,rad=0.1",
                               ax=ax, width=2)

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

    # Draw nodes
    critical_nodes = [n for n in G.nodes() if scheduler.activities[n].is_critical]
    non_critical_nodes = [n for n in G.nodes() if not scheduler.activities[n].is_critical]

    # Draw non-critical nodes
    nx.draw_networkx_nodes(G, pos, nodelist=non_critical_nodes,
                          node_color=ACTIVE_THEME["node_noncrit"], node_size=3000,
                          node_shape='s', ax=ax)

    # Draw critical nodes
    nx.draw_networkx_nodes(G, pos, nodelist=critical_nodes,
                          node_color=ACTIVE_THEME["node_crit"], node_size=3000,
                          node_shape='s', ax=ax, edgecolors=ACTIVE_THEME["critical"], linewidths=3)

    # Create detailed node labels
    labels = {}
    for node in G.nodes():
        act = scheduler.activities[node]
        if act.es is not None:
            labels[node] = f"{node}\nD:{act.duration}\nES:{act.es} EF:{act.ef}\nLS:{act.ls} LF:{act.lf}\nTF:{act.total_float}"
        else:
            labels[node] = f"{node}\nD:{act.duration}"

    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

    # Add legend
    legend_elements = [
        mpatches.Patch(color=ACTIVE_THEME["node_crit"], edgecolor=ACTIVE_THEME["critical"], linewidth=2, label='Critical Activity'),
        mpatches.Patch(color=ACTIVE_THEME["node_noncrit"], label='Non-Critical Activity'),
        plt.Line2D([0], [0], color=ACTIVE_THEME["edge_fs"], linewidth=2, linestyle='solid', label='FS (Finish-to-Start)'),
        plt.Line2D([0], [0], color=ACTIVE_THEME["edge_ss"], linewidth=2, linestyle='dashed', label='SS (Start-to-Start)'),
        plt.Line2D([0], [0], color=ACTIVE_THEME["edge_ff"], linewidth=2, linestyle='dotted', label='FF (Finish-to-Finish)'),
        plt.Line2D([0], [0], color=ACTIVE_THEME["edge_sf"], linewidth=2, linestyle='dashdot', label='SF (Start-to-Finish)'),
    ]
    # Move legend outside to the right
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8, facecolor='white', frameon=True)

    ax.set_title('Project Network Diagram (PDM - Activity on Node)', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    return fig


def create_gantt_chart(scheduler: PDMScheduler, scale: str = "Day") -> plt.Figure:
    """
    Create a Gantt chart visualization using Matplotlib.

    Shows activities as horizontal bars from ES to EF.
    Critical activities are highlighted in red.
    """
    if not scheduler.activities:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No activities to display', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    # Sort activities by ES, then by ID
    sorted_activities = sorted(
        scheduler.activities.values(),
        key=lambda x: (x.es if x.es is not None else 0, x.id),
        reverse=True  # Reverse for bottom-to-top display
    )

    fig, ax = plt.subplots(figsize=(14, max(6, len(sorted_activities) * 0.5)))

    y_positions = range(len(sorted_activities))

    for i, act in enumerate(sorted_activities):
        if act.es is None or act.ef is None:
            continue

        # Determine colors
        if act.is_critical:
            bar_color = ACTIVE_THEME["critical"]
            edge_color = ACTIVE_THEME["critical"]
        else:
            bar_color = ACTIVE_THEME["noncritical"]
            edge_color = ACTIVE_THEME["noncritical"]

        # Draw the activity bar
        ax.barh(i, act.duration, left=act.es, height=0.6,
               color=bar_color, edgecolor=edge_color, linewidth=2)

        # Add activity label inside the bar
        bar_center = act.es + act.duration / 2
        ax.text(bar_center, i, f"{act.id} ({act.duration}d)",
               ha='center', va='center', color='white', fontweight='bold', fontsize=9)

        # Add float indicator if not critical
        if not act.is_critical and act.total_float and act.total_float > 0:
            # Draw total float as a lighter bar
            ax.barh(i, act.total_float, left=act.ef, height=0.3,
                   color='lightgray', edgecolor='gray', linewidth=1, alpha=0.7)
            ax.text(act.ef + act.total_float/2, i, f'TF:{act.total_float}',
                   ha='center', va='center', fontsize=7, color='gray')

    # Set labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{act.id}: {act.description[:20]}..."
                        if len(act.description) > 20 else f"{act.id}: {act.description}"
                        for act in sorted_activities])

    ax.set_xlabel(f'Time ({scale}s)', fontsize=12)
    ax.set_ylabel('Activities', fontsize=12)
    ax.set_title('Project Gantt Chart', fontsize=14, fontweight='bold')

    # Project duration in days
    max_days = scheduler.project_duration + max((a.total_float or 0) for a in scheduler.activities.values()) + 1
    
    # Adjust ticks based on scale
    if scale == "Week":
        major_ticks = np.arange(0, max_days + 1, 7)
        ax.set_xticks(major_ticks)
        ax.set_xticklabels([f"W{int(t/7)}" for t in major_ticks])
    elif scale == "Month":
        major_ticks = np.arange(0, max_days + 1, 30)
        ax.set_xticks(major_ticks)
        ax.set_xticklabels([f"M{int(t/30)}" for t in major_ticks])
    else: # Default Day
        major_ticks = np.arange(0, max_days + 1, max(1, int(max_days/20)))
        ax.set_xticks(major_ticks)

    # Set x-axis limits
    ax.set_xlim(-0.5, max_days)

    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Add legend
    legend_elements = [
        mpatches.Patch(color=ACTIVE_THEME["critical"], label='Critical Activity'),
        mpatches.Patch(color=ACTIVE_THEME["noncritical"], label='Non-Critical Activity'),
        mpatches.Patch(color='lightgray', label='Total Float'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Add vertical line for project end
    ax.axvline(x=scheduler.project_duration, color=ACTIVE_THEME["critical"], linestyle='--', linewidth=2, label='Project End')
    ax.text(scheduler.project_duration, -0.5, f'Day {scheduler.project_duration}',
           ha='center', va='top', color=ACTIVE_THEME["critical"], fontweight='bold')

    plt.tight_layout()
    return fig


def create_plotly_gantt(scheduler: PDMScheduler, scale: str = "Day") -> go.Figure:
    """
    Create an interactive Gantt chart using Plotly.

    Uses ES/EF as offsets from a fixed base date for a clean timeline view.
    """
    if not scheduler.activities:
        fig = go.Figure()
        fig.add_annotation(text="No activities to display", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=400)
        return fig

    # Start date for timeline
    base_date = pd.Timestamp("2026-02-05")
    data = []
    for act in scheduler.activities.values():
        if act.es is None or act.ef is None:
            continue
        data.append(
            {
                "Task": f"{act.id} - {act.description}",
                "Start": base_date + pd.Timedelta(days=int(act.es)),
                "Finish": base_date + pd.Timedelta(days=int(act.ef)),
                "Critical": "Yes" if act.is_critical else "No",
                "ID": act.id,
                "Duration": act.duration,
                "ES": act.es,
                "EF": act.ef,
                "TF": act.total_float,
                "Constraint ES": act.constraint_es,
                "Owner": act.owner,
                "Progress": act.progress,
                "Risk": act.risk,
            }
        )

    df = pd.DataFrame(data)
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Critical",
        color_discrete_map={"Yes": ACTIVE_THEME["critical"], "No": ACTIVE_THEME["noncritical"]},
        hover_data=[
            "ID",
            "Owner",
            "Progress",
            "Risk",
            "Duration",
            "ES",
            "EF",
            "TF",
            "Constraint ES",
        ],
        custom_data=["ID", "ES", "EF", "Duration"],
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        height=max(450, len(df) * 32),
        margin=dict(l=10, r=10, t=30, b=10),
        title=f"Interactive Gantt Timeline ({scale} View)",
        xaxis_title="CalendarTimeline",
        yaxis_title="Activities",
        legend_title="Critical",
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=ACTIVE_THEME["surface"],
        plot_bgcolor=ACTIVE_THEME["surface"],
        font=dict(color=ACTIVE_THEME["ink"], family="Space Grotesk"),
    )
    fig.update_xaxes(gridcolor=ACTIVE_THEME["border"])
    fig.update_yaxes(gridcolor=ACTIVE_THEME["border"])

    # Robust scaling for date-based X-axis
    scale_lower = scale.lower()
    if scale_lower == "week":
        fig.update_xaxes(
            tickmode="linear",
            dtick=7 * 24 * 60 * 60 * 1000, # 1 week in ms
            tickformat="%b %d",
            tickangle=-45
        )
    elif scale_lower == "month":
        fig.update_xaxes(
            tickmode="months",
            dtick="M1",
            tickformat="%b %Y",
            tickangle=-45
        )
    else: # Day
        fig.update_xaxes(
            tickmode="linear",
            dtick=24 * 60 * 60 * 1000, # 1 day in ms
            tickformat="%b %d",
            tickangle=-45
        )
        
    return fig


def create_plotly_network(scheduler: PDMScheduler) -> go.Figure:
    """
    Create an interactive network diagram using Plotly.
    """
    if not scheduler.activities:
        fig = go.Figure()
        fig.add_annotation(text="No activities to display", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=400)
        return fig

    G = nx.DiGraph()
    for act_id, act in scheduler.activities.items():
        G.add_node(act_id, activity=act)
    for act_id, act in scheduler.activities.items():
        for pred_rel in act.predecessors:
            G.add_edge(
                pred_rel.predecessor_id,
                act_id,
                rel_type=pred_rel.relation_type,
                lag=pred_rel.lag,
            )

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=LR")
    except Exception:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    if any(scheduler.activities[n].es is not None for n in G.nodes()):
        for node in G.nodes():
            act = scheduler.activities[node]
            if act.es is not None:
                pos[node] = (act.es * 2, pos[node][1])

    edge_x = []
    edge_y = []
    edge_text = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"{u} -> {v} ({data.get('rel_type')} {data.get('lag')})")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.5, color=ACTIVE_THEME["graph_edge"]),
        hoverinfo="none",
        mode="lines",
    )

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    for node in G.nodes():
        act = scheduler.activities[node]
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(ACTIVE_THEME["critical"] if act.is_critical else ACTIVE_THEME["noncritical"])
        node_text.append(
            f"{act.id}<br>Owner: {act.owner}<br>Status: {act.status}<br>"
            f"Progress: {act.progress}%<br>Risk: {act.risk}<br>"
            f"Dur: {act.duration}<br>ES/EF: {act.es}/{act.ef}<br>"
            f"LS/LF: {act.ls}/{act.lf}<br>TF: {act.total_float}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[node for node in G.nodes()],
        textposition="bottom center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(size=18, color=node_color, line=dict(width=2, color=ACTIVE_THEME["surface2"])),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Interactive Network Diagram",
        showlegend=False,
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=ACTIVE_THEME["surface"],
        plot_bgcolor=ACTIVE_THEME["surface"],
        font=dict(color=ACTIVE_THEME["ink"], family="Space Grotesk"),
    )
    return fig


def _fig_to_base64(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    return encoded


def build_report_html(
    scheduler: PDMScheduler,
    results_df: pd.DataFrame,
    theme: Dict[str, Any],
) -> str:
    try:
        total_activities = len(scheduler.activities) if scheduler.activities else 0
        critical_count = (
            sum(1 for a in scheduler.activities.values() if a.is_critical)
            if scheduler.activities
            else 0
        )
        critical_paths = scheduler.critical_paths or []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        network_fig = create_network_diagram(scheduler)
        gantt_fig = create_gantt_chart(scheduler)
        network_img = _fig_to_base64(network_fig)
        gantt_img = _fig_to_base64(gantt_fig)
        plt.close(network_fig)
        plt.close(gantt_fig)

        table_html = results_df.to_html(index=False, border=0)
        log_text = "\n".join(scheduler.calculation_log)

        critical_paths_html = (
            "<ul>"
            + "".join(f"<li>{' -> '.join(path)}</li>" for path in critical_paths)
            + "</ul>"
            if critical_paths
            else "<p>No critical path identified.</p>"
        )

        return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>CPM Studio Report</title>
  <style>
    :root {{
      --bg: {theme["bg"]};
      --surface: {theme["surface"]};
      --ink: {theme["ink"]};
      --muted: {theme["muted"]};
      --border: {theme["border"]};
      --accent: {theme["accent"]};
    }}
    body {{
      font-family: "Space Grotesk", Arial, sans-serif;
      background: var(--bg);
      color: var(--ink);
      margin: 0;
      padding: 24px;
    }}
    h1, h2, h3 {{
      margin: 0 0 12px 0;
    }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      margin-bottom: 18px;
      box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08);
    }}
    .meta {{
      color: var(--muted);
      font-size: 12px;
    }}
    .kpi {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
    }}
    .kpi div {{
      background: rgba(255,255,255,0.8);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      font-size: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    thead th {{
      text-align: left;
      background: #f4f2ee;
      border-bottom: 1px solid var(--border);
      padding: 8px;
    }}
    tbody td {{
      border-bottom: 1px solid var(--border);
      padding: 6px 8px;
    }}
    img {{
      max-width: 100%;
      height: auto;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #fff;
    }}
    pre {{
      white-space: pre-wrap;
      background: #f6f5f2;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      font-size: 12px;
      color: var(--ink);
    }}
    .formula {{
      font-family: "Courier New", monospace;
      background: #f6f5f2;
      border: 1px dashed var(--border);
      border-radius: 8px;
      padding: 8px 10px;
      margin: 6px 0;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>CPM Studio Report</h1>
    <div class="meta">Generated: {timestamp}</div>
  </div>

  <div class="card kpi">
    <div><strong>Activities</strong><br>{total_activities}</div>
    <div><strong>Project Duration</strong><br>{scheduler.project_duration} days</div>
    <div><strong>Critical Activities</strong><br>{critical_count}</div>
  </div>

  <div class="card">
    <h2>Critical Paths</h2>
    {critical_paths_html}
  </div>

  <div class="card">
    <h2>Formulas</h2>
    <div class="formula">ES = max(Predecessor constraints, Start constraint)</div>
    <div class="formula">EF = ES + Duration</div>
    <div class="formula">LS = min(Successor constraints) - Duration</div>
    <div class="formula">LF = LS + Duration</div>
    <div class="formula">Total Float (TF) = LS - ES</div>
    <div class="formula">Free Float (FF) = min(Successor ES/EF constraints) - Current ES/EF</div>
  </div>

  <div class="card">
    <h2>Schedule Table</h2>
    {table_html}
  </div>

  <div class="card">
    <h2>Network Diagram</h2>
    <img src="data:image/png;base64,{network_img}" alt="Network Diagram"/>
  </div>

  <div class="card">
    <h2>Gantt Chart</h2>
    <img src="data:image/png;base64,{gantt_img}" alt="Gantt Chart"/>
  </div>

  <div class="card">
    <h2>Calculation Log</h2>
    <pre>{log_text}</pre>
  </div>
</body>
</html>
"""
    except Exception as e:
        return f"<html><body><h1>Report Generation Failed</h1><p>{str(e)}</p></body></html>"


def add_dependency_to_scheduler(
    scheduler: PDMScheduler,
    predecessor_id: str,
    successor_id: str,
    rel_type: str,
    lag: int,
) -> Tuple[bool, str]:
    if predecessor_id == successor_id:
        return False, "Predecessor and successor must be different."
    if predecessor_id not in scheduler.activities or successor_id not in scheduler.activities:
        return False, "Both predecessor and successor must exist."
    if rel_type not in PDMScheduler.VALID_RELATIONS:
        return False, "Invalid relationship type."
    for rel in scheduler.activities[successor_id].predecessors:
        if (
            rel.predecessor_id == predecessor_id
            and rel.relation_type == rel_type
            and rel.lag == lag
        ):
            return False, "This dependency already exists."
    scheduler.activities[successor_id].predecessors.append(
        Relationship(predecessor_id, rel_type, lag)
    )
    return True, "Dependency added."


def generate_dependency_suggestions(
    scheduler: PDMScheduler,
    act_id: str,
    tolerance: int = 2,
    max_suggestions: int = 6,
) -> List[Dict[str, object]]:
    if act_id not in scheduler.activities:
        return []
    act = scheduler.activities[act_id]
    if act.es is None or act.ef is None:
        return []

    existing = {
        (rel.predecessor_id, rel.relation_type, rel.lag)
        for rel in act.predecessors
    }
    suggestions: List[Dict[str, object]] = []
    critical_ids = {
        aid for aid, a in scheduler.activities.items() if a.total_float == 0
    }
    act_is_critical = act_id in critical_ids

    for pred_id, pred in scheduler.activities.items():
        if pred_id == act_id:
            continue

        if pred.ef is not None and act.es is not None:
            lag = act.es - pred.ef
            if abs(lag) <= tolerance:
                if (pred_id, "FS", lag) not in existing:
                    score = abs(lag)
                    if pred_id in critical_ids or act_is_critical:
                        score -= 0.3
                    suggestions.append(
                        {
                            "pred": pred_id,
                            "type": "FS",
                            "lag": lag,
                            "score": score,
                            "reason": "FS aligns successor ES to predecessor EF",
                            "critical_hint": pred_id in critical_ids or act_is_critical,
                        }
                    )

        if pred.es is not None and act.es is not None:
            lag = act.es - pred.es
            if abs(lag) <= tolerance:
                if (pred_id, "SS", lag) not in existing:
                    score = abs(lag) + 0.1
                    if pred_id in critical_ids or act_is_critical:
                        score -= 0.2
                    suggestions.append(
                        {
                            "pred": pred_id,
                            "type": "SS",
                            "lag": lag,
                            "score": score,
                            "reason": "SS aligns successor ES to predecessor ES",
                            "critical_hint": pred_id in critical_ids or act_is_critical,
                        }
                    )

        if pred.ef is not None and act.ef is not None:
            lag = act.ef - pred.ef
            if abs(lag) <= tolerance:
                if (pred_id, "FF", lag) not in existing:
                    score = abs(lag) + 0.2
                    if pred_id in critical_ids or act_is_critical:
                        score -= 0.15
                    suggestions.append(
                        {
                            "pred": pred_id,
                            "type": "FF",
                            "lag": lag,
                            "score": score,
                            "reason": "FF aligns successor EF to predecessor EF",
                            "critical_hint": pred_id in critical_ids or act_is_critical,
                        }
                    )

        if pred.es is not None and act.ef is not None:
            lag = act.ef - pred.es
            if abs(lag) <= tolerance:
                if (pred_id, "SF", lag) not in existing:
                    score = abs(lag) + 0.3
                    if pred_id in critical_ids or act_is_critical:
                        score -= 0.1
                    suggestions.append(
                        {
                            "pred": pred_id,
                            "type": "SF",
                            "lag": lag,
                            "score": score,
                            "reason": "SF aligns successor EF to predecessor ES",
                            "critical_hint": pred_id in critical_ids or act_is_critical,
                        }
                    )

    suggestions.sort(key=lambda x: (x["score"], x["pred"], x["type"]))
    return suggestions[:max_suggestions]


def analyze_dependency_conflicts(scheduler: PDMScheduler) -> pd.DataFrame:
    rows = []
    for act in scheduler.activities.values():
        if act.es is None or act.ef is None:
            continue
        for rel in act.predecessors:
            pred = scheduler.activities.get(rel.predecessor_id)
            if not pred or pred.es is None or pred.ef is None:
                continue
            violation = 0
            if rel.relation_type == "FS":
                violation = (pred.ef + rel.lag) - act.es
                constraint = f"ES >= EF({pred.id}) + {rel.lag}"
            elif rel.relation_type == "SS":
                violation = (pred.es + rel.lag) - act.es
                constraint = f"ES >= ES({pred.id}) + {rel.lag}"
            elif rel.relation_type == "FF":
                violation = (pred.ef + rel.lag) - act.ef
                constraint = f"EF >= EF({pred.id}) + {rel.lag}"
            elif rel.relation_type == "SF":
                violation = (pred.es + rel.lag) - act.ef
                constraint = f"EF >= ES({pred.id}) + {rel.lag}"
            else:
                continue

            if violation > 0:
                rows.append(
                    {
                        "Activity": act.id,
                        "Predecessor": pred.id,
                        "Type": rel.relation_type,
                        "Lag": rel.lag,
                        "Violation": violation,
                        "Constraint": constraint,
                    }
                )

    return pd.DataFrame(rows)


def compute_dependency_health(scheduler: PDMScheduler) -> Dict[str, object]:
    total_relations = 0
    for act in scheduler.activities.values():
        total_relations += len(act.predecessors)

    conflicts = analyze_dependency_conflicts(scheduler)
    conflict_count = len(conflicts)
    negative_ff = 0
    for act in scheduler.activities.values():
        if act.free_float is not None and act.free_float < 0:
            negative_ff += 1

    if total_relations == 0:
        base_score = 100
    else:
        base_score = max(0, int(100 - (conflict_count / total_relations) * 100))

    if negative_ff:
        base_score = max(0, base_score - min(20, negative_ff * 5))

    if total_relations < max(1, len(scheduler.activities) - 1):
        base_score = max(0, base_score - 10)

    status = "Healthy"
    if base_score < 70:
        status = "At Risk"
    if base_score < 40:
        status = "Critical"

    return {
        "score": base_score,
        "status": status,
        "total_relations": total_relations,
        "conflicts": conflict_count,
        "negative_ff": negative_ff,
    }


def render_graph_editor(scheduler: PDMScheduler) -> None:
    try:
        from streamlit_agraph import agraph, Node, Edge, Config
    except Exception:
        st.warning("streamlit-agraph is not installed. Add it to requirements.txt.")
        return

    st.caption(
        "Drag nodes to arrange the layout. Click a node to set predecessor, then click another to set successor."
    )

    if "graph_pred" not in st.session_state:
        st.session_state.graph_pred = None
    if "graph_succ" not in st.session_state:
        st.session_state.graph_succ = None

    nodes = []
    edges = []
    for act in scheduler.activities.values():
        label = f"{act.id}\\nD:{act.duration}"
        color = ACTIVE_THEME["critical"] if act.is_critical else ACTIVE_THEME["noncritical"]
        nodes.append(
            Node(id=act.id, label=label, size=25, color=color, shape="box")
        )
    for act in scheduler.activities.values():
        for rel in act.predecessors:
            label = f"{rel.relation_type}({rel.lag:+d})"
            edges.append(
                Edge(
                    source=rel.predecessor_id,
                    target=act.id,
                    label=label,
                    type="CURVE_SMOOTH",
                )
            )

    config = Config(
        width=1000,
        height=600,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor=ACTIVE_THEME["highlight"],
    )

    selection = agraph(nodes=nodes, edges=edges, config=config)
    selected_node = None
    if isinstance(selection, dict):
        selected_node = selection.get("node") or selection.get("selected_node")
    elif isinstance(selection, str):
        selected_node = selection

    if selected_node:
        if st.session_state.graph_pred is None:
            st.session_state.graph_pred = selected_node
        elif st.session_state.graph_succ is None and selected_node != st.session_state.graph_pred:
            st.session_state.graph_succ = selected_node

    if st.session_state.graph_pred or st.session_state.graph_succ:
        st.info(
            f"Selected: {st.session_state.graph_pred or '-'} -> {st.session_state.graph_succ or '-'}"
        )

    activity_ids = sorted(scheduler.activities.keys())

    col_a, col_b, col_c, col_d = st.columns([2, 2, 2, 1])
    with col_a:
        pred_index = 0
        if st.session_state.graph_pred in activity_ids:
            pred_index = 1 + activity_ids.index(st.session_state.graph_pred)
        pred = st.selectbox(
            "Predecessor",
            options=["(select)"] + activity_ids,
            index=pred_index,
            key="graph_pred_select",
        )
    with col_b:
        succ_index = 0
        if st.session_state.graph_succ in activity_ids:
            succ_index = 1 + activity_ids.index(st.session_state.graph_succ)
        succ = st.selectbox(
            "Successor",
            options=["(select)"] + activity_ids,
            index=succ_index,
            key="graph_succ_select",
        )
    with col_c:
        rel_type = st.selectbox("Type", options=["FS", "SS", "FF", "SF"], key="graph_rel_type")
    with col_d:
        lag = st.number_input("Lag", value=0, step=1, key="graph_lag")

    if st.button("Add Dependency", type="primary"):
        pred_id = pred if pred != "(select)" else st.session_state.graph_pred
        succ_id = succ if succ != "(select)" else st.session_state.graph_succ
        if not pred_id or not succ_id:
            st.error("Select predecessor and successor.")
        else:
            ok, msg = add_dependency_to_scheduler(
                scheduler, pred_id, succ_id, rel_type, int(lag)
            )
            if ok:
                st.success(msg)
                st.session_state.calculated = False
                st.session_state.graph_pred = None
                st.session_state.graph_succ = None
                st.rerun()
            else:
                st.error(msg)

    if st.button("Clear Selection"):
        st.session_state.graph_pred = None
        st.session_state.graph_succ = None
        st.rerun()


def render_gantt_editor(scheduler: PDMScheduler) -> None:
    try:
        from streamlit_plotly_events import plotly_events
    except Exception:
        st.warning("streamlit-plotly-events is not installed. Add it to requirements.txt.")
        return

    st.caption("Click a bar to select. Adjust start or duration, then apply.")

    fig = create_plotly_gantt(
        scheduler,
        scale=st.session_state.get("gantt_scale", "Day"),
    )
    fig.update_layout(dragmode="select")
    selected_points = plotly_events(
        fig,
        click_event=True,
        hover_event=False,
        select_event=False,
        override_height=500,
        override_width="100%",
    )

    if "gantt_selected_id" not in st.session_state:
        st.session_state.gantt_selected_id = None

    if selected_points:
        point = selected_points[0]
        custom = point.get("customdata") if isinstance(point, dict) else None
        if custom:
            st.session_state.gantt_selected_id = custom[0]

    activity_ids = sorted(scheduler.activities.keys())
    selected_id = st.selectbox(
        "Selected activity",
        options=["(select)"] + activity_ids,
        index=0
        if st.session_state.gantt_selected_id is None
        else 1 + activity_ids.index(st.session_state.gantt_selected_id),
        key="gantt_selected_select",
    )

    if selected_id == "(select)" and st.session_state.gantt_selected_id:
        selected_id = st.session_state.gantt_selected_id

    if not selected_id or selected_id == "(select)":
        st.info("Select a task from the Gantt chart to edit.")
        return

    act = scheduler.activities[selected_id]
    current_es = int(act.es or 0)
    current_duration = int(act.duration)

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        new_start = st.number_input(
            "Start (ES constraint)",
            value=current_es,
            step=1,
            help="This sets a Start-No-Earlier-Than constraint.",
        )
    with col_b:
        new_duration = st.number_input(
            "Duration",
            min_value=0,
            value=current_duration,
            step=1,
        )
    with col_c:
        if st.button("Clear Constraint"):
            act.constraint_es = None
            scheduler.calculate()
            st.session_state.calculated = True
            st.rerun()

    if st.button("Apply to Schedule", type="primary"):
        act.duration = int(new_duration)
        act.constraint_es = int(new_start)
        scheduler.calculate()
        st.session_state.calculated = True
        st.success("Schedule updated.")
        st.rerun()


# =============================================================================
# STREAMLIT APPLICATION
# =============================================================================

def main():
    """Main Streamlit application."""

    st.set_page_config(
        page_title="CPM Studio",
        page_icon="CPM",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if "theme_choice" not in st.session_state:
        st.session_state.theme_choice = "Warm Clay"

    with st.sidebar:
        st.subheader("Appearance")
        theme_choice = st.selectbox(
            "Theme",
            options=list(THEMES.keys()),
            index=list(THEMES.keys()).index(st.session_state.theme_choice),
            key="theme_choice",
        )

    set_active_theme(theme_choice)
    theme = ACTIVE_THEME

    css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
        :root {
            --cpm-bg: __CPM_BG__;
            --cpm-surface: __CPM_SURFACE__;
            --cpm-ink: __CPM_INK__;
            --cpm-muted: __CPM_MUTED__;
            --cpm-accent: __CPM_ACCENT__;
            --cpm-accent-2: __CPM_ACCENT2__;
            --cpm-border: __CPM_BORDER__;
            --cpm-surface-2: __CPM_SURFACE2__;
            --cpm-sidebar-ink: __CPM_SIDEBAR_INK__;
            --cpm-radius: 16px;
            --cpm-radius-sm: 12px;
            --cpm-shadow: 0 16px 36px rgba(15, 23, 42, 0.10);
        }
        html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
            font-family: "Space Grotesk", sans-serif;
        }
        .stApp {
            background: radial-gradient(
                circle at 10% 0%,
                color-mix(in srgb, var(--cpm-accent) 12%, #ffffff 88%) 0%,
                var(--cpm-bg) 45%,
                var(--cpm-surface) 100%
            );
        }
        [data-testid="stAppViewContainer"] {
            color: var(--cpm-ink);
        }
        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] h4,
        [data-testid="stAppViewContainer"] h5,
        [data-testid="stAppViewContainer"] h6,
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] label {
            color: var(--cpm-ink);
        }
        [data-testid="stAppViewContainer"] .stCaption,
        [data-testid="stAppViewContainer"] small {
            color: var(--cpm-muted);
        }
        [data-testid="stAppViewContainer"] table,
        [data-testid="stAppViewContainer"] thead th,
        [data-testid="stAppViewContainer"] tbody td {
            color: var(--cpm-ink);
        }
        [data-testid="stDataFrame"] *,
        [data-testid="stDataEditor"] * {
            color: var(--cpm-ink);
        }
        [data-testid="stMetricValue"] {
            color: var(--cpm-ink);
        }
        [data-testid="stMetricLabel"] {
            color: var(--cpm-muted);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                color-mix(in srgb, var(--cpm-accent) 6%, var(--cpm-surface) 94%),
                var(--cpm-bg)
            );
            border-right: 1px solid var(--cpm-border);
            box-shadow: inset -1px 0 0 color-mix(in srgb, var(--cpm-accent) 10%, transparent);
        }
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4 {
            padding: 6px 10px;
            border-radius: 10px;
            background: color-mix(in srgb, var(--cpm-accent) 6%, var(--cpm-surface) 94%);
            border: 1px solid color-mix(in srgb, var(--cpm-border) 65%, transparent);
        }
        [data-testid="stSidebar"] hr {
            border: none;
            height: 1px;
            background: color-mix(in srgb, var(--cpm-border) 60%, transparent);
            margin: 16px 0;
        }
        [data-testid="stSidebar"] * {
            color: var(--cpm-ink);
        }
        .cpm-hero {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 24px;
            padding: 20px 28px;
            background: var(--cpm-surface);
            border: 1px solid transparent;
            border-radius: var(--cpm-radius);
            box-shadow: var(--cpm-shadow);
            backdrop-filter: blur(12px);
            position: relative;
        }
        .cpm-hero::before,
        .cpm-metric::before {
            content: "";
            position: absolute;
            inset: 0;
            padding: 1px;
            border-radius: inherit;
            background: linear-gradient(135deg, color-mix(in srgb, var(--cpm-accent) 35%, transparent), rgba(255,255,255,0.2));
            -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            pointer-events: none;
        }
        .cpm-hero::after {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: inherit;
            background: linear-gradient(135deg, rgba(255,255,255,0.55), rgba(255,255,255,0.1));
            pointer-events: none;
        }
        .cpm-hero > * {
            position: relative;
            z-index: 1;
        }
        .cpm-hero h1 {
            font-size: 28px;
            margin: 0 0 6px 0;
            color: var(--cpm-ink);
        }
        .cpm-hero p {
            margin: 0;
            color: var(--cpm-muted);
            font-size: 14px;
        }
        .cpm-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 999px;
            border: 1px solid var(--cpm-border);
            background: var(--cpm-surface-2);
            color: var(--cpm-muted);
            font-size: 12px;
            margin-right: 8px;
            transition: 0.2s ease;
        }
        .cpm-chip:hover {
            color: var(--cpm-ink);
            border-color: color-mix(in srgb, var(--cpm-accent) 50%, var(--cpm-border) 50%);
            background: color-mix(in srgb, var(--cpm-accent) 10%, var(--cpm-surface-2) 90%);
        }
        .cpm-metric {
            padding: 16px;
            border-radius: var(--cpm-radius);
            border: 1px solid transparent;
            background: var(--cpm-surface);
            backdrop-filter: blur(10px);
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08);
            position: relative;
        }
        .cpm-metric:hover,
        .cpm-hero:hover {
            box-shadow: 0 20px 36px rgba(15, 23, 42, 0.16);
            transform: translateY(-1px);
            transition: 0.25s ease;
        }
        .cpm-status {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid var(--cpm-border);
            font-size: 12px;
            margin-right: 6px;
            color: var(--cpm-ink);
        }
        .stButton > button {
            border-radius: var(--cpm-radius-sm);
            font-weight: 600;
            padding: 10px 18px;
            border: 1px solid var(--cpm-border);
            box-shadow: 0 10px 20px rgba(15, 23, 42, 0.10);
            transition: 0.2s ease;
            background: var(--cpm-surface-2);
            color: var(--cpm-ink);
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 28px rgba(15, 23, 42, 0.18);
        }
        .stButton > button[kind="primary"] {
            background: var(--cpm-accent);
            color: #ffffff;
            border: none;
        }
        .stButton > button:disabled {
            opacity: 0.6;
            color: var(--cpm-muted);
            background: var(--cpm-surface-2);
            border-color: var(--cpm-border);
        }
        .stButton > button[kind="primary"]:hover {
            background: color-mix(in srgb, var(--cpm-accent) 85%, #000 15%);
            color: #ffffff;
        }
        .stButton > button:focus-visible {
            outline: none;
            box-shadow: 0 0 0 4px color-mix(in srgb, var(--cpm-accent) 25%, transparent);
        }
        :focus-visible {
            outline: none;
            box-shadow: 0 0 0 3px color-mix(in srgb, var(--cpm-accent) 35%, transparent);
            border-radius: var(--cpm-radius-sm);
        }
        [data-testid="stDataFrame"] :focus-visible,
        [data-testid="stDataEditor"] :focus-visible {
            box-shadow: 0 0 0 3px color-mix(in srgb, var(--cpm-accent) 30%, transparent);
        }
        .cpm-focus-trail:focus-within {
            position: relative;
        }
        .cpm-focus-trail:focus-within::after {
            content: "";
            position: absolute;
            inset: -6px;
            border-radius: calc(var(--cpm-radius) + 6px);
            border: 1px solid color-mix(in srgb, var(--cpm-accent) 35%, transparent);
            box-shadow: 0 0 0 6px color-mix(in srgb, var(--cpm-accent) 12%, transparent);
            animation: focusTrail 0.4s ease;
            pointer-events: none;
        }
        @keyframes focusTrail {
            from { opacity: 0; transform: scale(0.98); }
            to { opacity: 1; transform: scale(1); }
        }
        [data-testid="stTabs"] button {
            transition: 0.2s ease;
            border-radius: 12px;
        }
        [data-testid="stTabs"] button:hover {
            background: color-mix(in srgb, var(--cpm-accent) 8%, transparent);
        }
        [data-testid="stTabs"] [aria-selected="true"] {
            background: color-mix(in srgb, var(--cpm-accent) 18%, transparent);
            color: var(--cpm-ink);
            box-shadow: inset 0 -2px 0 var(--cpm-accent);
        }
        .cpm-shimmer {
            position: relative;
            overflow: hidden;
            background: color-mix(in srgb, var(--cpm-accent) 6%, var(--cpm-surface) 94%);
            border-radius: 12px;
            height: 12px;
            border: 1px solid var(--cpm-border);
        }
        .cpm-shimmer::after {
            content: "";
            position: absolute;
            inset: 0;
            transform: translateX(-100%);
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);
            animation: shimmer 1.6s infinite;
        }
        @keyframes shimmer {
            100% { transform: translateX(100%); }
        }
        .cpm-glass {
            background: color-mix(in srgb, var(--cpm-surface) 70%, transparent);
            border-radius: var(--cpm-radius);
            border: 1px solid color-mix(in srgb, var(--cpm-border) 60%, transparent);
            box-shadow: 0 18px 32px rgba(15, 23, 42, 0.10);
            backdrop-filter: blur(12px);
            padding: 12px;
        }
        .cpm-fade-in {
            animation: fadeInUp 0.4s ease;
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            border-radius: var(--cpm-radius-sm);
            background: var(--cpm-surface);
            border: 1px solid var(--cpm-border);
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
            transition: 0.2s ease;
        }
        .stTextInput input:hover,
        .stNumberInput input:hover,
        .stTextArea textarea:hover,
        .stSelectbox div[data-baseweb="select"] > div:hover,
        .stMultiSelect div[data-baseweb="select"] > div:hover {
            border-color: color-mix(in srgb, var(--cpm-accent) 25%, var(--cpm-border) 75%);
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.12);
        }
        .stTextInput input:focus,
        .stNumberInput input:focus,
        .stTextArea textarea:focus,
        .stSelectbox div[data-baseweb="select"] > div:focus,
        .stMultiSelect div[data-baseweb="select"] > div:focus {
            outline: none;
            border-color: color-mix(in srgb, var(--cpm-accent) 55%, var(--cpm-border) 45%);
            box-shadow: 0 0 0 4px color-mix(in srgb, var(--cpm-accent) 20%, transparent);
        }
        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stNumberInput input,
        [data-testid="stSidebar"] .stTextArea textarea,
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
            background: var(--cpm-surface);
            border-color: var(--cpm-border);
            color: var(--cpm-ink);
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
            outline: none;
            appearance: none;
        }
        [data-testid="stSidebar"] div[data-baseweb="input"] > div,
        [data-testid="stSidebar"] div[data-baseweb="input"] input,
        [data-testid="stSidebar"] div[data-baseweb="textarea"] > div,
        [data-testid="stSidebar"] div[data-baseweb="textarea"] textarea {
            border: 1px solid var(--cpm-border) !important;
            background: var(--cpm-surface) !important;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08) !important;
            outline: none !important;
        }
        [data-testid="stSidebar"] .stTextInput input:hover,
        [data-testid="stSidebar"] .stNumberInput input:hover,
        [data-testid="stSidebar"] .stTextArea textarea:hover,
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div:hover {
            border-color: color-mix(in srgb, var(--cpm-accent) 25%, var(--cpm-border) 75%);
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.12);
        }
        [data-testid="stSidebar"] .stTextInput input:focus,
        [data-testid="stSidebar"] .stNumberInput input:focus,
        [data-testid="stSidebar"] .stTextArea textarea:focus,
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div:focus {
            border-color: color-mix(in srgb, var(--cpm-accent) 55%, var(--cpm-border) 45%);
            box-shadow: 0 0 0 4px color-mix(in srgb, var(--cpm-accent) 18%, transparent);
        }
        [data-testid="stSidebar"] div[data-baseweb="input"]:focus-within > div,
        [data-testid="stSidebar"] div[data-baseweb="input"]:focus-within input,
        [data-testid="stSidebar"] div[data-baseweb="textarea"]:focus-within > div,
        [data-testid="stSidebar"] div[data-baseweb="textarea"]:focus-within textarea {
            border-color: color-mix(in srgb, var(--cpm-accent) 55%, var(--cpm-border) 45%) !important;
            box-shadow: 0 0 0 4px color-mix(in srgb, var(--cpm-accent) 18%, transparent) !important;
        }
        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stNumberInput input {
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
        }
        [data-testid="stNumberInput"] button {
            background: var(--cpm-surface-2);
            border: 1px solid var(--cpm-border);
            border-radius: 10px;
            box-shadow: 0 6px 14px rgba(15, 23, 42, 0.08);
        }
        [data-testid="stNumberInput"] div[data-baseweb="button-group"] {
            background: var(--cpm-surface);
            border-radius: 10px;
            padding: 2px;
        }
        [data-testid="stNumberInput"] button svg {
            color: var(--cpm-accent);
            fill: var(--cpm-accent);
        }
        [data-testid="stNumberInput"] button:hover {
            background: color-mix(in srgb, var(--cpm-accent) 10%, var(--cpm-surface-2) 90%);
            border-color: color-mix(in srgb, var(--cpm-accent) 35%, var(--cpm-border) 65%);
        }
        [data-testid="stDataFrame"] th,
        [data-testid="stDataFrame"] td,
        [data-testid="stDataEditor"] th,
        [data-testid="stDataEditor"] td {
            background: var(--cpm-surface);
            border-color: var(--cpm-border);
        }
        [data-testid="stDataFrame"] thead th,
        [data-testid="stDataEditor"] thead th {
            background: var(--cpm-surface-2);
            color: var(--cpm-ink);
        }
        [data-testid="stDataFrame"] table,
        [data-testid="stDataEditor"] table {
            border-radius: var(--cpm-radius-sm);
            overflow: hidden;
        }
        [data-testid="stDataFrame"],
        [data-testid="stDataEditor"] {
            background: var(--cpm-surface);
            border-radius: calc(var(--cpm-radius) + 2px);
            border: 1px solid color-mix(in srgb, var(--cpm-border) 70%, transparent);
            box-shadow: 0 16px 28px rgba(15, 23, 42, 0.10);
            padding: 6px;
        }
        [data-testid="stDataFrame"] tbody tr:hover td,
        [data-testid="stDataEditor"] tbody tr:hover td {
            background: color-mix(in srgb, var(--cpm-accent) 6%, var(--cpm-surface) 94%);
        }
        [data-testid="stAlert"] {
            background: color-mix(in srgb, var(--cpm-accent) 10%, var(--cpm-surface) 90%);
            border-color: color-mix(in srgb, var(--cpm-accent) 25%, var(--cpm-border) 75%);
        }
        [data-testid="stAlert"] code {
            background: color-mix(in srgb, var(--cpm-accent) 10%, var(--cpm-surface) 90%);
            color: var(--cpm-ink);
            border: 1px solid color-mix(in srgb, var(--cpm-accent) 25%, var(--cpm-border) 75%);
            border-radius: 8px;
            padding: 2px 6px;
        }
        [data-testid="stMarkdown"] code {
            background: color-mix(in srgb, var(--cpm-accent) 10%, var(--cpm-surface) 90%);
            color: var(--cpm-ink);
            border: 1px solid color-mix(in srgb, var(--cpm-accent) 25%, var(--cpm-border) 75%);
            border-radius: 8px;
            padding: 2px 6px;
        }
        [data-testid="stMarkdown"] pre {
            background: color-mix(in srgb, var(--cpm-accent) 6%, var(--cpm-surface) 94%);
            border-radius: 12px;
            border: 1px solid color-mix(in srgb, var(--cpm-border) 65%, transparent);
            padding: 12px;
        }
        </style>
        """
    replacements = {
        "__CPM_BG__": theme["bg"],
        "__CPM_SURFACE__": theme["surface"],
        "__CPM_INK__": theme["ink"],
        "__CPM_MUTED__": theme["muted"],
        "__CPM_ACCENT__": theme["accent"],
        "__CPM_ACCENT2__": theme["accent2"],
        "__CPM_BORDER__": theme["border"],
        "__CPM_SURFACE2__": theme["surface2"],
        "__CPM_SIDEBAR_INK__": theme["sidebar_ink"],
    }
    for token, value in replacements.items():
        css = css.replace(token, value)

    st.markdown(css, unsafe_allow_html=True)


    st.markdown(
        """
        <div class="cpm-hero">
            <div>
                <h1>CPM Studio</h1>
                <p>Professional PDM scheduling workspace with critical path analytics and visual planning.</p>
                <div>
                    <span class="cpm-chip">PDM Engine</span>
                    <span class="cpm-chip">Multi-Relation Links</span>
                    <span class="cpm-chip">Critical Paths</span>
                </div>
            </div>
            <div style="text-align:right">
                <div style="font-size:12px;color:var(--cpm-muted);">Workspace</div>
                <div style="font-size:18px;font-weight:600;color:var(--cpm-ink);">{scheduler.project_name}</div>
                <div style="font-size:12px;color:var(--cpm-muted);">Last sync: auto</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if 'scheduler' not in st.session_state:
        st.session_state.scheduler = PDMScheduler()
    if 'calculated' not in st.session_state:
        st.session_state.calculated = False
    if 'pred_list' not in st.session_state:
        st.session_state.pred_list = []

    scheduler = st.session_state.scheduler
    # Safety check for migration/session state persistence
    if not hasattr(scheduler, 'project_name'):
        scheduler.project_name = "Default Portfolio"

    def _is_missing(value: object) -> bool:
        try:
            return value is None or pd.isna(value)
        except Exception:
            return False

    def _safe_str(value: object) -> str:
        if _is_missing(value):
            return ""
        return str(value).strip()

    def _safe_int(value: object, default: int = 0) -> int:
        if _is_missing(value):
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return default

    with st.sidebar:
        st.subheader("Project Settings")
        scheduler.project_name = st.text_input(
            "Project Name",
            value=scheduler.project_name,
            key="project_name_input"
        )
        st.divider()
        st.subheader("Create Activity")

        st.markdown("<div class='cpm-focus-trail'>", unsafe_allow_html=True)
        activity_id = st.text_input(
            "Activity ID",
            placeholder="e.g., A, B, TASK1",
            help="Unique identifier (starts with letter, letters/numbers/underscores)",
            key="activity_id_input"
        ).strip().upper()

        description = st.text_input(
            "Description",
            placeholder="e.g., Design phase",
            key="description_input"
        )

        status = st.selectbox(
            "Status",
            options=STATUS_OPTIONS,
            index=0,
            key="status_input",
        )

        owner = st.text_input(
            "Owner",
            placeholder="e.g., Alex, PMO Team",
            key="owner_input",
        )

        progress = st.slider(
            "Progress %",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            key="progress_input",
        )

        risk = st.selectbox(
            "Risk",
            options=RISK_OPTIONS,
            index=0,
            key="risk_input",
        )

        duration = st.number_input(
            "Duration (days)",
            min_value=0,
            value=1,
            step=1,
            key="duration_input"
        )

        st.markdown("**Predecessors** (add multiple if needed)")

        if st.button("Add Predecessor", key="add_pred_btn"):
            st.session_state.pred_list.append({"pred_id": "", "rel_type": "FS", "lag": 0})
            st.rerun()

        predecessors = []
        to_remove = []

        existing_activities = ["(none)"] + sorted(list(scheduler.activities.keys()))

        for idx, pred in enumerate(st.session_state.pred_list):
            cols = st.columns([3, 2, 2, 1])
            with cols[0]:
                selected_id = st.selectbox(
                    f"Predecessor #{idx+1}",
                    options=existing_activities,
                    index=existing_activities.index(pred.get("pred_id", "")) if pred.get("pred_id") in existing_activities else 0,
                    key=f"pred_id_{idx}"
                )
            with cols[1]:
                rel_type = st.selectbox(
                    "Type",
                    options=["FS", "SS", "FF", "SF"],
                    index=["FS", "SS", "FF", "SF"].index(pred.get("rel_type", "FS")),
                    key=f"rel_type_{idx}"
                )
            with cols[2]:
                lag = st.number_input(
                    "Lag (days)",
                    value=pred.get("lag", 0),
                    step=1,
                    key=f"lag_{idx}"
                )
            with cols[3]:
                if st.button("Remove", key=f"remove_pred_{idx}"):
                    to_remove.append(idx)

            if idx < len(st.session_state.pred_list):
                st.session_state.pred_list[idx]["pred_id"] = selected_id
                st.session_state.pred_list[idx]["rel_type"] = rel_type
                st.session_state.pred_list[idx]["lag"] = lag

            if selected_id != "(none)":
                predecessors.append(Relationship(selected_id, rel_type, int(lag)))

        if to_remove:
            for idx in sorted(to_remove, reverse=True):
                del st.session_state.pred_list[idx]
            st.rerun()

        if st.button("Add Activity", type="primary", use_container_width=True, key="add_activity_btn"):
            if not activity_id:
                st.error("Activity ID is required.")
            elif not re.match(r'^[A-Z][A-Z0-9_]*$', activity_id):
                st.error("Invalid Activity ID format.")
            elif activity_id in scheduler.activities:
                st.error(f"Activity '{activity_id}' already exists.")
            else:
                pred_str = ";".join(f"{p.predecessor_id}:{p.relation_type}:{p.lag}" for p in predecessors)
                success, message = scheduler.add_activity(
                    activity_id,
                    description,
                    duration,
                    pred_str,
                    status,
                    owner,
                    progress,
                    risk,
                )
                if success:
                    st.success(f"Activity {activity_id} added with {len(predecessors)} predecessor(s).")
                    st.session_state.pred_list = []
                    st.session_state.calculated = False
                    st.rerun()
                else:
                    st.error(message)
        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        with st.expander("Predecessor Format Guide", expanded=False):
            st.markdown("""
            **Relationship Types:**
            - `FS` - Finish-to-Start (default)
            - `SS` - Start-to-Start
            - `FF` - Finish-to-Finish
            - `SF` - Start-to-Finish

            **Format:** `ID:TYPE:LAG`

            **Examples:**
            - `A:FS:0` - Start after A finishes
            - `B:SS:5` - Start 5 days after B starts
            - `C:FF:-3` - Finish 3 days before C finishes (lead)
            - `D:SF:2` - Finish 2 days after D starts

            **Multiple predecessors:** Separate with semicolons
            `A:FS:0;B:SS:5;C:FF:-3`
            """)

        with st.expander("Sample Projects", expanded=False):
            if st.button("Load Sample Project", use_container_width=True):
                scheduler.clear()
                sample_activities = [
                    ("A", "Project Planning", 3, ""),
                    ("B", "Requirements Analysis", 5, "A:FS:0"),
                    ("C", "Design", 4, "B:FS:0"),
                    ("D", "Development Phase 1", 8, "C:FS:0"),
                    ("E", "Development Phase 2", 6, "C:FS:0;D:SS:2"),
                    ("F", "Testing", 5, "D:FS:0;E:FF:0"),
                    ("G", "Documentation", 3, "D:FS:0"),
                    ("H", "Deployment", 2, "F:FS:0;G:FS:0"),
                ]
                for act_id, desc, dur, preds in sample_activities:
                    scheduler.add_activity(act_id, desc, dur, preds)
                st.success("Sample project loaded.")
                st.session_state.calculated = False
                st.rerun()

            if st.button("Load Complex Sample (All Relations)", use_container_width=True):
                scheduler.clear()
                complex_activities = [
                    ("A", "Foundation Work", 5, ""),
                    ("B", "Parallel Prep Work", 3, "A:SS:2"),
                    ("C", "Main Construction", 10, "A:FS:0;B:FS:0"),
                    ("D", "Finishing Work", 4, "C:FF:-2"),
                    ("E", "Inspection", 2, "C:FS:0;D:SF:1"),
                    ("F", "Final Review", 1, "E:FS:0"),
                ]
                for act_id, desc, dur, preds in complex_activities:
                    scheduler.add_activity(act_id, desc, dur, preds)
                st.success("Complex sample loaded.")
                st.session_state.calculated = False
                st.rerun()

            if st.button("Clear All Activities", use_container_width=True, type="secondary"):
                scheduler.clear()
                st.session_state.calculated = False
                st.success("All activities cleared.")
                st.rerun()

        with st.expander("Import / Export", expanded=False):
            st.caption("CSV columns: ID, Description, Status, Owner, Progress, Risk, Duration, Predecessors")

            export_df = scheduler.get_activities_dataframe()
            # Add project name to export
            export_df.insert(0, "Project Name", scheduler.project_name)
            
            csv_data = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Export CSV",
                data=csv_data,
                file_name=f"{scheduler.project_name.replace(' ', '_').lower()}_project.csv",
                mime="text/csv",
                use_container_width=True,
            )

            uploaded_file = st.file_uploader("Import CSV", type=["csv"], key="import_csv")
            replace_current = st.checkbox(
                "Replace current project",
                value=False,
                key="import_replace",
            )

            if st.button("Import CSV", use_container_width=True, key="import_btn"):
                if uploaded_file is None:
                    st.warning("Choose a CSV file to import.")
                else:
                    try:
                        df = pd.read_csv(uploaded_file)
                    except Exception as exc:
                        st.error(f"Failed to read CSV: {exc}")
                    else:
                        if df.empty:
                            st.error("CSV file is empty.")
                        else:
                            df.columns = [str(c).strip().lower() for c in df.columns]
                            required_cols = {"id", "duration"}
                            missing = required_cols - set(df.columns)
                            if missing:
                                st.error(
                                    "Missing required columns: "
                                    + ", ".join(sorted(missing))
                                )
                            else:
                                def load_df_into(
                                    scheduler_obj: PDMScheduler,
                                    frame: pd.DataFrame,
                                    label: str,
                                ) -> tuple[int, list[str]]:
                                    errors: list[str] = []
                                    count = 0
                                    for idx, row in frame.iterrows():
                                        act_id = _safe_str(row.get("id")).upper()
                                        if not act_id:
                                            errors.append(f"{label} row {idx+1}: Missing ID.")
                                            continue
                                        description = _safe_str(row.get("description"))
                                        owner = _safe_str(row.get("owner"))
                                        duration = _safe_int(row.get("duration"), default=0)
                                        predecessors = _safe_str(row.get("predecessors"))
                                        if predecessors and re.fullmatch(r"[-\u2013\u2014]+", predecessors):
                                            predecessors = ""
                                        status = _safe_str(row.get("status")) or "Not Started"
                                        if status not in STATUS_OPTIONS:
                                            status = "Not Started"
                                        progress = _safe_int(row.get("progress"), default=0)
                                        risk = _safe_str(row.get("risk")) or "Low"
                                        if risk not in RISK_OPTIONS:
                                            risk = "Low"
                                        ok, msg = scheduler_obj.add_activity(
                                            act_id,
                                            description,
                                            duration,
                                            predecessors,
                                            status,
                                            owner,
                                            progress,
                                            risk,
                                        )
                                        if not ok:
                                            errors.append(f"{label} row {idx+1}: {msg}")
                                        else:
                                            count += 1
                                    return count, errors

                                new_scheduler = PDMScheduler()
                                errors: list[str] = []
                                if not replace_current and scheduler.activities:
                                    base_df = scheduler.get_activities_dataframe()
                                    base_df.columns = [str(c).strip().lower() for c in base_df.columns]
                                    _, base_errors = load_df_into(new_scheduler, base_df, "Existing")
                                    errors.extend(base_errors)

                                imported_count, import_errors = load_df_into(new_scheduler, df, "Import")
                                errors.extend(import_errors)

                                if errors:
                                    st.error("Import failed:\n" + "\n".join(errors))
                                else:
                                    # Try to get project name from CSV
                                    if "project name" in df.columns:
                                        new_project_name = _safe_str(df["project name"].iloc[0])
                                        if new_project_name:
                                            new_scheduler.project_name = new_project_name
                                            
                                    st.session_state.scheduler = new_scheduler
                                    st.session_state.calculated = False
                                    st.success(f"Imported {imported_count} activities.")
                                    st.rerun()

    col1, col2 = st.columns([2.4, 1.4])

    with col1:
        board_head = st.columns([3, 1])
        with board_head[0]:
            st.header("Board")
        with board_head[1]:
            edit_mode = st.toggle("Edit mode", value=True, key="board_edit_mode")

        if scheduler.activities:
            board_df = scheduler.get_activities_dataframe()
        else:
            board_df = pd.DataFrame(
                columns=[
                    "ID",
                    "Description",
                    "Status",
                    "Owner",
                    "Progress",
                    "Risk",
                    "Duration",
                    "Predecessors",
                ]
            )

        if edit_mode:
            st.markdown("<div class='cpm-focus-trail'>", unsafe_allow_html=True)
            edited_df = st.data_editor(
                board_df,
                use_container_width=True,
                num_rows="dynamic",
                hide_index=True,
                column_config={
                    "ID": st.column_config.TextColumn("ID", required=True),
                    "Description": st.column_config.TextColumn("Description"),
                    "Status": st.column_config.SelectboxColumn(
                        "Status",
                        options=STATUS_OPTIONS,
                        required=True,
                    ),
                    "Owner": st.column_config.TextColumn("Owner"),
                    "Progress": st.column_config.NumberColumn(
                        "Progress",
                        min_value=0,
                        max_value=100,
                        step=5,
                    ),
                    "Risk": st.column_config.SelectboxColumn(
                        "Risk",
                        options=RISK_OPTIONS,
                        required=True,
                    ),
                    "Duration": st.column_config.NumberColumn("Duration", min_value=0, step=1),
                    "Predecessors": st.column_config.TextColumn(
                        "Predecessors",
                        help="Format: A:FS:0;B:SS:2",
                    ),
                },
                key="board_editor",
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            legend_html = " ".join(
                [
                    f"<span class='cpm-status' style='background:{STATUS_COLORS[s]};'>{s}</span>"
                    for s in STATUS_OPTIONS
                ]
            )
            st.markdown(legend_html, unsafe_allow_html=True)
            risk_html = " ".join(
                [
                    f"<span class='cpm-status' style='background:{RISK_COLORS[r]};'>Risk: {r}</span>"
                    for r in RISK_OPTIONS
                ]
            )
            st.markdown(risk_html, unsafe_allow_html=True)

            st.markdown("<div class='cpm-focus-trail'>", unsafe_allow_html=True)
            filter_cols = st.columns([2, 2, 2, 1])
            with filter_cols[0]:
                status_filter = st.multiselect(
                    "Status filter",
                    options=STATUS_OPTIONS,
                    default=STATUS_OPTIONS,
                )
            with filter_cols[1]:
                risk_filter = st.multiselect(
                    "Risk filter",
                    options=RISK_OPTIONS,
                    default=RISK_OPTIONS,
                )
            with filter_cols[2]:
                search_text = st.text_input("Search")
            with filter_cols[3]:
                critical_only = st.checkbox("Critical only", value=False)
            st.markdown("</div>", unsafe_allow_html=True)

            view_df = board_df.copy()
            if status_filter:
                view_df = view_df[view_df["Status"].isin(status_filter)]
            if risk_filter:
                view_df = view_df[view_df["Risk"].isin(risk_filter)]
            if search_text:
                search_text_lower = search_text.lower()
                id_series = view_df["ID"].fillna("").str.lower()
                desc_series = view_df["Description"].fillna("").str.lower()
                owner_series = view_df["Owner"].fillna("").str.lower()
                view_df = view_df[
                    id_series.str.contains(search_text_lower)
                    | desc_series.str.contains(search_text_lower)
                    | owner_series.str.contains(search_text_lower)
                ]
            if critical_only:
                if st.session_state.calculated:
                    critical_ids = [
                        act.id for act in scheduler.activities.values() if act.is_critical
                    ]
                    view_df = view_df[view_df["ID"].isin(critical_ids)]
                else:
                    st.info("Run calculation to filter by critical activities.")

            def status_style(value: str) -> str:
                color = STATUS_COLORS.get(value, ACTIVE_THEME["surface2"])
                return f"background-color: {color}; color: {ACTIVE_THEME['ink']};"

            def risk_style(value: str) -> str:
                color = RISK_COLORS.get(value, ACTIVE_THEME["surface2"])
                return f"background-color: {color}; color: {ACTIVE_THEME['ink']};"

            styled = (
                view_df.style
                .applymap(status_style, subset=["Status"])
                .applymap(risk_style, subset=["Risk"])
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)
            edited_df = None


        action_cols = st.columns([1, 1, 2])
        with action_cols[0]:
            if st.button("Apply Changes", type="primary", disabled=not edit_mode):
                new_scheduler = PDMScheduler()
                errors = []
                for row in edited_df.to_dict("records"):
                    act_id = _safe_str(row.get("ID")).upper()
                    if not act_id:
                        continue
                    description = _safe_str(row.get("Description"))
                    owner = _safe_str(row.get("Owner"))
                    duration = _safe_int(row.get("Duration"), default=0)
                    predecessors = _safe_str(row.get("Predecessors"))
                    if predecessors and re.fullmatch(r"[-\u2013\u2014]+", predecessors):
                        predecessors = ""
                    status = _safe_str(row.get("Status")) or "Not Started"
                    if status not in STATUS_OPTIONS:
                        status = "Not Started"
                    progress = _safe_int(row.get("Progress"), default=0)
                    risk = _safe_str(row.get("Risk")) or "Low"
                    if risk not in RISK_OPTIONS:
                        risk = "Low"
                    ok, msg = new_scheduler.add_activity(
                        act_id, description, duration, predecessors, status, owner, progress, risk
                    )
                    if not ok:
                        errors.append(msg)
                if errors:
                    st.error("Cannot apply changes:\n" + "\n".join(errors))
                else:
                    st.session_state.scheduler = new_scheduler
                    st.session_state.calculated = False
                    st.success("Board updated.")
                    st.rerun()

        with action_cols[1]:
            if scheduler.activities:
                act_to_remove = st.selectbox(
                    "Remove",
                    options=list(scheduler.activities.keys()),
                    label_visibility="collapsed",
                )
                if st.button("Delete Selected"):
                    success, message = scheduler.remove_activity(act_to_remove)
                    if success:
                        st.success(message)
                        st.session_state.calculated = False
                        st.rerun()
                    else:
                        st.error(message)
        with action_cols[2]:
            st.caption("Tip: edit cells inline, then Apply Changes.")

    with col2:
        # Aligning top of button with top of Board table
        # st.header("Board") roughly corresponds to ~60px including margins
        st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)

        _, centered_col, _ = st.columns([1, 2, 1])
        with centered_col:
            if st.button(
                "Calculate Schedule",
                type="primary",
                use_container_width=True,
                disabled=len(scheduler.activities) == 0,
            ):
                success, message = scheduler.calculate()
                if success:
                    st.session_state.calculated = True
                    st.success(message)
                else:
                    st.error(message)

            total_activities = len(scheduler.activities) if scheduler.activities else 0
            st.metric("Activities", f"{total_activities}")

            if st.session_state.calculated:
                st.metric("Project Duration", f"{scheduler.project_duration} days")
                st.metric(
                    "Critical Activities",
                    f"{sum(1 for a in scheduler.activities.values() if a.is_critical)}",
                )



    if st.session_state.calculated and scheduler.activities:
        st.divider()
        st.header("Calculation Results")

        results_df = scheduler.get_results_dataframe()

        st.markdown("<div class='cpm-glass cpm-fade-in'>", unsafe_allow_html=True)
        def highlight_critical(row):
            if row['Critical'] == 'Yes':
                return [f'background-color: {ACTIVE_THEME["critical_soft"]}'] * len(row)
            return [''] * len(row)

        styled_df = results_df.style.apply(highlight_critical, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Critical Paths")
        st.markdown("<div class='cpm-glass cpm-fade-in'>", unsafe_allow_html=True)
        if scheduler.critical_paths:
            for idx, path in enumerate(scheduler.critical_paths, start=1):
                st.markdown(f"**{idx}. {' -> '.join(path)}**")
        else:
            st.info("No critical path identified.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Gantt Overview")
        st.markdown("<div class='cpm-glass cpm-fade-in'>", unsafe_allow_html=True)
        gantt_scale = st.selectbox(
            "Gantt scale",
            options=["Day", "Week", "Month"],
            index=0,
            key="gantt_scale",
        )
        gantt_fig = create_plotly_gantt(scheduler, scale=gantt_scale)
        st.plotly_chart(gantt_fig, use_container_width=True, key="gantt_overview")
        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Report")
        st.markdown("<div class='cpm-glass cpm-fade-in'>", unsafe_allow_html=True)
        if st.button("Generate Report", type="primary", use_container_width=True):
            st.session_state.report_html = build_report_html(
                scheduler,
                results_df,
                ACTIVE_THEME,
            )
        if "report_html" in st.session_state:
            st.download_button(
                "Download Report (HTML)",
                data=st.session_state.report_html.encode("utf-8"),
                file_name="cpm_report.html",
                mime="text/html",
                use_container_width=True,
            )
            with st.expander("Report Preview", expanded=False):
                st.components.v1.html(
                    st.session_state.report_html,
                    height=700,
                    scrolling=True,
                )
            st.caption("Open the HTML and use your browser print dialog to export PDF.")
        st.markdown("</div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "Network Diagram",
                "Gantt Chart",
                "Graph Editor",
                "Gantt Editor",
                "Calculation Details",
            ]
        )

        with tab1:
            st.subheader("Network Diagram (PDM - Activity on Node)")
            fig = create_network_diagram(scheduler)
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Legend: Red nodes = Critical activities | Blue nodes = Non-critical activities. "
                "Edge styles: Solid=FS, Dashed=SS, Dotted=FF, Dashdot=SF."
            )

        with tab2:
            st.subheader("Gantt Chart")
            fig = create_gantt_chart(scheduler, scale=gantt_scale)
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Legend: Red bars = Critical activities | Blue bars = Non-critical activities. "
                "Gray extensions show Total Float available for non-critical activities."
            )

        with tab3:
            st.subheader("Graph Editor")
            if scheduler.activities:
                st.markdown("<div class='cpm-focus-trail'>", unsafe_allow_html=True)
                render_graph_editor(scheduler)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Add activities first to edit dependencies.")

        with tab4:
            st.subheader("Gantt Editor")
            if scheduler.activities:
                st.markdown("<div class='cpm-focus-trail'>", unsafe_allow_html=True)
                render_gantt_editor(scheduler)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Add activities first to edit the schedule.")

        with tab5:
            st.subheader("Detailed Calculation Log")
            log_text = "\n".join(scheduler.calculation_log)
            st.text_area("Calculation Steps", value=log_text, height=500, disabled=True)

            st.info(
                "This calculation follows the standard PDM method with all four relationship types, "
                "positive or negative lags, forward/backward passes, floats, and critical path detection."
            )

        st.divider()
        st.header("Dependency Health")
        st.markdown("<div class='cpm-glass cpm-fade-in'>", unsafe_allow_html=True)
        if st.session_state.calculated:
            health = compute_dependency_health(scheduler)
            col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1])
            with col_a:
                st.metric("Health Score", f"{health['score']} / 100")
            with col_b:
                st.metric("Status", health["status"])
            with col_c:
                st.metric("Relations", f"{health['total_relations']}")
            with col_d:
                st.metric("Conflicts", f"{health['conflicts']}")

            conflicts = analyze_dependency_conflicts(scheduler)
            if conflicts.empty:
                st.success("No dependency conflicts detected.")
            else:
                st.warning("Dependency conflicts detected.")
                st.dataframe(conflicts, use_container_width=True, hide_index=True)
        else:
            st.info("Run calculation to evaluate dependency health.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown(
        """
        ---
        **CPM Studio** | Precedence Diagramming Method | Activity-on-Node Format

        *Professional CPM/PDM scheduling workspace.*
        """
    )


if __name__ == "__main__":
    main()
