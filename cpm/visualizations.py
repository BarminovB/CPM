import io
import base64
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
from .engine import PDMScheduler

@st.cache_resource(show_spinner="Generating Network Diagram...")
def create_network_diagram(scheduler_data: Dict[str, Any], theme: Dict[str, Any]) -> plt.Figure:
    """
    Create a network diagram visualization using NetworkX and Matplotlib.
    Expects scheduler_data from scheduler.to_dict() for caching compatibility.
    """
    # Create temporary scheduler for rendering
    scheduler = PDMScheduler.from_dict(scheduler_data)
    
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

    # Create figure with dynamic size
    num_nodes = len(G.nodes())
    dynamic_width = max(14, int(num_nodes * 0.8))
    dynamic_height = max(10, int(num_nodes * 0.5))
    
    fig, ax = plt.subplots(figsize=(dynamic_width, dynamic_height))

    # Calculate layout
    try:
        # Try to use graphviz layout if available
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=LR')
    except:
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        if any(scheduler.activities[n].es is not None for n in G.nodes()):
            for node in G.nodes():
                act = scheduler.activities[node]
                if act.es is not None:
                    pos[node] = (act.es * 3, pos[node][1])

    # Draw edges
    edge_colors = {
        'FS': theme["edge_fs"],
        'SS': theme["edge_ss"],
        'FF': theme["edge_ff"],
        'SF': theme["edge_sf"],
    }
    edge_styles = {'FS': 'solid', 'SS': 'dashed', 'FF': 'dotted', 'SF': 'dashdot'}

    for edge in G.edges(data=True):
        rel_type = edge[2].get('rel_type', 'FS')
        color = edge_colors.get(rel_type, theme["edge_fs"])
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

    nx.draw_networkx_nodes(G, pos, nodelist=non_critical_nodes,
                          node_color=theme["node_noncrit"], node_size=3000,
                          node_shape='s', ax=ax)

    nx.draw_networkx_nodes(G, pos, nodelist=critical_nodes,
                          node_color=theme["node_crit"], node_size=3000,
                          node_shape='s', ax=ax, edgecolors=theme["critical"], linewidths=3)

    # Node labels
    labels = {}
    for node in G.nodes():
        act = scheduler.activities[node]
        if act.es is not None:
            labels[node] = f"{node}\nD:{act.duration}\nES:{act.es} EF:{act.ef}\nLS:{act.ls} LF:{act.lf}\nTF:{act.total_float}"
        else:
            labels[node] = f"{node}\nD:{act.duration}"

    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

    # Legend
    legend_elements = [
        mpatches.Patch(color=theme["node_crit"], edgecolor=theme["critical"], linewidth=2, label='Critical Activity'),
        mpatches.Patch(color=theme["node_noncrit"], label='Non-Critical Activity'),
        plt.Line2D([0], [0], color=theme["edge_fs"], linewidth=2, linestyle='solid', label='FS (Finish-to-Start)'),
        plt.Line2D([0], [0], color=theme["edge_ss"], linewidth=2, linestyle='dashed', label='SS (Start-to-Start)'),
        plt.Line2D([0], [0], color=theme["edge_ff"], linewidth=2, linestyle='dotted', label='FF (Finish-to-Finish)'),
        plt.Line2D([0], [0], color=theme["edge_sf"], linewidth=2, linestyle='dashdot', label='SF (Start-to-Finish)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8, facecolor='white', frameon=True)
    ax.set_title('Project Network Diagram (PDM - Activity on Node)', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig

@st.cache_resource(show_spinner="Generating Static Gantt...")
def create_gantt_chart(scheduler_data: Dict[str, Any], theme: Dict[str, Any], scale: str = "Day") -> plt.Figure:
    """
    Create a Gantt chart visualization using Matplotlib.
    """
    scheduler = PDMScheduler.from_dict(scheduler_data)
    if not scheduler.activities:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No activities to display', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    sorted_activities = sorted(
        scheduler.activities.values(),
        key=lambda x: (x.es if x.es is not None else 0, x.id),
        reverse=True
    )

    fig, ax = plt.subplots(figsize=(14, max(6, len(sorted_activities) * 0.5)))
    y_positions = range(len(sorted_activities))

    for i, act in enumerate(sorted_activities):
        if act.es is None or act.ef is None:
            continue

        if act.is_critical:
            bar_color = theme["critical"]
            edge_color = theme["critical"]
        else:
            bar_color = theme["noncritical"]
            edge_color = theme["noncritical"]

        ax.barh(i, act.duration, left=act.es, height=0.6,
               color=bar_color, edgecolor=edge_color, linewidth=2)

        bar_center = act.es + act.duration / 2
        ax.text(bar_center, i, f"{act.id} ({act.duration}d)",
               ha='center', va='center', color='white', fontweight='bold', fontsize=9)

        if not act.is_critical and act.total_float and act.total_float > 0:
            ax.barh(i, act.total_float, left=act.ef, height=0.3,
                   color='lightgray', edgecolor='gray', linewidth=1, alpha=0.7)
            ax.text(act.ef + act.total_float/2, i, f'TF:{act.total_float}',
                   ha='center', va='center', fontsize=7, color='gray')

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{act.id}: {act.description[:20]}..."
                        if len(act.description) > 20 else f"{act.id}: {act.description}"
                        for act in sorted_activities])

    ax.set_xlabel(f'Time ({scale}s)', fontsize=12)
    ax.set_ylabel('Activities', fontsize=12)
    ax.set_title('Project Gantt Chart', fontsize=14, fontweight='bold')

    max_days = scheduler.project_duration + max([a.total_float or 0 for a in scheduler.activities.values()] + [0]) + 1
    
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

    ax.set_xlim(-0.5, max_days)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    legend_elements = [
        mpatches.Patch(color=theme["critical"], label='Critical Activity'),
        mpatches.Patch(color=theme["noncritical"], label='Non-Critical Activity'),
        mpatches.Patch(color='lightgray', label='Total Float'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.axvline(x=scheduler.project_duration, color=theme["critical"], linestyle='--', linewidth=2, label='Project End')
    ax.text(scheduler.project_duration, -0.5, f'Day {scheduler.project_duration}',
           ha='center', va='top', color=theme["critical"], fontweight='bold')

    plt.tight_layout()
    return fig

@st.cache_data(show_spinner="Generating Interactive Gantt...")
def create_plotly_gantt(scheduler_data: Dict[str, Any], theme: Dict[str, Any], scale: str = "Day") -> go.Figure:
    """
    Create an interactive Gantt chart using Plotly.
    """
    scheduler = PDMScheduler.from_dict(scheduler_data)
    if not scheduler.activities:
        fig = go.Figure()
        fig.add_annotation(text="No activities to display", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=400)
        return fig

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
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No calculated activities to display", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=400)
        return fig

    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Critical",
        color_discrete_map={"Yes": theme["critical"], "No": theme["noncritical"]},
        hover_data=[
            "ID", "Owner", "Progress", "Risk", "Duration", "ES", "EF", "TF", "Constraint ES",
        ],
        custom_data=["ID", "ES", "EF", "Duration"],
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        height=max(450, len(df) * 32),
        margin=dict(l=10, r=10, t=30, b=10),
        title=f"Interactive Gantt Timeline ({scale} View)",
        xaxis_title="Calendar Timeline",
        yaxis_title="Activities",
        legend_title="Critical",
        template="plotly_white",
        paper_bgcolor=theme["surface"],
        plot_bgcolor=theme["surface"],
        font=dict(color=theme["ink"], family="Space Grotesk"),
    )
    fig.update_xaxes(gridcolor=theme["border"])
    fig.update_yaxes(gridcolor=theme["border"])

    scale_lower = scale.lower()
    if scale_lower == "week":
        fig.update_xaxes(tickmode="linear", dtick=7*24*60*60*1000, tickformat="%b %d", tickangle=-45)
    elif scale_lower == "month":
        fig.update_xaxes(dtick="M1", tickformat="%b %Y", tickangle=-45)
    else: # Day
        fig.update_xaxes(tickmode="linear", dtick=24*60*60*1000, tickformat="%b %d", tickangle=-45)
        
    return fig

@st.cache_data(show_spinner="Generating Interactive Network...")
def create_plotly_network(scheduler_data: Dict[str, Any], theme: Dict[str, Any]) -> go.Figure:
    """
    Create an interactive network diagram using Plotly.
    """
    scheduler = PDMScheduler.from_dict(scheduler_data)
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
            G.add_edge(pred_rel.predecessor_id, act_id, rel_type=pred_rel.relation_type, lag=pred_rel.lag)

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=LR")
    except Exception:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    if any(scheduler.activities[n].es is not None for n in G.nodes()):
        for node in G.nodes():
            act = scheduler.activities[node]
            if act.es is not None:
                pos[node] = (act.es * 2, pos[node][1])

    edge_x, edge_y, edge_text = [], [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"{u} -> {v} ({data.get('rel_type')} {data.get('lag')})")

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color=theme["graph_edge"]), hoverinfo="none", mode="lines")

    node_x, node_y, node_text, node_color = [], [], [], []
    for node in G.nodes():
        act = scheduler.activities[node]
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(theme["critical"] if act.is_critical else theme["noncritical"])
        node_text.append(
            f"{act.id}<br>Owner: {act.owner}<br>Status: {act.status}<br>"
            f"Progress: {act.progress}%<br>Risk: {act.risk}<br>"
            f"Dur: {act.duration}<br>ES/EF: {act.es}/{act.ef}<br>"
            f"LS/LF: {act.ls}/{act.lf}<br>TF: {act.total_float}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=[node for node in G.nodes()],
        textposition="bottom center", hovertext=node_text, hoverinfo="text",
        marker=dict(size=18, color=node_color, line=dict(width=2, color=theme["surface2"])),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Interactive Network Diagram", showlegend=False, height=600, margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white",
        paper_bgcolor=theme["surface"],
        plot_bgcolor=theme["surface"],
        font=dict(color=theme["ink"], family="Space Grotesk"),
    )
    return fig

def fig_to_base64(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    return encoded
