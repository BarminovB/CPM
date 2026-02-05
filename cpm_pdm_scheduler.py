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
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import re

from cpm.engine import PDMScheduler
from cpm.models import Relationship


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

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

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
                    pos[node] = (act.es * 2, pos[node][1])

    # Draw edges with different styles based on relationship type
    edge_colors = {'FS': '#2196F3', 'SS': '#4CAF50', 'FF': '#FF9800', 'SF': '#9C27B0'}
    edge_styles = {'FS': 'solid', 'SS': 'dashed', 'FF': 'dotted', 'SF': 'dashdot'}

    for edge in G.edges(data=True):
        rel_type = edge[2].get('rel_type', 'FS')
        color = edge_colors.get(rel_type, '#2196F3')
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
                          node_color='lightblue', node_size=3000,
                          node_shape='s', ax=ax)

    # Draw critical nodes
    nx.draw_networkx_nodes(G, pos, nodelist=critical_nodes,
                          node_color='#ffcccb', node_size=3000,
                          node_shape='s', ax=ax, edgecolors='red', linewidths=3)

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
        mpatches.Patch(color='#ffcccb', edgecolor='red', linewidth=2, label='Critical Activity'),
        mpatches.Patch(color='lightblue', label='Non-Critical Activity'),
        plt.Line2D([0], [0], color='#2196F3', linewidth=2, linestyle='solid', label='FS (Finish-to-Start)'),
        plt.Line2D([0], [0], color='#4CAF50', linewidth=2, linestyle='dashed', label='SS (Start-to-Start)'),
        plt.Line2D([0], [0], color='#FF9800', linewidth=2, linestyle='dotted', label='FF (Finish-to-Finish)'),
        plt.Line2D([0], [0], color='#9C27B0', linewidth=2, linestyle='dashdot', label='SF (Start-to-Finish)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

    ax.set_title('Project Network Diagram (PDM - Activity on Node)', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    return fig


def create_gantt_chart(scheduler: PDMScheduler) -> plt.Figure:
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
            bar_color = '#e74c3c'  # Red for critical
            edge_color = '#c0392b'
        else:
            bar_color = '#3498db'  # Blue for non-critical
            edge_color = '#2980b9'

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

    ax.set_xlabel('Time (Days)', fontsize=12)
    ax.set_ylabel('Activities', fontsize=12)
    ax.set_title('Project Gantt Chart', fontsize=14, fontweight='bold')

    # Set x-axis limits
    ax.set_xlim(-0.5, scheduler.project_duration + max(
        (a.total_float or 0) for a in scheduler.activities.values()) + 1)

    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Add legend
    legend_elements = [
        mpatches.Patch(color='#e74c3c', label='Critical Activity'),
        mpatches.Patch(color='#3498db', label='Non-Critical Activity'),
        mpatches.Patch(color='lightgray', label='Total Float'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Add vertical line for project end
    ax.axvline(x=scheduler.project_duration, color='red', linestyle='--', linewidth=2, label='Project End')
    ax.text(scheduler.project_duration, -0.5, f'Day {scheduler.project_duration}',
           ha='center', va='top', color='red', fontweight='bold')

    plt.tight_layout()
    return fig


# =============================================================================
# STREAMLIT APPLICATION
# =============================================================================

def main():
    """Main Streamlit application."""

    st.set_page_config(
        page_title="IPMA CPM/PDM Scheduler",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📊 IPMA ICB 4.0/4.1 CPM/PDM Project Scheduler")
    st.markdown("""
    **Precedence Diagramming Method (PDM) - Activity on Node**

    This application implements the standard CPM/PDM scheduling methodology as described in
    IPMA International Competence Baseline (ICB) 4.0/4.1 for time management competence.
    """)

    # Initialize session state
    if 'scheduler' not in st.session_state:
        st.session_state.scheduler = PDMScheduler()
    if 'calculated' not in st.session_state:
        st.session_state.calculated = False
    # Initialize predecessor list if not present
    if 'pred_list' not in st.session_state:
        st.session_state.pred_list = []
    scheduler = st.session_state.scheduler
    if 'pred_list' not in st.session_state:
        st.session_state.pred_list = []
    # Sidebar for adding activities
    with st.sidebar:
        st.header("Add Activity")

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

        duration = st.number_input(
            "Duration (days)",
            min_value=0,
            value=1,
            step=1,
            key="duration_input"
        )

        st.markdown("**Predecessors** (add multiple if needed)")

        if 'pred_list' not in st.session_state:
            st.session_state.pred_list = []

        if st.button("➕ Add Predecessor", key="add_pred_btn"):
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
                if st.button("🗑", key=f"remove_pred_{idx}"):
                    to_remove.append(idx)

            # Update session state with current widget values (persists on rerun)
            if idx < len(st.session_state.pred_list):
                st.session_state.pred_list[idx]["pred_id"] = selected_id
                st.session_state.pred_list[idx]["rel_type"] = rel_type
                st.session_state.pred_list[idx]["lag"] = lag

            if selected_id != "(none)":
                predecessors.append(Relationship(selected_id, rel_type, int(lag)))

        # Remove selected predecessors
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
                    pred_str
                )
                if success:
                    st.success(f"Activity **{activity_id}** added successfully with {len(predecessors)} predecessor(s)!")
                    st.session_state.pred_list = []  # Clear after add
                    st.session_state.calculated = False
                    st.rerun()
                else:
                    st.error(message)

        st.divider()

        st.header("Predecessor Format Guide")
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

        st.divider()

        # Sample project loader
        st.header("Sample Projects")

        if st.button("Load Sample Project", use_container_width=True):
            scheduler.clear()
            # Add sample activities
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
            st.success("Sample project loaded!")
            st.session_state.calculated = False
            st.rerun()

        if st.button("Load Complex Sample (All Relations)", use_container_width=True):
            scheduler.clear()
            # Add complex sample with all relationship types
            complex_activities = [
                ("A", "Foundation Work", 5, ""),
                ("B", "Parallel Prep Work", 3, "A:SS:2"),  # SS relationship
                ("C", "Main Construction", 10, "A:FS:0;B:FS:0"),
                ("D", "Finishing Work", 4, "C:FF:-2"),  # FF with negative lag (lead)
                ("E", "Inspection", 2, "C:FS:0;D:SF:1"),  # SF relationship
                ("F", "Final Review", 1, "E:FS:0"),
            ]
            for act_id, desc, dur, preds in complex_activities:
                scheduler.add_activity(act_id, desc, dur, preds)
            st.success("Complex sample loaded!")
            st.session_state.calculated = False
            st.rerun()

        if st.button("Clear All Activities", use_container_width=True, type="secondary"):
            scheduler.clear()
            st.session_state.calculated = False
            st.success("All activities cleared!")
            st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Activities")

        if scheduler.activities:
            df = scheduler.get_activities_dataframe()
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Delete activity option
            with st.expander("Remove Activity"):
                act_to_remove = st.selectbox(
                    "Select activity to remove",
                    options=list(scheduler.activities.keys())
                )
                if st.button("Remove Selected Activity"):
                    success, message = scheduler.remove_activity(act_to_remove)
                    if success:
                        st.success(message)
                        st.session_state.calculated = False
                        st.rerun()
                    else:
                        st.error(message)
        else:
            st.info("No activities added yet. Use the sidebar to add activities.")

    with col2:
        st.header("Actions")

        if st.button("🔢 Calculate Critical Path & Floats",
                    use_container_width=True, type="primary",
                    disabled=len(scheduler.activities) == 0):
            success, message = scheduler.calculate()
            if success:
                st.session_state.calculated = True
                st.success(message)
            else:
                st.error(message)

        st.divider()

        if st.session_state.calculated:
            st.metric("Project Duration", f"{scheduler.project_duration} days")
            st.metric("Critical Activities",
                     f"{sum(1 for a in scheduler.activities.values() if a.is_critical)}")

    # Results section
    if st.session_state.calculated and scheduler.activities:
        st.divider()
        st.header("📈 Calculation Results")

        # Results table
        results_df = scheduler.get_results_dataframe()

        # Style the dataframe
        def highlight_critical(row):
            if row['Critical'] == 'Yes':
                return ['background-color: #ffcccb'] * len(row)
            return [''] * len(row)

        styled_df = results_df.style.apply(highlight_critical, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Critical path display
        st.subheader("Critical Path")
        critical_path_str = " → ".join(scheduler.critical_path)
        st.markdown(f"**{critical_path_str}**")

        # Tabs for visualizations and details
        tab1, tab2, tab3 = st.tabs(["📊 Network Diagram", "📅 Gantt Chart", "📝 Calculation Details"])

        with tab1:
            st.subheader("Network Diagram (PDM - Activity on Node)")
            fig = create_network_diagram(scheduler)
            st.pyplot(fig)
            plt.close(fig)

            st.caption("""
            **Legend:** Red nodes = Critical activities | Blue nodes = Non-critical activities
            Edge styles indicate relationship types: Solid=FS, Dashed=SS, Dotted=FF, Dashdot=SF
            """)

        with tab2:
            st.subheader("Gantt Chart")
            fig = create_gantt_chart(scheduler)
            st.pyplot(fig)
            plt.close(fig)

            st.caption("""
            **Legend:** Red bars = Critical activities | Blue bars = Non-critical activities
            Gray extensions show Total Float available for non-critical activities.
            """)

        with tab3:
            st.subheader("Detailed Calculation Log")

            # Display calculation log
            log_text = "\n".join(scheduler.calculation_log)
            st.text_area("Calculation Steps", value=log_text, height=500, disabled=True)

            # IPMA compliance note
            st.info("""
            **IPMA ICB 4.0/4.1 Compliance Note**

            This calculation follows the standard Precedence Diagramming Method (PDM) as recommended
            by IPMA International Competence Baseline for the Time Management competence element.

            Key features implemented:
            - All four precedence relationships (FS, SS, FF, SF) with positive/negative lags
            - Forward pass calculating Early Start (ES) and Early Finish (EF)
            - Backward pass calculating Late Start (LS) and Late Finish (LF)
            - Total Float (TF) = LS - ES = LF - EF
            - Free Float (FF) correctly calculated for each relationship type
            - Critical Path identification (activities where TF = 0)

            **Limitations:**
            - Free Float calculation for complex networks with mixed relationship types
              may require additional constraints in real-world scenarios.
            - This implementation uses discrete integer time units (days).
            """)

    # Footer
    st.divider()
    st.markdown("""
    ---
    **IPMA CPM/PDM Project Scheduler** | Precedence Diagramming Method | Activity-on-Node Format

    *Implements standard CPM/PDM scheduling methodology as per IPMA ICB 4.0/4.1*
    """)


if __name__ == "__main__":
    main()
