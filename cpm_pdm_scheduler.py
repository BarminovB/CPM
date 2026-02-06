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
import re
import base64
import json
import io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, List, Dict, Any

from cpm.engine import PDMScheduler
from cpm.models import Activity, Relationship
from cpm.ui_styles import THEMES, get_active_theme, STATUS_OPTIONS, RISK_OPTIONS, get_theme_css
from cpm.visualizations import (
    create_network_diagram,
    create_gantt_chart,
    create_plotly_gantt,
    fig_to_base64
)
from cpm.ui_components import (
    _is_missing,
    _safe_str,
    _safe_int,
    build_report_html,
    add_dependency_to_scheduler,
    generate_dependency_suggestions,
    analyze_dependency_conflicts,
    compute_dependency_health,
    render_graph_editor,
    render_gantt_editor
)


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

    active_theme = get_active_theme(theme_choice)
    st.markdown(get_theme_css(active_theme), unsafe_allow_html=True)

    st.markdown(
        f"""
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
                <div style="font-size:18px;font-weight:600;color:var(--cpm-ink);">{st.session_state.get('scheduler', PDMScheduler()).project_name}</div>
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
            status_colors = active_theme["status"]
            legend_html = " ".join(
                [
                    f"<span class='cpm-status' style='background:{status_colors[s]};'>{s}</span>"
                    for s in STATUS_OPTIONS
                ]
            )
            st.markdown(legend_html, unsafe_allow_html=True)
            risk_colors = active_theme["risk"]
            risk_html = " ".join(
                [
                    f"<span class='cpm-status' style='background:{risk_colors[r]};'>Risk: {r}</span>"
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
                color = active_theme["status"].get(value, active_theme["surface2"])
                return f"background-color: {color}; color: {active_theme['ink']};"

            def risk_style(value: str) -> str:
                color = active_theme["risk"].get(value, active_theme["surface2"])
                return f"background-color: {color}; color: {active_theme['ink']};"

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
                return [f'background-color: {active_theme["critical_soft"]}'] * len(row)
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
        scheduler_data = scheduler.to_dict()
        gantt_fig = create_plotly_gantt(scheduler_data, theme=active_theme, scale=gantt_scale)
        st.plotly_chart(gantt_fig, use_container_width=True, key="gantt_overview")
        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Report")
        st.markdown("<div class='cpm-glass cpm-fade-in'>", unsafe_allow_html=True)
        if st.button("Generate Report", type="primary", use_container_width=True):
            st.session_state.report_html = build_report_html(
                scheduler,
                results_df,
                active_theme,
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

        st.subheader("Detail Views")
        view_tab = st.radio(
            "Select View",
            options=[
                "Network Diagram",
                "Graph Editor",
                "Calculation Details",
            ],
            horizontal=True,
            key="view_tab_radio"
        )

        st.markdown("<div class='cpm-glass cpm-fade-in'>", unsafe_allow_html=True)
        if view_tab == "Network Diagram":
            st.subheader("Network Diagram (PDM - Activity on Node)")
            fig = create_network_diagram(scheduler_data, theme=active_theme)
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "Legend: Red nodes = Critical activities | Blue nodes = Non-critical activities. "
                "Edge styles: Solid=FS, Dashed=SS, Dotted=FF, Dashdot=SF."
            )

        elif view_tab == "Graph Editor":
            st.subheader("Graph Editor")
            if scheduler.activities:
                st.markdown("<div class='cpm-focus-trail'>", unsafe_allow_html=True)
                render_graph_editor(scheduler, theme=active_theme)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Add activities first to edit dependencies.")

        elif view_tab == "Calculation Details":
            st.subheader("Detailed Calculation Log")
            log_text = "\n".join(scheduler.calculation_log)
            st.text_area("Calculation Steps", value=log_text, height=500, disabled=True)
            st.info(
                "This calculation follows the standard PDM method with all four relationship types, "
                "positive or negative lags, forward/backward passes, floats, and critical path detection."
            )
        st.markdown("</div>", unsafe_allow_html=True)

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
