import re
from typing import Dict, Any, List, Optional
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from .engine import PDMScheduler
from .models import Relationship
from .visualizations import create_network_diagram, create_gantt_chart, create_plotly_gantt, fig_to_base64

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

def build_report_html(scheduler: PDMScheduler, results_df: pd.DataFrame, theme: Dict[str, Any]) -> str:
    """
    Build a comprehensive HTML report for the project.
    """
    scheduler_data = scheduler.to_dict()
    net_fig = create_network_diagram(scheduler_data, theme)
    gantt_fig = create_gantt_chart(scheduler_data, theme)
    
    net_b64 = fig_to_base64(net_fig)
    gantt_b64 = fig_to_base64(gantt_fig)
    
    plt.close(net_fig)
    plt.close(gantt_fig)
    
    rows_html = ""
    for _, row in results_df.iterrows():
        is_crit = row['Critical'] == 'Yes'
        style = f"background-color: {theme['critical_soft']}; font-weight: bold;" if is_crit else ""
        rows_html += f"""
        <tr style="{style}">
            <td>{row['ID']}</td>
            <td>{row['Description']}</td>
            <td>{row['Duration']}</td>
            <td>{row['ES']}</td>
            <td>{row['EF']}</td>
            <td>{row['LS']}</td>
            <td>{row['LF']}</td>
            <td>{row['TF']}</td>
            <td>{row['Critical']}</td>
        </tr>
        """
        
    critical_paths_html = ""
    for idx, path in enumerate(scheduler.critical_paths, 1):
        critical_paths_html += f"<li>{' &rarr; '.join(path)}</li>"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>CPM Project Report - {scheduler.project_name}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: {theme['ink']}; background: {theme['bg']}; line-height: 1.6; }}
            .container {{ max-width: 1000px; margin: 0 auto; padding: 40px; background: white; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
            h1 {{ color: {theme['accent']}; border-bottom: 2px solid {theme['accent']}; padding-bottom: 10px; }}
            h2 {{ color: {theme['accent2']}; margin-top: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
            th {{ background-color: #f8f9fa; }}
            .img-container {{ text-align: center; margin: 30px 0; }}
            .img-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            .summary-box {{ display: flex; gap: 20px; margin: 20px 0; }}
            .summary-item {{ flex: 1; padding: 20px; background: #f8f9fa; border-radius: 8px; text-align: center; }}
            .summary-value {{ font-size: 24px; font-weight: bold; color: {theme['accent']}; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Project Schedule Report: {scheduler.project_name}</h1>
            
            <div class="summary-box">
                <div class="summary-item">
                    <div class="summary-label">Duration</div>
                    <div class="summary-value">{scheduler.project_duration} Days</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Activities</div>
                    <div class="summary-value">{len(scheduler.activities)}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Critical Path(s)</div>
                    <div class="summary-value">{len(scheduler.critical_paths)}</div>
                </div>
            </div>

            <h2>Schedule Table</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID</th><th>Description</th><th>Dur</th><th>ES</th><th>EF</th><th>LS</th><th>LF</th><th>TF</th><th>Crit</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>

            <h2>Critical Path Analysis</h2>
            <ul>{critical_paths_html}</ul>

            <h2>Gantt Chart</h2>
            <div class="img-container">
                <img src="data:image/png;base64,{gantt_b64}" alt="Gantt Chart">
            </div>

            <h2>Network Diagram (PDM)</h2>
            <div class="img-container">
                <img src="data:image/png;base64,{net_b64}" alt="Network Diagram">
            </div>
            
            <p style="font-size: 12px; color: #888; margin-top: 40px; text-align: center;">
                Generated by CPM Studio &bull; {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
            </p>
        </div>
    </body>
    </html>
    """
    return html

def add_dependency_to_scheduler(scheduler: PDMScheduler, pred_id: str, succ_id: str, rel_type: str, lag: int) -> tuple[bool, str]:
    if pred_id == succ_id:
        return False, "An activity cannot be its own predecessor."
    
    # Check if dependency already exists
    act = scheduler.activities.get(succ_id)
    if act:
        for rel in act.predecessors:
            if rel.predecessor_id == pred_id and rel.relation_type == rel_type:
                return False, f"Dependency {pred_id} ({rel_type}) -> {succ_id} already exists."
    
    # Try to add temporarily to check for cycles
    rel_str = f"{pred_id}:{rel_type}:{lag}"
    original_preds = [f"{r.predecessor_id}:{r.relation_type}:{r.lag}" for r in act.predecessors]
    new_preds_str = ";".join(original_preds + [rel_str])
    
    # Save state
    old_preds = act.predecessors
    
    success, message = scheduler.add_activity(
        act.id, act.description, act.duration, new_preds_str,
        act.status, act.owner, act.progress, act.risk
    )
    
    if not success:
        # Restore if failed (e.g. cycle)
        act.predecessors = old_preds
        return False, message
        
    return True, f"Added dependency: {pred_id} {rel_type} (+{lag}) -> {succ_id}"

def generate_dependency_suggestions(scheduler: PDMScheduler, act_id: str, tolerance: int = 2, max_suggestions: int = 6) -> List[Dict[str, Any]]:
    act = scheduler.activities.get(act_id)
    if not act or act.es is None:
        return []

    suggestions = []
    existing = {(r.predecessor_id, r.relation_type, r.lag) for r in act.predecessors}
    critical_ids = {a.id for a in scheduler.activities.values() if a.is_critical}
    act_is_critical = act_id in critical_ids

    for pred_id, pred in scheduler.activities.items():
        if pred_id == act_id:
            continue
        
        # Suggested Finish-to-Start
        if pred.ef is not None and act.es is not None:
            lag = act.es - pred.ef
            if abs(lag) <= tolerance:
                if (pred_id, "FS", lag) not in existing:
                    score = abs(lag)
                    if pred_id in critical_ids or act_is_critical:
                        score -= 0.5
                    suggestions.append({
                        "pred": pred_id, "type": "FS", "lag": lag, "score": score,
                        "reason": "FS aligns successor ES to predecessor EF",
                        "critical_hint": pred_id in critical_ids or act_is_critical,
                    })

        # Suggested Start-to-Start
        if pred.es is not None and act.es is not None:
            lag = act.es - pred.es
            if abs(lag) <= tolerance:
                if (pred_id, "SS", lag) not in existing:
                    score = abs(lag) + 0.1
                    suggestions.append({
                        "pred": pred_id, "type": "SS", "lag": lag, "score": score,
                        "reason": "SS aligns successor ES to predecessor ES",
                        "critical_hint": pred_id in critical_ids or act_is_critical,
                    })

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
                rows.append({
                    "Activity": act.id, "Predecessor": pred.id, "Type": rel.relation_type,
                    "Lag": rel.lag, "Violation": violation, "Constraint": constraint,
                })

    return pd.DataFrame(rows)

def compute_dependency_health(scheduler: PDMScheduler) -> Dict[str, object]:
    total_relations = sum(len(act.predecessors) for act in scheduler.activities.values())
    conflicts = analyze_dependency_conflicts(scheduler)
    conflict_count = len(conflicts)
    negative_ff = sum(1 for act in scheduler.activities.values() if act.free_float is not None and act.free_float < 0)

    if total_relations == 0:
        base_score = 100
    else:
        base_score = max(0, int(100 - (conflict_count / total_relations) * 100))

    if negative_ff:
        base_score = max(0, base_score - min(20, negative_ff * 5))
    if total_relations < max(1, len(scheduler.activities) - 1):
        base_score = max(0, base_score - 10)

    status = "Healthy"
    if base_score < 70: status = "At Risk"
    if base_score < 40: status = "Critical"

    return {
        "score": base_score, "status": status, "total_relations": total_relations,
        "conflicts": conflict_count, "negative_ff": negative_ff,
    }

def render_graph_editor(scheduler: PDMScheduler, theme: Dict[str, Any]) -> None:
    try:
        from streamlit_agraph import agraph, Node, Edge, Config
    except Exception:
        st.warning("streamlit-agraph is not installed.")
        return

    st.caption("Drag nodes to arrange the layout. Click a node to set predecessor, then click another to set successor.")

    if "graph_pred" not in st.session_state: st.session_state.graph_pred = None
    if "graph_succ" not in st.session_state: st.session_state.graph_succ = None

    nodes = []
    for act in scheduler.activities.values():
        # Standard IPMA-style field set
        es_val = act.es if act.es is not None else "-"
        ef_val = act.ef if act.ef is not None else "-"
        ls_val = act.ls if act.ls is not None else "-"
        lf_val = act.lf if act.lf is not None else "-"
        ff_val = act.free_float if act.free_float is not None else "-"
        
        lbl = (
            f"{act.id}\n"
            f"ES:{es_val} EF:{ef_val}\n"
            f"LS:{ls_val} LF:{lf_val}\n"
            f"FF:{ff_val} D:{act.duration}"
        )
        nodes.append(Node(
            id=act.id, 
            label=lbl, 
            size=30, 
            color=theme["critical"] if act.is_critical else theme["noncritical"], 
            shape="box"
        ))
    
    edges = [Edge(source=rel.predecessor_id, target=act.id, label=f"{rel.relation_type}({rel.lag:+d})", type="CURVE_SMOOTH")
             for act in scheduler.activities.values() for rel in act.predecessors]

    config = Config(width=1000, height=600, directed=True, physics=True, nodeHighlightBehavior=True, highlightColor=theme["highlight"])
    selection = agraph(nodes=nodes, edges=edges, config=config)
    
    selected_node = None
    if isinstance(selection, dict):
        selected_node = selection.get("node") or selection.get("selected_node")
    elif isinstance(selection, str):
        selected_node = selection

    if selected_node:
        if st.session_state.graph_pred is None: st.session_state.graph_pred = selected_node
        elif st.session_state.graph_succ is None and selected_node != st.session_state.graph_pred:
            st.session_state.graph_succ = selected_node

    if st.session_state.graph_pred or st.session_state.graph_succ:
        st.info(f"Selected: {st.session_state.graph_pred or '-'} -> {st.session_state.graph_succ or '-'}")

    activity_ids = sorted(scheduler.activities.keys())
    col_a, col_b, col_c, col_d = st.columns([2, 2, 2, 1])
    with col_a:
        pred_index = 0
        if st.session_state.graph_pred in activity_ids: pred_index = 1 + activity_ids.index(st.session_state.graph_pred)
        pred = st.selectbox("Predecessor", options=["(select)"] + activity_ids, index=pred_index)
    with col_b:
        succ_index = 0
        if st.session_state.graph_succ in activity_ids: succ_index = 1 + activity_ids.index(st.session_state.graph_succ)
        succ = st.selectbox("Successor", options=["(select)"] + activity_ids, index=succ_index)
    with col_c:
        rel_type = st.selectbox("Type", options=["FS", "SS", "FF", "SF"])
    with col_d:
        lag = st.number_input("Lag", value=0, step=1)

    if st.button("Add Dependency", type="primary"):
        pred_id = pred if pred != "(select)" else st.session_state.graph_pred
        succ_id = succ if succ != "(select)" else st.session_state.graph_succ
        if not pred_id or not succ_id:
            st.error("Select predecessor and successor.")
        else:
            ok, msg = add_dependency_to_scheduler(scheduler, pred_id, succ_id, rel_type, int(lag))
            if ok:
                st.success(msg)
                st.session_state.calculated = False
                st.session_state.graph_pred = st.session_state.graph_succ = None
                st.rerun()
            else: st.error(msg)

    if st.button("Clear Selection"):
        st.session_state.graph_pred = st.session_state.graph_succ = None
        st.rerun()

def render_gantt_editor(scheduler: PDMScheduler, theme: Dict[str, Any]) -> None:
    try:
        from streamlit_plotly_events import plotly_events
    except Exception:
        st.warning("streamlit-plotly-events is not installed.")
        return

    st.caption("Click a bar to select. Adjust start or duration, then apply.")
    fig = create_plotly_gantt(scheduler.to_dict(), theme, scale=st.session_state.get("gantt_scale", "Day"))
    fig.update_layout(dragmode="select")
    selected_points = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=500, override_width="100%")

    if "gantt_selected_id" not in st.session_state: st.session_state.gantt_selected_id = None
    if selected_points:
        point = selected_points[0]
        custom = point.get("customdata") if isinstance(point, dict) else None
        if custom: st.session_state.gantt_selected_id = custom[0]

    activity_ids = sorted(scheduler.activities.keys())
    selected_id = st.selectbox("Selected activity", options=["(select)"] + activity_ids, 
                              index=0 if st.session_state.gantt_selected_id is None else 1 + activity_ids.index(st.session_state.gantt_selected_id))

    if selected_id == "(select)" and st.session_state.gantt_selected_id: selected_id = st.session_state.gantt_selected_id
    if not selected_id or selected_id == "(select)":
        st.info("Select a task from the Gantt chart to edit.")
        return

    act = scheduler.activities[selected_id]
    current_es = int(act.es or 0)
    current_duration = int(act.duration)

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a: new_start = st.number_input("Start (ES constraint)", value=current_es, step=1)
    with col_b: new_duration = st.number_input("Duration", min_value=0, value=current_duration, step=1)
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
