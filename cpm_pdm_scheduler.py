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
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import re
from io import StringIO


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Relationship:
    """Represents a precedence relationship between two activities."""
    predecessor_id: str
    relation_type: str  # FS, SS, FF, SF
    lag: int  # Can be positive or negative

    def __str__(self):
        lag_str = f"+{self.lag}" if self.lag >= 0 else str(self.lag)
        return f"{self.predecessor_id}:{self.relation_type}:{lag_str}"


@dataclass
class Activity:
    """Represents a project activity with all scheduling attributes."""
    id: str
    description: str
    duration: int
    predecessors: List[Relationship] = field(default_factory=list)

    # Forward pass results
    es: Optional[int] = None  # Early Start
    ef: Optional[int] = None  # Early Finish

    # Backward pass results
    ls: Optional[int] = None  # Late Start
    lf: Optional[int] = None  # Late Finish

    # Float calculations
    total_float: Optional[int] = None  # Total Float (TF)
    free_float: Optional[int] = None   # Free Float (FF)

    # Critical path flag
    is_critical: bool = False

    def reset_calculations(self):
        """Reset all calculated values."""
        self.es = None
        self.ef = None
        self.ls = None
        self.lf = None
        self.total_float = None
        self.free_float = None
        self.is_critical = False


# =============================================================================
# SCHEDULING ENGINE
# =============================================================================

class PDMScheduler:
    """
    Precedence Diagramming Method (PDM) Scheduler.

    Implements the standard CPM/PDM logic as described in IPMA ICB 4.0/4.1
    for time management competence.
    """

    VALID_RELATIONS = {'FS', 'SS', 'FF', 'SF'}

    def __init__(self):
        self.activities: Dict[str, Activity] = {}
        self.calculation_log: List[str] = []
        self.project_duration: int = 0
        self.critical_path: List[str] = []

    def clear(self):
        """Clear all activities and calculations."""
        self.activities.clear()
        self.calculation_log.clear()
        self.project_duration = 0
        self.critical_path = []

    def add_activity(self, activity_id: str, description: str, duration: int,
                     predecessors_str: str = "") -> Tuple[bool, str]:
        """
        Add an activity to the project.

        Args:
            activity_id: Unique identifier for the activity
            description: Activity description
            duration: Duration in days (must be >= 0)
            predecessors_str: Predecessors in format "A:FS:0;B:SS:5;C:FF:-3"

        Returns:
            Tuple of (success, message)
        """
        # Validate activity ID
        activity_id = activity_id.strip().upper()
        if not activity_id:
            return False, "Activity ID cannot be empty."
        if not re.match(r'^[A-Z][A-Z0-9_]*$', activity_id):
            return False, "Activity ID must start with a letter and contain only letters, numbers, and underscores."
        if activity_id in self.activities:
            return False, f"Activity '{activity_id}' already exists."
        if activity_id in ('START', 'FINISH'):
            return False, "Cannot use reserved IDs: START, FINISH."

        # Validate duration
        if duration < 0:
            return False, "Duration must be non-negative."

        # Parse predecessors
        predecessors = []
        if predecessors_str.strip():
            for pred_def in predecessors_str.split(';'):
                pred_def = pred_def.strip()
                if not pred_def:
                    continue

                parts = pred_def.split(':')
                if len(parts) != 3:
                    return False, f"Invalid predecessor format: '{pred_def}'. Use format 'ID:TYPE:LAG' (e.g., 'A:FS:0')."

                pred_id = parts[0].strip().upper()
                rel_type = parts[1].strip().upper()

                try:
                    lag = int(parts[2].strip())
                except ValueError:
                    return False, f"Invalid lag value in '{pred_def}'. Lag must be an integer."

                if rel_type not in self.VALID_RELATIONS:
                    return False, f"Invalid relationship type '{rel_type}'. Must be one of: FS, SS, FF, SF."

                if pred_id == activity_id:
                    return False, "An activity cannot be its own predecessor."

                predecessors.append(Relationship(pred_id, rel_type, lag))

        # Create and store activity
        activity = Activity(
            id=activity_id,
            description=description,
            duration=duration,
            predecessors=predecessors
        )
        self.activities[activity_id] = activity

        return True, f"Activity '{activity_id}' added successfully."

    def remove_activity(self, activity_id: str) -> Tuple[bool, str]:
        """Remove an activity from the project."""
        if activity_id not in self.activities:
            return False, f"Activity '{activity_id}' not found."

        # Check if any other activity depends on this one
        for act in self.activities.values():
            for pred in act.predecessors:
                if pred.predecessor_id == activity_id:
                    return False, f"Cannot remove '{activity_id}': Activity '{act.id}' depends on it."

        del self.activities[activity_id]
        return True, f"Activity '{activity_id}' removed."

    def validate_network(self) -> Tuple[bool, str]:
        """
        Validate the network for calculation readiness.

        Checks for:
        - Missing predecessor references
        - Circular dependencies
        """
        if not self.activities:
            return False, "No activities defined."

        # Check for missing predecessors
        for act in self.activities.values():
            for pred in act.predecessors:
                if pred.predecessor_id not in self.activities:
                    return False, f"Activity '{act.id}' references undefined predecessor '{pred.predecessor_id}'."

        # Check for cycles using DFS
        cycle = self._detect_cycle()
        if cycle:
            return False, f"Circular dependency detected: {' -> '.join(cycle)}"

        return True, "Network is valid."

    def _detect_cycle(self) -> Optional[List[str]]:
        """Detect cycles in the activity network using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {act_id: WHITE for act_id in self.activities}
        parent = {}

        def dfs(node):
            color[node] = GRAY
            for pred in self.activities[node].predecessors:
                pred_id = pred.predecessor_id
                if color[pred_id] == GRAY:
                    # Found a cycle - reconstruct it
                    cycle = [pred_id, node]
                    current = node
                    while current in parent and parent[current] != pred_id:
                        current = parent[current]
                        cycle.insert(1, current)
                    return cycle
                elif color[pred_id] == WHITE:
                    parent[pred_id] = node
                    result = dfs(pred_id)
                    if result:
                        return result
            color[node] = BLACK
            return None

        for act_id in self.activities:
            if color[act_id] == WHITE:
                result = dfs(act_id)
                if result:
                    return result
        return None

    def calculate(self) -> Tuple[bool, str]:
        """
        Perform full CPM/PDM calculation.

        Executes:
        1. Network validation
        2. Forward pass (ES, EF calculation)
        3. Backward pass (LS, LF calculation)
        4. Float calculations (TF, FF)
        5. Critical path identification

        Returns:
            Tuple of (success, message)
        """
        self.calculation_log.clear()
        self._log("=" * 70)
        self._log("IPMA ICB 4.0/4.1 COMPLIANT CPM/PDM CALCULATION")
        self._log("Precedence Diagramming Method (Activity-on-Node)")
        self._log("=" * 70)
        self._log("")

        # Validate network
        valid, msg = self.validate_network()
        if not valid:
            self._log(f"ERROR: {msg}")
            return False, msg

        # Reset all calculations
        for act in self.activities.values():
            act.reset_calculations()

        # Perform calculations
        self._forward_pass()
        self._backward_pass()
        self._calculate_floats()
        self._identify_critical_path()

        self._log("")
        self._log("=" * 70)
        self._log("CALCULATION COMPLETE")
        self._log(f"Project Duration: {self.project_duration} days")
        self._log(f"Critical Path: {' -> '.join(self.critical_path)}")
        self._log("=" * 70)

        return True, "Calculation completed successfully."

    def _log(self, message: str):
        """Add message to calculation log."""
        self.calculation_log.append(message)

    def _get_topological_order(self) -> List[str]:
        """Get activities in topological order (predecessors before successors)."""
        # Build adjacency list (successors)
        successors = defaultdict(list)
        in_degree = {act_id: 0 for act_id in self.activities}

        for act_id, act in self.activities.items():
            for pred in act.predecessors:
                successors[pred.predecessor_id].append(act_id)
                in_degree[act_id] += 1

        # Kahn's algorithm for topological sort
        queue = [act_id for act_id, degree in in_degree.items() if degree == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for succ in successors[node]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        return order

    def _forward_pass(self):
        """
        Forward pass calculation to determine Early Start (ES) and Early Finish (EF).

        For each relationship type:
        - FS (Finish-to-Start): ES_succ >= EF_pred + lag
        - SS (Start-to-Start): ES_succ >= ES_pred + lag
        - FF (Finish-to-Finish): EF_succ >= EF_pred + lag → ES_succ = EF_succ - duration
        - SF (Start-to-Finish): EF_succ >= ES_pred + lag → ES_succ = EF_succ - duration
        """
        self._log("FORWARD PASS (Calculating ES and EF)")
        self._log("-" * 50)

        order = self._get_topological_order()

        for act_id in order:
            act = self.activities[act_id]

            if not act.predecessors:
                # Activity with no predecessors starts at time 0
                act.es = 0
                act.ef = act.duration
                self._log(f"\n{act_id} (no predecessors):")
                self._log(f"  ES = 0 (project start)")
                self._log(f"  EF = ES + Duration = 0 + {act.duration} = {act.ef}")
            else:
                # Calculate ES based on all predecessors
                es_candidates = []
                ef_candidates = []

                self._log(f"\n{act_id} (predecessors: {', '.join(p.predecessor_id for p in act.predecessors)}):")

                for pred_rel in act.predecessors:
                    pred = self.activities[pred_rel.predecessor_id]
                    rel_type = pred_rel.relation_type
                    lag = pred_rel.lag

                    if rel_type == 'FS':
                        # Finish-to-Start: successor can start after predecessor finishes + lag
                        candidate_es = pred.ef + lag
                        es_candidates.append(candidate_es)
                        self._log(f"  From {pred_rel.predecessor_id} (FS, lag={lag}): ES >= EF({pred_rel.predecessor_id}) + lag = {pred.ef} + {lag} = {candidate_es}")

                    elif rel_type == 'SS':
                        # Start-to-Start: successor can start after predecessor starts + lag
                        candidate_es = pred.es + lag
                        es_candidates.append(candidate_es)
                        self._log(f"  From {pred_rel.predecessor_id} (SS, lag={lag}): ES >= ES({pred_rel.predecessor_id}) + lag = {pred.es} + {lag} = {candidate_es}")

                    elif rel_type == 'FF':
                        # Finish-to-Finish: successor finishes after predecessor finishes + lag
                        candidate_ef = pred.ef + lag
                        ef_candidates.append(candidate_ef)
                        self._log(f"  From {pred_rel.predecessor_id} (FF, lag={lag}): EF >= EF({pred_rel.predecessor_id}) + lag = {pred.ef} + {lag} = {candidate_ef}")

                    elif rel_type == 'SF':
                        # Start-to-Finish: successor finishes after predecessor starts + lag
                        candidate_ef = pred.es + lag
                        ef_candidates.append(candidate_ef)
                        self._log(f"  From {pred_rel.predecessor_id} (SF, lag={lag}): EF >= ES({pred_rel.predecessor_id}) + lag = {pred.es} + {lag} = {candidate_ef}")

                # Calculate ES from direct ES constraints
                if es_candidates:
                    es_from_es = max(es_candidates)
                else:
                    es_from_es = 0

                # Calculate ES from EF constraints (FF, SF)
                if ef_candidates:
                    min_ef_from_constraints = max(ef_candidates)
                    es_from_ef = min_ef_from_constraints - act.duration
                else:
                    es_from_ef = 0

                # Final ES is the maximum of all constraints
                act.es = max(0, es_from_es, es_from_ef)

                # EF must satisfy both duration and any direct EF constraints
                ef_from_duration = act.es + act.duration
                if ef_candidates:
                    act.ef = max(ef_from_duration, max(ef_candidates))
                    # Adjust ES if EF was pushed by constraints
                    if act.ef > ef_from_duration:
                        act.es = act.ef - act.duration
                else:
                    act.ef = ef_from_duration

                # Ensure non-negative
                act.es = max(0, act.es)
                act.ef = max(act.es + act.duration, act.ef)

                self._log(f"  → ES = max(0, all constraints) = {act.es}")
                self._log(f"  → EF = ES + Duration = {act.es} + {act.duration} = {act.ef}")

        # Project duration is the maximum EF
        self.project_duration = max(act.ef for act in self.activities.values())
        self._log(f"\nProject Duration = max(all EF values) = {self.project_duration} days")

    def _backward_pass(self):
        """
        Backward pass calculation to determine Late Start (LS) and Late Finish (LF).

        For each relationship type (working backwards):
        - FS: LF_pred <= LS_succ - lag
        - SS: LS_pred <= LS_succ - lag
        - FF: LF_pred <= LF_succ - lag
        - SF: LS_pred <= LF_succ - lag
        """
        self._log("\n\nBACKWARD PASS (Calculating LS and LF)")
        self._log("-" * 50)

        # Build successor relationships for backward pass
        successors = defaultdict(list)
        for act_id, act in self.activities.items():
            for pred_rel in act.predecessors:
                successors[pred_rel.predecessor_id].append(
                    (act_id, pred_rel.relation_type, pred_rel.lag)
                )

        # Process in reverse topological order
        order = self._get_topological_order()
        order.reverse()

        for act_id in order:
            act = self.activities[act_id]
            succ_list = successors.get(act_id, [])

            if not succ_list:
                # Activity with no successors - LF equals project duration
                act.lf = self.project_duration
                act.ls = act.lf - act.duration
                self._log(f"\n{act_id} (no successors):")
                self._log(f"  LF = Project Duration = {self.project_duration}")
                self._log(f"  LS = LF - Duration = {act.lf} - {act.duration} = {act.ls}")
            else:
                # Calculate LF/LS based on all successors
                lf_candidates = []
                ls_candidates = []

                self._log(f"\n{act_id} (successors: {', '.join(s[0] for s in succ_list)}):")

                for succ_id, rel_type, lag in succ_list:
                    succ = self.activities[succ_id]

                    if rel_type == 'FS':
                        # FS: predecessor must finish before successor starts - lag
                        candidate_lf = succ.ls - lag
                        lf_candidates.append(candidate_lf)
                        self._log(f"  To {succ_id} (FS, lag={lag}): LF <= LS({succ_id}) - lag = {succ.ls} - {lag} = {candidate_lf}")

                    elif rel_type == 'SS':
                        # SS: predecessor must start before successor starts - lag
                        candidate_ls = succ.ls - lag
                        ls_candidates.append(candidate_ls)
                        self._log(f"  To {succ_id} (SS, lag={lag}): LS <= LS({succ_id}) - lag = {succ.ls} - {lag} = {candidate_ls}")

                    elif rel_type == 'FF':
                        # FF: predecessor must finish before successor finishes - lag
                        candidate_lf = succ.lf - lag
                        lf_candidates.append(candidate_lf)
                        self._log(f"  To {succ_id} (FF, lag={lag}): LF <= LF({succ_id}) - lag = {succ.lf} - {lag} = {candidate_lf}")

                    elif rel_type == 'SF':
                        # SF: predecessor must start before successor finishes - lag
                        candidate_ls = succ.lf - lag
                        ls_candidates.append(candidate_ls)
                        self._log(f"  To {succ_id} (SF, lag={lag}): LS <= LF({succ_id}) - lag = {succ.lf} - {lag} = {candidate_ls}")

                # Calculate LF from direct LF constraints
                if lf_candidates:
                    lf_from_lf = min(lf_candidates)
                else:
                    lf_from_lf = self.project_duration

                # Calculate LF from LS constraints
                if ls_candidates:
                    min_ls = min(ls_candidates)
                    lf_from_ls = min_ls + act.duration
                else:
                    lf_from_ls = self.project_duration

                # Final LF is the minimum of all constraints
                act.lf = min(lf_from_lf, lf_from_ls)
                act.ls = act.lf - act.duration

                # Handle direct LS constraints (from SS, SF relationships)
                if ls_candidates:
                    direct_ls = min(ls_candidates)
                    if direct_ls < act.ls:
                        act.ls = direct_ls
                        act.lf = act.ls + act.duration

                # Ensure non-negative
                act.ls = max(0, act.ls)
                act.lf = max(act.ls + act.duration, act.lf)

                self._log(f"  → LF = min(all LF constraints) = {act.lf}")
                self._log(f"  → LS = LF - Duration = {act.lf} - {act.duration} = {act.ls}")

    def _calculate_floats(self):
        """
        Calculate Total Float (TF) and Free Float (FF) for all activities.

        Total Float (TF) = LS - ES = LF - EF
        Free Float (FF) = minimum slack to any immediate successor

        For Free Float calculation by relationship type:
        - FS: FF = ES_succ - EF_pred - lag
        - SS: FF = ES_succ - ES_pred - lag
        - FF: FF = EF_succ - EF_pred - lag
        - SF: FF = EF_succ - ES_pred - lag
        """
        self._log("\n\nFLOAT CALCULATIONS")
        self._log("-" * 50)

        # Build successor relationships
        successors = defaultdict(list)
        for act_id, act in self.activities.items():
            for pred_rel in act.predecessors:
                successors[pred_rel.predecessor_id].append(
                    (act_id, pred_rel.relation_type, pred_rel.lag)
                )

        for act_id, act in self.activities.items():
            # Total Float
            act.total_float = act.ls - act.es

            self._log(f"\n{act_id}:")
            self._log(f"  Total Float (TF) = LS - ES = {act.ls} - {act.es} = {act.total_float}")

            # Free Float calculation
            succ_list = successors.get(act_id, [])

            if not succ_list:
                # No successors - Free Float equals Total Float
                act.free_float = self.project_duration - act.ef
                self._log(f"  Free Float (FF) = Project Duration - EF = {self.project_duration} - {act.ef} = {act.free_float}")
                self._log(f"  (No successors - FF is time until project end)")
            else:
                ff_candidates = []

                for succ_id, rel_type, lag in succ_list:
                    succ = self.activities[succ_id]

                    if rel_type == 'FS':
                        # FS: FF = ES_succ - EF_pred - lag
                        ff = succ.es - act.ef - lag
                        ff_candidates.append(ff)
                        self._log(f"  To {succ_id} (FS): FF = ES({succ_id}) - EF({act_id}) - lag = {succ.es} - {act.ef} - {lag} = {ff}")

                    elif rel_type == 'SS':
                        # SS: FF = ES_succ - ES_pred - lag
                        ff = succ.es - act.es - lag
                        ff_candidates.append(ff)
                        self._log(f"  To {succ_id} (SS): FF = ES({succ_id}) - ES({act_id}) - lag = {succ.es} - {act.es} - {lag} = {ff}")

                    elif rel_type == 'FF':
                        # FF: FF = EF_succ - EF_pred - lag
                        ff = succ.ef - act.ef - lag
                        ff_candidates.append(ff)
                        self._log(f"  To {succ_id} (FF): FF = EF({succ_id}) - EF({act_id}) - lag = {succ.ef} - {act.ef} - {lag} = {ff}")

                    elif rel_type == 'SF':
                        # SF: FF = EF_succ - ES_pred - lag
                        ff = succ.ef - act.es - lag
                        ff_candidates.append(ff)
                        self._log(f"  To {succ_id} (SF): FF = EF({succ_id}) - ES({act_id}) - lag = {succ.ef} - {act.es} - {lag} = {ff}")

                act.free_float = max(0, min(ff_candidates)) if ff_candidates else 0
                self._log(f"  → Free Float (FF) = min(all successor constraints) = {act.free_float}")

    def _identify_critical_path(self):
        """
        Identify the critical path (activities with TF = 0).

        The critical path is the longest path through the network and
        determines the minimum project duration.
        """
        self._log("\n\nCRITICAL PATH IDENTIFICATION")
        self._log("-" * 50)

        # Mark critical activities
        critical_activities = []
        for act_id, act in self.activities.items():
            if act.total_float == 0:
                act.is_critical = True
                critical_activities.append(act_id)
                self._log(f"{act_id}: TF = {act.total_float} → CRITICAL")
            else:
                self._log(f"{act_id}: TF = {act.total_float} → Not critical")

        # Build critical path sequence
        self.critical_path = self._build_critical_path_sequence(critical_activities)

        self._log(f"\nCritical Path Sequence: {' -> '.join(self.critical_path)}")
        self._log("\nNote: Multiple parallel critical paths may exist if shown activities")
        self._log("form separate chains. All shown activities have zero Total Float.")

    def _build_critical_path_sequence(self, critical_activities: List[str]) -> List[str]:
        """Build a sequential representation of the critical path."""
        if not critical_activities:
            return []

        # Build successor relationships among critical activities
        critical_set = set(critical_activities)
        successors = defaultdict(list)
        has_predecessor = set()

        for act_id in critical_activities:
            act = self.activities[act_id]
            for pred_rel in act.predecessors:
                if pred_rel.predecessor_id in critical_set:
                    successors[pred_rel.predecessor_id].append(act_id)
                    has_predecessor.add(act_id)

        # Find starting activities (no critical predecessors)
        start_activities = [a for a in critical_activities if a not in has_predecessor]

        # Build path using DFS
        path = []
        visited = set()

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            path.append(node)
            for succ in sorted(successors.get(node, []),
                             key=lambda x: self.activities[x].es):
                dfs(succ)

        # Start from activities with earliest ES
        start_activities.sort(key=lambda x: self.activities[x].es)
        for start in start_activities:
            dfs(start)

        return path

    def get_results_dataframe(self) -> pd.DataFrame:
        """Get calculation results as a pandas DataFrame."""
        data = []
        for act_id in sorted(self.activities.keys()):
            act = self.activities[act_id]
            data.append({
                'ID': act_id,
                'Description': act.description,
                'Duration': act.duration,
                'ES': act.es if act.es is not None else '-',
                'EF': act.ef if act.ef is not None else '-',
                'LS': act.ls if act.ls is not None else '-',
                'LF': act.lf if act.lf is not None else '-',
                'TF': act.total_float if act.total_float is not None else '-',
                'FF': act.free_float if act.free_float is not None else '-',
                'Critical': 'Yes' if act.is_critical else 'No'
            })
        return pd.DataFrame(data)

    def get_activities_dataframe(self) -> pd.DataFrame:
        """Get activities list as a pandas DataFrame."""
        data = []
        for act_id in sorted(self.activities.keys()):
            act = self.activities[act_id]
            pred_str = ';'.join(str(p) for p in act.predecessors) if act.predecessors else '-'
            data.append({
                'ID': act_id,
                'Description': act.description,
                'Duration': act.duration,
                'Predecessors': pred_str
            })
        return pd.DataFrame(data)


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

    # Sidebar for adding activities
    with st.sidebar:
        st.header("Add Activity")

        with st.form("add_activity_form", clear_on_submit=False):
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

            # Dynamic predecessor list
            if 'pred_list' not in st.session_state:
                st.session_state.pred_list = []

            # Button to add new predecessor row
            if st.button("➕ Add Predecessor", key="add_pred_btn"):
                st.session_state.pred_list.append({"pred_id": "", "rel_type": "FS", "lag": 0})
                st.rerun()

            predecessors = []
            to_remove = []

            existing_activities = ["(none)"] + sorted([act_id for act_id in scheduler.activities.keys() if act_id != activity_id])

            for idx, pred in enumerate(st.session_state.pred_list):
                cols = st.columns([3, 2, 2, 1])
                with cols[0]:
                    selected_id = st.selectbox(
                        f"Predecessor #{idx+1}",
                        options=existing_activities,
                        index=existing_activities.index(pred["pred_id"]) if pred["pred_id"] in existing_activities else 0,
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

                if selected_id != "(none)":
                    predecessors.append(Relationship(selected_id, rel_type, int(lag)))

            # Remove selected predecessors
            for idx in sorted(to_remove, reverse=True):
                del st.session_state.pred_list[idx]

            submitted = st.form_submit_button("Add Activity", type="primary", use_container_width=True)

            if submitted:
                if not activity_id:
                    st.error("Activity ID is required.")
                elif not re.match(r'^[A-Z][A-Z0-9_]*$', activity_id):
                    st.error("Invalid Activity ID format.")
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
                        st.session_state.pred_list = []  # Clear the list
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
