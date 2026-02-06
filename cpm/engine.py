from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import re

import pandas as pd

from .models import Activity, Relationship


class PDMScheduler:
    """
    Precedence Diagramming Method (PDM) Scheduler.

    Implements standard CPM/PDM logic with all four relationship types
    and positive/negative lags.
    """

    VALID_RELATIONS = {"FS", "SS", "FF", "SF"}

    def __init__(self, normalize_to_zero: bool = True, project_start: int = 0):
        self.activities: Dict[str, Activity] = {}
        self.calculation_log: List[str] = []
        self.project_duration: int = 0
        self.critical_path: List[str] = []
        self.critical_paths: List[List[str]] = []
        self.normalize_to_zero = normalize_to_zero
        self.project_start = project_start

    def clear(self) -> None:
        """Clear all activities and calculations."""
        self.activities.clear()
        self.calculation_log.clear()
        self.project_duration = 0
        self.critical_path = []
        self.critical_paths = []

    def add_activity(
        self,
        activity_id: str,
        description: str,
        duration: int,
        predecessors_str: str = "",
        status: str = "Not Started",
        owner: str = "Unassigned",
        progress: int = 0,
        risk: str = "Low",
    ) -> Tuple[bool, str]:
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
        activity_id = activity_id.strip().upper()
        if not activity_id:
            return False, "Activity ID cannot be empty."
        if not re.match(r"^[A-Z][A-Z0-9_]*$", activity_id):
            return (
                False,
                "Activity ID must start with a letter and contain only letters, numbers, and underscores.",
            )
        if activity_id in self.activities:
            return False, f"Activity '{activity_id}' already exists."
        if activity_id in ("START", "FINISH"):
            return False, "Cannot use reserved IDs: START, FINISH."

        if duration < 0:
            return False, "Duration must be non-negative."

        predecessors, parse_error = self._parse_predecessors(predecessors_str, activity_id)
        if parse_error:
            return False, parse_error

        status = status.strip() if status else "Not Started"
        owner = owner.strip() if owner else "Unassigned"
        try:
            progress_value = int(progress)
        except (TypeError, ValueError):
            progress_value = 0
        progress_value = max(0, min(100, progress_value))
        risk = risk.strip() if risk else "Low"
        activity = Activity(
            id=activity_id,
            description=description,
            duration=duration,
            status=status,
            owner=owner,
            progress=progress_value,
            risk=risk,
            predecessors=predecessors,
        )
        self.activities[activity_id] = activity

        return True, f"Activity '{activity_id}' added successfully."

    def _parse_predecessors(
        self, predecessors_str: str, activity_id: str
    ) -> Tuple[List[Relationship], Optional[str]]:
        predecessors: List[Relationship] = []
        if not predecessors_str or not predecessors_str.strip():
            return predecessors, None

        seen: set[tuple[str, str, int]] = set()
        for pred_def in re.split(r"[;,]", predecessors_str):
            pred_def = pred_def.strip()
            if not pred_def or pred_def in {"-", "—"}:
                continue

            parts = [p.strip() for p in pred_def.split(":")]
            if len(parts) != 3:
                return (
                    [],
                    f"Invalid predecessor format: '{pred_def}'. Use format 'ID:TYPE:LAG' (e.g., 'A:FS:0').",
                )

            pred_id = parts[0].upper()
            rel_type = parts[1].upper()
            lag_raw = parts[2]

            if not pred_id or not re.match(r"^[A-Z][A-Z0-9_]*$", pred_id):
                return (
                    [],
                    f"Invalid predecessor ID '{pred_id}'. Use letters/numbers/underscores and start with a letter.",
                )

            try:
                lag = int(lag_raw)
            except ValueError:
                return [], f"Invalid lag value in '{pred_def}'. Lag must be an integer."

            if rel_type not in self.VALID_RELATIONS:
                return (
                    [],
                    f"Invalid relationship type '{rel_type}'. Must be one of: FS, SS, FF, SF.",
                )

            if pred_id == activity_id:
                return [], "An activity cannot be its own predecessor."

            key = (pred_id, rel_type, lag)
            if key in seen:
                continue
            seen.add(key)
            predecessors.append(Relationship(pred_id, rel_type, lag))

        return predecessors, None

    def remove_activity(self, activity_id: str) -> Tuple[bool, str]:
        """Remove an activity from the project."""
        if activity_id not in self.activities:
            return False, f"Activity '{activity_id}' not found."

        for act in self.activities.values():
            for pred in act.predecessors:
                if pred.predecessor_id == activity_id:
                    return (
                        False,
                        f"Cannot remove '{activity_id}': Activity '{act.id}' depends on it.",
                    )

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

        for act in self.activities.values():
            for pred in act.predecessors:
                if pred.predecessor_id not in self.activities:
                    return (
                        False,
                        f"Activity '{act.id}' references undefined predecessor '{pred.predecessor_id}'.",
                    )

        cycle = self._detect_cycle()
        if cycle:
            return False, f"Circular dependency detected: {' -> '.join(cycle)}"

        return True, "Network is valid."

    def _detect_cycle(self) -> Optional[List[str]]:
        """Detect cycles in the activity network using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {act_id: WHITE for act_id in self.activities}
        parent: Dict[str, str] = {}

        def dfs(node: str) -> Optional[List[str]]:
            color[node] = GRAY
            for pred in self.activities[node].predecessors:
                pred_id = pred.predecessor_id
                if color[pred_id] == GRAY:
                    cycle = [pred_id, node]
                    current = node
                    while current in parent and parent[current] != pred_id:
                        current = parent[current]
                        cycle.insert(1, current)
                    return cycle
                if color[pred_id] == WHITE:
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
        """
        self.calculation_log.clear()
        self._log("=" * 70)
        self._log("CPM/PDM CALCULATION")
        self._log("Precedence Diagramming Method (Activity-on-Node)")
        self._log("=" * 70)
        self._log("")

        valid, msg = self.validate_network()
        if not valid:
            self._log(f"ERROR: {msg}")
            return False, msg

        for act in self.activities.values():
            act.reset_calculations()

        self._forward_pass()
        self.project_duration = max(act.ef for act in self.activities.values() if act.ef is not None)
        self._backward_pass()
        self._calculate_floats()
        self._identify_critical_paths()
        self._normalize_schedule_if_needed()

        self._log("")
        self._log("=" * 70)
        self._log("CALCULATION COMPLETE")
        self._log(f"Project Duration: {self.project_duration} days")
        if self.critical_paths:
            self._log(f"Critical Paths: {len(self.critical_paths)}")
            for idx, path in enumerate(self.critical_paths, start=1):
                self._log(f"  {idx}. {' -> '.join(path)}")
        else:
            self._log("Critical Path: (none)")
        self._log("=" * 70)

        return True, "Calculation completed successfully."

    def _log(self, message: str) -> None:
        self.calculation_log.append(message)

    def _get_topological_order(self) -> List[str]:
        """Get activities in topological order (predecessors before successors)."""
        successors = defaultdict(list)
        in_degree = {act_id: 0 for act_id in self.activities}

        for act_id, act in self.activities.items():
            for pred in act.predecessors:
                successors[pred.predecessor_id].append(act_id)
                in_degree[act_id] += 1

        queue = deque([act_id for act_id, degree in in_degree.items() if degree == 0])
        order: List[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for succ in successors[node]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        return order

    def _forward_pass(self) -> None:
        """
        Forward pass calculation to determine Early Start (ES) and Early Finish (EF).
        """
        self._log("FORWARD PASS (Calculating ES and EF)")
        self._log("-" * 50)

        order = self._get_topological_order()

        for act_id in order:
            act = self.activities[act_id]

            if not act.predecessors:
                act.es = self.project_start
                act.ef = act.es + act.duration
                self._log(f"\n{act_id} (no predecessors):")
                self._log(f"  ES = Project Start = {act.es}")
                self._log(f"  EF = ES + Duration = {act.es} + {act.duration} = {act.ef}")
                continue

            es_candidates: List[int] = []
            ef_candidates: List[int] = []

            self._log(
                f"\n{act_id} (predecessors: {', '.join(p.predecessor_id for p in act.predecessors)}):"
            )

            for pred_rel in act.predecessors:
                pred = self.activities[pred_rel.predecessor_id]
                rel_type = pred_rel.relation_type
                lag = pred_rel.lag

                if rel_type == "FS":
                    candidate_es = pred.ef + lag
                    es_candidates.append(candidate_es)
                    self._log(
                        f"  From {pred_rel.predecessor_id} (FS, lag={lag}): "
                        f"ES >= EF({pred_rel.predecessor_id}) + lag = {pred.ef} + {lag} = {candidate_es}"
                    )
                elif rel_type == "SS":
                    candidate_es = pred.es + lag
                    es_candidates.append(candidate_es)
                    self._log(
                        f"  From {pred_rel.predecessor_id} (SS, lag={lag}): "
                        f"ES >= ES({pred_rel.predecessor_id}) + lag = {pred.es} + {lag} = {candidate_es}"
                    )
                elif rel_type == "FF":
                    candidate_ef = pred.ef + lag
                    ef_candidates.append(candidate_ef)
                    self._log(
                        f"  From {pred_rel.predecessor_id} (FF, lag={lag}): "
                        f"EF >= EF({pred_rel.predecessor_id}) + lag = {pred.ef} + {lag} = {candidate_ef}"
                    )
                elif rel_type == "SF":
                    candidate_ef = pred.es + lag
                    ef_candidates.append(candidate_ef)
                    self._log(
                        f"  From {pred_rel.predecessor_id} (SF, lag={lag}): "
                        f"EF >= ES({pred_rel.predecessor_id}) + lag = {pred.es} + {lag} = {candidate_ef}"
                    )

            es_from_es = max(es_candidates) if es_candidates else None
            es_from_ef = max(ef_candidates) - act.duration if ef_candidates else None
            constraint_es = act.constraint_es
            candidate_values = [
                v for v in (es_from_es, es_from_ef, constraint_es) if v is not None
            ]

            if not candidate_values:
                act.es = self.project_start
            else:
                act.es = max(candidate_values)

            act.ef = act.es + act.duration

            if ef_candidates:
                min_ef_from_constraints = max(ef_candidates)
                if act.ef < min_ef_from_constraints:
                    act.ef = min_ef_from_constraints
                    act.es = act.ef - act.duration

            if act.constraint_es is not None and act.es < act.constraint_es:
                act.es = act.constraint_es
                act.ef = act.es + act.duration

            self._log(f"  -> ES = {act.es}")
            self._log(f"  -> EF = ES + Duration = {act.es} + {act.duration} = {act.ef}")

        self._log(f"\nProject Finish = max(all EF values) = {max(act.ef for act in self.activities.values())}")

    def _backward_pass(self) -> None:
        """
        Backward pass calculation to determine Late Start (LS) and Late Finish (LF).
        """
        self._log("\n\nBACKWARD PASS (Calculating LS and LF)")
        self._log("-" * 50)

        successors = defaultdict(list)
        for act_id, act in self.activities.items():
            for pred_rel in act.predecessors:
                successors[pred_rel.predecessor_id].append(
                    (act_id, pred_rel.relation_type, pred_rel.lag)
                )

        order = self._get_topological_order()
        order.reverse()

        for act_id in order:
            act = self.activities[act_id]
            succ_list = successors.get(act_id, [])

            if not succ_list:
                act.lf = self.project_duration
                act.ls = act.lf - act.duration
                self._log(f"\n{act_id} (no successors):")
                self._log(f"  LF = Project Finish = {self.project_duration}")
                self._log(f"  LS = LF - Duration = {act.lf} - {act.duration} = {act.ls}")
                continue

            lf_candidates: List[int] = []
            ls_candidates: List[int] = []

            self._log(f"\n{act_id} (successors: {', '.join(s[0] for s in succ_list)}):")

            for succ_id, rel_type, lag in succ_list:
                succ = self.activities[succ_id]
                if rel_type == "FS":
                    candidate_lf = succ.ls - lag
                    lf_candidates.append(candidate_lf)
                    self._log(
                        f"  To {succ_id} (FS, lag={lag}): "
                        f"LF <= LS({succ_id}) - lag = {succ.ls} - {lag} = {candidate_lf}"
                    )
                elif rel_type == "SS":
                    candidate_ls = succ.ls - lag
                    ls_candidates.append(candidate_ls)
                    self._log(
                        f"  To {succ_id} (SS, lag={lag}): "
                        f"LS <= LS({succ_id}) - lag = {succ.ls} - {lag} = {candidate_ls}"
                    )
                elif rel_type == "FF":
                    candidate_lf = succ.lf - lag
                    lf_candidates.append(candidate_lf)
                    self._log(
                        f"  To {succ_id} (FF, lag={lag}): "
                        f"LF <= LF({succ_id}) - lag = {succ.lf} - {lag} = {candidate_lf}"
                    )
                elif rel_type == "SF":
                    candidate_ls = succ.lf - lag
                    ls_candidates.append(candidate_ls)
                    self._log(
                        f"  To {succ_id} (SF, lag={lag}): "
                        f"LS <= LF({succ_id}) - lag = {succ.lf} - {lag} = {candidate_ls}"
                    )

            lf_ub = min(lf_candidates) if lf_candidates else self.project_duration
            lf_ub = min(lf_ub, self.project_duration)
            ls_ub = min(ls_candidates) if ls_candidates else float("inf")

            act.ls = min(ls_ub, lf_ub - act.duration)
            act.lf = act.ls + act.duration

            self._log(f"  -> LF upper bound = {lf_ub}")
            if ls_candidates:
                self._log(f"  -> LS upper bound = {ls_ub}")
            self._log(f"  -> LS = {act.ls}")
            self._log(f"  -> LF = LS + Duration = {act.ls} + {act.duration} = {act.lf}")

    def _calculate_floats(self) -> None:
        """
        Calculate Total Float (TF) and Free Float (FF) for all activities.
        """
        self._log("\n\nFLOAT CALCULATIONS")
        self._log("-" * 50)

        successors = defaultdict(list)
        for act_id, act in self.activities.items():
            for pred_rel in act.predecessors:
                successors[pred_rel.predecessor_id].append(
                    (act_id, pred_rel.relation_type, pred_rel.lag)
                )

        for act_id, act in self.activities.items():
            act.total_float = act.ls - act.es

            self._log(f"\n{act_id}:")
            self._log(f"  Total Float (TF) = LS - ES = {act.ls} - {act.es} = {act.total_float}")

            succ_list = successors.get(act_id, [])
            if not succ_list:
                act.free_float = self.project_duration - act.ef
                self._log(
                    f"  Free Float (FF) = Project Finish - EF = {self.project_duration} - {act.ef} = {act.free_float}"
                )
                self._log("  (No successors - FF is time until project end)")
                continue

            ff_candidates: List[int] = []
            for succ_id, rel_type, lag in succ_list:
                succ = self.activities[succ_id]
                if rel_type == "FS":
                    ff = succ.es - act.ef - lag
                    ff_candidates.append(ff)
                    self._log(
                        f"  To {succ_id} (FS): FF = ES({succ_id}) - EF({act_id}) - lag = "
                        f"{succ.es} - {act.ef} - {lag} = {ff}"
                    )
                elif rel_type == "SS":
                    ff = succ.es - act.es - lag
                    ff_candidates.append(ff)
                    self._log(
                        f"  To {succ_id} (SS): FF = ES({succ_id}) - ES({act_id}) - lag = "
                        f"{succ.es} - {act.es} - {lag} = {ff}"
                    )
                elif rel_type == "FF":
                    ff = succ.ef - act.ef - lag
                    ff_candidates.append(ff)
                    self._log(
                        f"  To {succ_id} (FF): FF = EF({succ_id}) - EF({act_id}) - lag = "
                        f"{succ.ef} - {act.ef} - {lag} = {ff}"
                    )
                elif rel_type == "SF":
                    ff = succ.ef - act.es - lag
                    ff_candidates.append(ff)
                    self._log(
                        f"  To {succ_id} (SF): FF = EF({succ_id}) - ES({act_id}) - lag = "
                        f"{succ.ef} - {act.es} - {lag} = {ff}"
                    )

            act.free_float = min(ff_candidates) if ff_candidates else 0
            self._log(f"  -> Free Float (FF) = min(all successor constraints) = {act.free_float}")

    def _identify_critical_paths(self) -> None:
        """Identify critical activities and build critical path sequences."""
        self._log("\n\nCRITICAL PATH IDENTIFICATION")
        self._log("-" * 50)

        critical_set = set()
        for act_id, act in self.activities.items():
            if act.total_float == 0:
                act.is_critical = True
                critical_set.add(act_id)
                self._log(f"{act_id}: TF = {act.total_float} -> CRITICAL")
            else:
                act.is_critical = False
                self._log(f"{act_id}: TF = {act.total_float} -> Not critical")

        self.critical_paths = self._build_critical_paths(critical_set)
        self.critical_path = self.critical_paths[0] if self.critical_paths else []

        if self.critical_paths:
            self._log(f"\nCritical Paths Found: {len(self.critical_paths)}")
            for idx, path in enumerate(self.critical_paths, start=1):
                self._log(f"  {idx}. {' -> '.join(path)}")
        else:
            self._log("\nCritical Path: (none)")

    def _build_critical_paths(self, critical_set: set[str]) -> List[List[str]]:
        """Build sequential representations of all critical paths."""
        if not critical_set:
            return []

        successors: Dict[str, List[str]] = defaultdict(list)
        incoming: Dict[str, int] = defaultdict(int)

        for succ_id, succ in self.activities.items():
            if succ_id not in critical_set:
                continue
            for rel in succ.predecessors:
                pred_id = rel.predecessor_id
                if pred_id not in critical_set:
                    continue
                pred = self.activities[pred_id]
                if self._is_critical_link(pred, succ, rel):
                    successors[pred_id].append(succ_id)
                    incoming[succ_id] += 1

        for pred_id in successors:
            successors[pred_id] = sorted(
                set(successors[pred_id]),
                key=lambda x: (self.activities[x].es, x),
            )

        start_nodes = [nid for nid in critical_set if incoming[nid] == 0]
        start_nodes.sort(key=lambda x: (self.activities[x].es, x))

        paths: List[List[str]] = []

        def dfs(node: str, path: List[str]) -> None:
            new_path = path + [node]
            if node not in successors or not successors[node]:
                paths.append(new_path)
                return
            for succ in successors[node]:
                dfs(succ, new_path)

        for start in start_nodes:
            dfs(start, [])

        return paths

    def _is_critical_link(self, pred: Activity, succ: Activity, rel: Relationship) -> bool:
        if rel.relation_type == "FS":
            return succ.es == pred.ef + rel.lag
        if rel.relation_type == "SS":
            return succ.es == pred.es + rel.lag
        if rel.relation_type == "FF":
            return succ.ef == pred.ef + rel.lag
        if rel.relation_type == "SF":
            return succ.ef == pred.es + rel.lag
        return False

    def _normalize_schedule_if_needed(self) -> None:
        if not self.normalize_to_zero:
            return
        min_es = min(act.es for act in self.activities.values() if act.es is not None)
        if min_es >= self.project_start:
            return

        offset = self.project_start - min_es
        for act in self.activities.values():
            if act.es is not None:
                act.es += offset
            if act.ef is not None:
                act.ef += offset
            if act.ls is not None:
                act.ls += offset
            if act.lf is not None:
                act.lf += offset
        self.project_duration += offset
        self._log("")
        self._log(
            f"Schedule normalized by +{offset} to align earliest ES to Project Start ({self.project_start})."
        )

    def get_results_dataframe(self) -> pd.DataFrame:
        """Get calculation results as a pandas DataFrame."""
        data = []
        for act_id in sorted(self.activities.keys()):
            act = self.activities[act_id]
            data.append(
                {
                    "ID": act_id,
                    "Description": act.description,
                    "Status": act.status,
                    "Owner": act.owner,
                    "Progress": act.progress,
                    "Risk": act.risk,
                    "Duration": act.duration,
                    "ES": act.es if act.es is not None else "-",
                    "EF": act.ef if act.ef is not None else "-",
                    "LS": act.ls if act.ls is not None else "-",
                    "LF": act.lf if act.lf is not None else "-",
                    "TF": act.total_float if act.total_float is not None else "-",
                    "FF": act.free_float if act.free_float is not None else "-",
                    "Constraint ES": act.constraint_es if act.constraint_es is not None else "-",
                    "Critical": "Yes" if act.is_critical else "No",
                }
            )
        return pd.DataFrame(data)

    def get_activities_dataframe(self) -> pd.DataFrame:
        """Get activities list as a pandas DataFrame."""
        data = []
        for act_id in sorted(self.activities.keys()):
            act = self.activities[act_id]
            pred_str = ";".join(str(p) for p in act.predecessors) if act.predecessors else ""
            data.append(
                {
                    "ID": act_id,
                    "Description": act.description,
                    "Status": act.status,
                    "Owner": act.owner,
                    "Progress": act.progress,
                    "Risk": act.risk,
                    "Duration": act.duration,
                    "Predecessors": pred_str,
                }
            )
        return pd.DataFrame(data)
