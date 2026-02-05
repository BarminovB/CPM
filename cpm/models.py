from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Relationship:
    """Represents a precedence relationship between two activities."""

    predecessor_id: str
    relation_type: str  # FS, SS, FF, SF
    lag: int  # Can be positive or negative

    def __str__(self) -> str:
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

    def reset_calculations(self) -> None:
        """Reset all calculated values."""
        self.es = None
        self.ef = None
        self.ls = None
        self.lf = None
        self.total_float = None
        self.free_float = None
        self.is_critical = False
