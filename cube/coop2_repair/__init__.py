"""COOP2-Repair for CUBE.

The environment-independent core (plan views, predictor, evaluator, repair
controller, message protocol) lives in ../../coop2_repair_core and is shared
with MA-Crafter. It is added to this package's search path below, so
``coop2_repair.core`` and friends resolve to the shared files. This package
adds the CUBE adapter.
"""

from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[2] / "coop2_repair_core"))

from .core import (
    ActionOutcome,
    AgentPlanView,
    ConstraintResult,
    ConstraintType,
    Coop2TraceLogger,
    PettingZooParallelAdapter,
    TaskSpec,
)
from .evaluator import PreExecutionConstraintEvaluator, PreExecutionEvaluation
from .repair_controller import Coop2RepairController
from .cube_adapter import CubeCoopAdapter

__all__ = [
    "ActionOutcome",
    "AgentPlanView",
    "ConstraintResult",
    "ConstraintType",
    "Coop2TraceLogger",
    "PettingZooParallelAdapter",
    "TaskSpec",
    "PreExecutionConstraintEvaluator",
    "PreExecutionEvaluation",
    "Coop2RepairController",
    "CubeCoopAdapter",
]
