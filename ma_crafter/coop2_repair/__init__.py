"""COOP2-Repair for MA-Crafter.

The environment-independent core (plan views, predictor, evaluator, repair
controller, message protocol) lives in ../../coop2_repair_core and is shared
with CUBE. It is added to this package's search path below, so
``coop2_repair.core`` and friends resolve to the shared files. This package
adds the MA-Crafter adapter, its plan-effect prediction, and repair guidance.
"""

from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[2] / "coop2_repair_core"))

from .core import (
    ActionOutcome,
    AgentPlanView,
    COOP2_CONSTRAINT_TYPES,
    ConstraintResult,
    ConstraintType,
    Coop2TraceLogger,
    Coop2TraceEvent,
    ParallelEnvAdapter,
    PettingZooParallelAdapter,
    TaskSpec,
)
from .evaluator import PreExecutionConstraintEvaluator, PreExecutionEvaluation
from .macrafter_adapter import MacrafterCoopAdapter
from .prediction import (
    HeuristicPlanTimelinePredictor,
    PlanTimelinePrediction,
    PredictedActionEvent,
    PredictedAgentState,
)
from .repair_controller import Coop2RepairController
from .message_protocol import (
    COOP2_REPAIR_CONTENT_TYPE,
    COOP2_REPAIR_MESSAGE_TYPE,
    COOP2_REPAIR_SENDER_ID,
    MESSAGE_TYPE_METADATA_KEY,
    coop2_repair_metadata,
)

__all__ = [
    "ActionOutcome",
    "AgentPlanView",
    "ConstraintResult",
    "COOP2_CONSTRAINT_TYPES",
    "ConstraintType",
    "Coop2TraceLogger",
    "Coop2TraceEvent",
    "Coop2RepairController",
    "ParallelEnvAdapter",
    "PettingZooParallelAdapter",
    "PreExecutionConstraintEvaluator",
    "PreExecutionEvaluation",
    "HeuristicPlanTimelinePredictor",
    "PlanTimelinePrediction",
    "PredictedActionEvent",
    "PredictedAgentState",
    "TaskSpec",
    "MacrafterCoopAdapter",
    "COOP2_REPAIR_CONTENT_TYPE",
    "COOP2_REPAIR_MESSAGE_TYPE",
    "COOP2_REPAIR_SENDER_ID",
    "MESSAGE_TYPE_METADATA_KEY",
    "coop2_repair_metadata",
]
