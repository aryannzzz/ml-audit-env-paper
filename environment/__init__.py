"""
ML Experiment Integrity Auditor — OpenEnv Environment

An RL environment for training agents to audit ML experiments
for data leakage, methodology violations, and result manipulation.
"""
from environment.env import MLAuditEnv
from environment.models import (
    Observation,
    Action,
    EpisodeState,
    FlagRecord,
    GraderRequest,
    GraderResponse,
)
from environment.grader import grade, grade_single_flag, evidence_found
from environment.generator import POOL, generate, get_pool_stats

__all__ = [
    "MLAuditEnv",
    "Observation",
    "Action",
    "EpisodeState",
    "FlagRecord",
    "GraderRequest",
    "GraderResponse",
    "grade",
    "grade_single_flag",
    "evidence_found",
    "POOL",
    "generate",
    "get_pool_stats",
]

__version__ = "1.2.0"
