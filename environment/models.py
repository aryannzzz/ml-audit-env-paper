"""
Pydantic v2 models for ML Experiment Integrity Auditor.
All typed models for observations, actions, state, and API requests/responses.
"""
from typing import Optional
from pydantic import BaseModel, Field, model_validator


# === Flag Record ===

class FlagRecord(BaseModel):
    """Record of a violation flag raised by the agent."""
    flag_id: str = Field(description="Unique identifier for this flag")
    violation_type: str = Field(description="V1-V6 violation type")
    evidence_artifact: str = Field(description="Artifact containing evidence")
    evidence_quote: str = Field(description="Exact quote from artifact")
    severity: str = Field(description="high | medium | low")
    step_raised: int = Field(description="Step number when flag was raised")


# === Action Model ===

VALID_ACTION_TYPES = {"inspect", "compare", "flag", "unflag", "submit"}
VALID_VIOLATION_TYPES = {"V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"}
VALID_VERDICTS = {"pass", "revise", "reject"}
VALID_SEVERITIES = {"high", "medium", "low"}


class Action(BaseModel):
    """
    Agent action model with validation for each action type.

    Action types:
    - inspect: Read a single artifact
    - compare: Read two artifacts side-by-side
    - flag: Raise a violation flag with evidence
    - unflag: Remove a previously raised flag
    - submit: End episode with verdict and summary
    """
    type: str = Field(description="Action type: inspect | compare | flag | unflag | submit")

    # inspect fields
    artifact: Optional[str] = Field(default=None, description="Artifact name for inspect action")

    # compare fields
    artifact_a: Optional[str] = Field(default=None, description="First artifact for compare")
    artifact_b: Optional[str] = Field(default=None, description="Second artifact for compare")

    # flag fields
    violation_type: Optional[str] = Field(default=None, description="V1-V6 violation type")
    evidence_artifact: Optional[str] = Field(default=None, description="Artifact with evidence")
    evidence_quote: Optional[str] = Field(default=None, description="Exact quote from artifact")
    severity: Optional[str] = Field(default=None, description="high | medium | low")

    # unflag fields
    flag_id: Optional[str] = Field(default=None, description="Flag ID to remove")

    # submit fields
    verdict: Optional[str] = Field(default=None, description="pass | revise | reject")
    summary: Optional[str] = Field(default=None, description="Brief summary of findings")

    @model_validator(mode='after')
    def validate_action_fields(self):
        """Validate that required fields are present for each action type."""
        action_type = self.type

        if action_type not in VALID_ACTION_TYPES:
            raise ValueError(f"Invalid action type '{action_type}'. Must be one of: {VALID_ACTION_TYPES}")

        if action_type == "inspect":
            if not self.artifact:
                raise ValueError("inspect action requires 'artifact' field")

        elif action_type == "compare":
            if not self.artifact_a or not self.artifact_b:
                raise ValueError("compare action requires 'artifact_a' and 'artifact_b' fields")
            if self.artifact_a == self.artifact_b:
                raise ValueError("compare action requires two different artifacts")

        elif action_type == "flag":
            if not self.violation_type:
                raise ValueError("flag action requires 'violation_type' field")
            if self.violation_type not in VALID_VIOLATION_TYPES:
                raise ValueError(f"Invalid violation_type '{self.violation_type}'. Must be one of: {VALID_VIOLATION_TYPES}")
            if not self.evidence_artifact:
                raise ValueError("flag action requires 'evidence_artifact' field")
            if not self.evidence_quote:
                raise ValueError("flag action requires 'evidence_quote' field")
            if self.severity and self.severity not in VALID_SEVERITIES:
                raise ValueError(f"Invalid severity '{self.severity}'. Must be one of: {VALID_SEVERITIES}")

        elif action_type == "unflag":
            if not self.flag_id:
                raise ValueError("unflag action requires 'flag_id' field")

        elif action_type == "submit":
            if not self.verdict:
                raise ValueError("submit action requires 'verdict' field")
            if self.verdict not in VALID_VERDICTS:
                raise ValueError(f"Invalid verdict '{self.verdict}'. Must be one of: {VALID_VERDICTS}")
            if not self.summary:
                raise ValueError("submit action requires 'summary' field")

        return self


# === Observation Model ===

class Observation(BaseModel):
    """
    Full observation returned after each step.
    Ground truth (violations list, expected_verdict) is NEVER included.
    """
    # Episode context
    experiment_id: str = Field(description="Unique experiment identifier")
    task_description: str = Field(description="Human-readable task instructions")
    goal: str = Field(description="Human-readable description of what the agent should do")

    # Dataset metadata
    dataset_type: str = Field(description="tabular | timeseries")

    # Artifact state
    available_artifacts: list[str] = Field(description="Artifacts available for inspection")
    inspected_artifacts: list[str] = Field(description="Artifacts already read by agent")

    # Last action feedback
    last_action_result: Optional[str] = Field(
        default=None,
        description="Content returned by last action"
    )
    last_action_error: Optional[str] = Field(
        default=None,
        description="Error message if last action failed"
    )

    # Flags state
    flags_raised: list[FlagRecord] = Field(
        default_factory=list,
        description="All flags currently raised by agent"
    )

    # Progress tracking
    steps_used: int = Field(description="Number of steps taken so far")
    step_budget: int = Field(description="Maximum steps allowed")

    # Reward tracking
    step_reward: float = Field(description="Reward from last action")
    cumulative_reward: float = Field(description="Total reward so far")

    # System message
    message: str = Field(default="", description="System message")


# === Episode State Model ===

class EpisodeState(BaseModel):
    """
    Complete episode state for checkpointing and debugging.
    Returned by GET /state endpoint.
    """
    experiment_id: str = Field(description="Unique experiment identifier")
    archetype: str = Field(description="tabular_clf | timeseries_reg | tabular_multi")
    task_difficulty: str = Field(description="easy | medium | hard")
    dataset_type: str = Field(description="tabular | timeseries")
    steps_used: int = Field(description="Steps taken so far")
    step_budget: int = Field(description="Maximum allowed steps")
    inspected_artifacts: list[str] = Field(default_factory=list)
    flags_raised: list[dict] = Field(default_factory=list)
    cumulative_reward: float = Field(description="Total reward accumulated")
    done: bool = Field(description="Whether episode has ended")
    is_clean_experiment: bool = Field(
        description="True if this is an adversarially clean experiment"
    )


# === API Request/Response Models ===

class ResetRequest(BaseModel):
    """Request body for /reset endpoint."""
    task: str = Field(default="easy", description="Task difficulty: easy | medium | hard")


class StepRequest(BaseModel):
    """Request body for /step endpoint."""
    action: dict = Field(description="Action to execute")


class GraderRequest(BaseModel):
    """Request body for /grader endpoint."""
    task: str = Field(description="Task difficulty")
    experiment_id: Optional[str] = Field(default=None, description="Optional experiment ID")
    flags: list[dict] = Field(description="List of flag objects")
    verdict: str = Field(description="Verdict: pass | revise | reject")
    steps_used: int = Field(description="Steps taken")


class GraderResponse(BaseModel):
    """Response from /grader endpoint."""
    score: float = Field(description="Final score 0.0-1.0")
    breakdown: dict = Field(description="Score breakdown")
    flag_results: list[dict] = Field(description="Per-flag results")


class TaskInfo(BaseModel):
    """Task metadata for /tasks endpoint."""
    id: str
    description: str
    difficulty: int
    max_steps: int


class HealthResponse(BaseModel):
    """Response from /health endpoint."""
    status: str
    environment: str
    version: str


class BaselineResponse(BaseModel):
    """Response from /baseline endpoint."""
    task: str
    agent_type: str
    score: float
    steps: int
    violations_found: int
    false_positives: int
    verdict: str