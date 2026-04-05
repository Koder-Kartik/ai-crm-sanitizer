# models.py
# CRM Sanitizer — Typed data models
#
# These Pydantic models define the complete API contract of the environment.
# Every action, observation, and state is typed and validated automatically.
# Judges read this file to understand what the environment does.

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# BASE CLASSES (required by OpenEnv spec)
# ─────────────────────────────────────────────

class Action(BaseModel):
    """Base class for all actions."""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """Base class for all observations."""
    done: bool = False
    reward: Optional[float] = None


class State(BaseModel):
    """Base class for all states."""
    episode_id: Optional[str] = None
    step_count: int = 0


# ─────────────────────────────────────────────
# CRM SANITIZER — ACTION
# ─────────────────────────────────────────────

class CRMAction(Action):
    """
    Every action the agent can take in the CRM Sanitizer environment.

    Operations available:
      fill_missing       — Fill a missing/null value in a specific cell
      remove_duplicate   — Remove a duplicate row by its uid
      standardize_format — Standardize format of a column (dates, phones, etc.)
      fix_value          — Correct a specific wrong value in a cell
      get_column_stats   — Explore a column before acting (returns stats)
      bulk_fix_column    — Apply a standard fix to all rows in a column
      flag_ambiguous     — Explicitly flag a row as having no single correct answer
      submit             — Declare the task complete

    All row references use permanent uid — never row index.
    This prevents index-shifting bugs when rows are deleted.
    """

    operation: str = Field(
        description=(
            "One of: fill_missing, remove_duplicate, standardize_format, "
            "fix_value, get_column_stats, bulk_fix_column, flag_ambiguous, submit"
        )
    )

    column: str = Field(
        default="",
        description="Column name to operate on. Empty string for submit."
    )

    row_uid: int = Field(
        default=-1,
        description=(
            "Permanent unique ID of the target row. "
            "Use -1 for column-level operations or submit."
        )
    )

    value: str = Field(
        default="",
        description=(
            "New value to set, or parameter for the operation. "
            "For standardize_format: use 'date', 'phone', 'email', 'city'. "
            "For bulk_fix_column: use same format options. "
            "For flag_ambiguous: use your chosen resolution value."
        )
    )

    reason: str = Field(
        default="",
        description=(
            "Optional. Explain why you chose this action. "
            "Not graded but logged — helps human reviewers understand agent reasoning."
        )
    )


# ─────────────────────────────────────────────
# CRM SANITIZER — OBSERVATION
# ─────────────────────────────────────────────

class CRMObservation(Observation):
    """
    Everything the agent can see after each step.

    Hint levels vary by task difficulty:
      Easy   — issues_remaining contains full list of all problems
      Medium — issues_remaining contains only affected column names
      Hard   — issues_remaining is empty, agent must discover issues
    """

    # The current state of the CRM table, rendered as Markdown
    # Agents read this to understand what needs fixing
    table_markdown: str = Field(
        description="Current CRM table rendered as a Markdown table."
    )

    # Hint system — tiered by difficulty
    issues_remaining: List[str] = Field(
        default_factory=list,
        description=(
            "Easy: full issue descriptions. "
            "Medium: affected column names only. "
            "Hard: always empty — agent must discover issues."
        )
    )

    # Progress tracking
    issues_fixed: int = Field(
        default=0,
        description="Number of issues correctly fixed so far this episode."
    )

    total_issues: int = Field(
        default=0,
        description="Total number of issues in this episode."
    )

    # Feedback from the last action
    last_action_result: str = Field(
        default="",
        description=(
            "Result of the previous action. Examples: "
            "'success: filled missing email for uid 1003', "
            "'error: column does not exist', "
            "'no_change: value was already correct'"
        )
    )

    # Task context
    task_description: str = Field(
        default="",
        description="Human-readable description of the current task."
    )

    task_id: str = Field(
        default="",
        description="Machine-readable task identifier."
    )

    step_number: int = Field(
        default=0,
        description="Current step number in this episode."
    )

    # Column stats (returned when agent uses get_column_stats)
    column_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Populated when agent uses get_column_stats operation. "
            "Contains: null_count, unique_count, sample_values, dtype."
        )
    )


# ─────────────────────────────────────────────
# CRM SANITIZER — STATE
# ─────────────────────────────────────────────

class CRMState(State):
    """
    Episode-level metadata. Returned by the state() method.
    Agents typically don't read this — it's for monitoring and debugging.
    """

    task_id: str = Field(
        default="",
        description="Which task is currently running."
    )

    seed: int = Field(
        default=42,
        description="Random seed used to generate this episode's dataset."
    )

    total_issues: int = Field(
        default=0,
        description="Total issues injected at episode start."
    )

    issues_fixed: int = Field(
        default=0,
        description="Issues correctly fixed so far."
    )

    issues_wrongly_touched: int = Field(
        default=0,
        description="Number of incorrect fix attempts."
    )

    max_steps: int = Field(
        default=15,
        description="Maximum steps allowed for this task."
    )

    is_complete: bool = Field(
        default=False,
        description="True when agent submits or max steps reached."
    )