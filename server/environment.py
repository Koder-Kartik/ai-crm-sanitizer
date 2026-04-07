# server/environment.py
# CRM Sanitizer — Core Environment Logic
#
# This file implements the three required OpenEnv methods:
#   reset()  → start a new episode, generate fresh dirty dataset
#   step()   → agent takes one action, get reward and observation
#   state()  → return current episode metadata
#
# One instance of CRMEnvironment runs inside the Docker container.
# The FastAPI server in app.py calls these methods on every request.

import sys
import os
import uuid
import copy
from typing import Any, Dict, List, Optional

# Add parent directory to path so we can import models.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CRMAction, CRMObservation, CRMState
from tasks import TaskData, generate_task, get_column_stats
from grader import EpisodeGrader


# ─────────────────────────────────────────────
# TABLE RENDERER
# Converts internal table to Markdown
# This is what the agent reads
# ─────────────────────────────────────────────

def render_table_markdown(table: List[Dict[str, Any]]) -> str:
    """
    Convert a list of row dicts to a clean Markdown table.
    None values shown as ██ MISSING ██ so they stand out.

    Example output:
    | uid  | name       | email           | phone        |
    |------|------------|-----------------|--------------|
    | 1001 | John Smith | john@acme.com   | (555) 123-4567 |
    | 1002 | Jane Doe   | ██ MISSING ██   | (555) 987-6543 |
    """
    if not table:
        return "| (empty table) |"

    columns = list(table[0].keys())

    # Build header row
    header = "| " + " | ".join(str(col) for col in columns) + " |"

    # Build separator row
    separator = "| " + " | ".join("---" for _ in columns) + " |"

    # Build data rows
    rows = []
    for row in table:
        cells = []
        for col in columns:
            val = row.get(col)
            if val is None:
                cells.append("██ MISSING ██")
            else:
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join([header, separator] + rows)


# ─────────────────────────────────────────────
# HINT BUILDER
# Generates issues_remaining based on hint level
# Easy   → full descriptions
# Medium → column names only
# Hard   → empty list
# ─────────────────────────────────────────────

def build_hints(
    task_data: TaskData,
    fixed_issue_keys: set,
) -> List[str]:
    """
    Build the issues_remaining hint list based on task difficulty.
    Only shows unfixed issues.
    """
    hint_level = task_data.hint_level

    if hint_level == "none":
        # Hard task — agent must discover everything
        return []

    hints = []
    for issue in task_data.issues:
        issue_key = f"{issue.uid}::{issue.column}"
        if issue_key in fixed_issue_keys:
            continue  # already fixed, don't hint it

        if hint_level == "full":
            # Easy task — tell agent exactly what's wrong
            if issue.issue_type == "missing_value":
                hints.append(
                    f"uid {issue.uid}: column '{issue.column}' has a missing value"
                )
            elif issue.issue_type == "duplicate_row":
                hints.append(
                    f"uid {issue.uid}: this row is a duplicate — remove it"
                )
            elif issue.issue_type == "phone_format":
                hints.append(
                    f"uid {issue.uid}: column 'phone' has inconsistent format"
                )
            elif issue.issue_type == "city_format":
                hints.append(
                    f"uid {issue.uid}: column 'city' has inconsistent casing"
                )
            elif issue.issue_type == "date_format":
                hints.append(
                    f"uid {issue.uid}: column 'join_date' has non-standard format"
                )
            elif issue.issue_type == "negative_value":
                hints.append(
                    f"uid {issue.uid}: column 'loyalty_points' has a negative value"
                )
            elif issue.is_ambiguous:
                hints.append(
                    f"uid {issue.uid}: column '{issue.column}' has an ambiguous value"
                )

        elif hint_level == "partial":
            # Medium task — tell agent which columns have issues
            hint = f"column '{issue.column}' has issues"
            if hint not in hints:
                hints.append(hint)

    return hints


# ─────────────────────────────────────────────
# MAIN ENVIRONMENT CLASS
# ─────────────────────────────────────────────

class CRMEnvironment:
    """
    CRM Sanitizer Environment.

    Implements the OpenEnv interface:
        reset(task_id, seed)  → CRMObservation
        step(action)          → CRMObservation
        state()               → CRMState

    One instance handles one episode at a time.
    The FastAPI server creates a new session per WebSocket connection.
    """

    # Required by OpenEnv — allows multiple clients simultaneously
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        # Episode state — all None until reset() is called
        self._task_data: Optional[TaskData] = None
        self._current_table: Optional[List[Dict[str, Any]]] = None
        self._grader: Optional[EpisodeGrader] = None
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._done: bool = False
        self._seed: int = 42
        self._task_id: str = ""
        self._recent_actions: List[str] = []

    # ─────────────────────────────────────────
    # reset()
    # Starts a fresh episode
    # ─────────────────────────────────────────

    def reset(
        self,
        task_id: str = "easy_basic_fix",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> CRMObservation:
        """
        Start a new episode.

        Args:
            task_id:    Which task to run.
                        One of: easy_basic_fix, medium_format_dedup, hard_full_audit
            seed:       Random seed for dataset generation.
                        Same seed → same dirty table → reproducible scores.
                        Default: 42 (used by baseline inference script)
            episode_id: Optional identifier for this episode.

        Returns:
            CRMObservation with the initial dirty table and task description.
        """
        # Use provided seed or default
        self._seed       = seed if seed is not None else 42
        self._task_id    = task_id
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._done       = False
        self._recent_actions = []

        # Generate fresh dirty dataset using seed
        self._task_data = generate_task(task_id, self._seed)

        # Work on a copy of the dirty table
        # (we never modify the original — grader needs it for reference)
        self._current_table = copy.deepcopy(self._task_data.dirty_table)

        # Create fresh grader for this episode
        self._grader = EpisodeGrader(self._task_data)

        # Build initial observation
        return self._build_observation(
            last_action_result=f"Episode started. Task: {self._task_data.task_name}",
            reward=None,
            done=False,
            column_stats=None,
        )

    # ─────────────────────────────────────────
    # step()
    # Agent takes one action
    # ─────────────────────────────────────────

    def step(
        self,
        action: CRMAction,
        **kwargs,
    ) -> CRMObservation:
        """
        Execute one agent action and return the result.

        Args:
            action: CRMAction with operation, column, row_uid, value, reason

        Returns:
            CRMObservation with updated table, reward, and feedback.
        """
        # Guard: environment must be reset first
        if self._task_data is None or self._grader is None:
            return CRMObservation(
                done=True,
                reward=0.0,
                table_markdown="",
                last_action_result="error: call reset() before step()",
                task_description="",
                task_id="",
                step_number=0,
            )

        # Guard: episode already finished
        if self._done:
            return CRMObservation(
                done=True,
                reward=0.0,
                table_markdown=render_table_markdown(self._current_table),
                last_action_result="error: episode is already done — call reset()",
                task_description=self._task_data.task_description,
                task_id=self._task_id,
                step_number=self._step_count,
                issues_fixed=self._grader.progress_summary()["issues_fixed"],
                total_issues=len(self._task_data.issues),
            )

        self._step_count += 1
        # Track action for agent memory
        action_summary = f"{action.operation}(uid={action.row_uid},col={action.column!r})"

        # Log the agent's reasoning if provided
        if action.reason:
            print(
                f"[AGENT REASON] step={self._step_count} "
                f"op={action.operation} uid={action.row_uid} "
                f"col={action.column} reason={action.reason!r}",
                flush=True,
            )

        # ── Apply the action to the table ──
        column_stats_result = None

        if action.operation == "remove_duplicate":
            # Remove the row from current table
            before_len = len(self._current_table)
            self._current_table = [
                row for row in self._current_table
                if row["uid"] != action.row_uid
            ]
            removed = before_len - len(self._current_table)
            if removed == 0:
                # uid not found in table — will get penalized by grader
                pass

        elif action.operation == "fill_missing":
            # Fill the specific cell
            for row in self._current_table:
                if row["uid"] == action.row_uid:
                    if row.get(action.column) is None:
                        row[action.column] = action.value
                    break

        elif action.operation == "standardize_format":
            # Update the specific cell with standardized value
            for row in self._current_table:
                if row["uid"] == action.row_uid:
                    row[action.column] = action.value
                    break

        elif action.operation == "fix_value":
            # Update the specific cell
            for row in self._current_table:
                if row["uid"] == action.row_uid:
                    row[action.column] = action.value
                    break

        elif action.operation == "bulk_fix_column":
            # The grader handles which rows get fixed
            # We update the entire column in the table
            for row in self._current_table:
                if action.column in row and row[action.column] is not None:
                    row[action.column] = action.value

        elif action.operation == "flag_ambiguous":
            # Record the agent's chosen resolution in the table
            for row in self._current_table:
                if row["uid"] == action.row_uid:
                    row[action.column] = action.value
                    break

        elif action.operation == "get_column_stats":
            # No table modification — just gather stats
            column_stats_result = get_column_stats(
                self._current_table, action.column
            )

        elif action.operation == "submit":
            # No table modification — grader handles scoring
            pass

        # ── Score the action ──
        reward, result_message = self._grader.grade_action(
            operation=action.operation,
            uid=action.row_uid,
            column=action.column,
            value=action.value,
            current_table=self._current_table,
        )

        # Add to recent actions memory
        self._recent_actions.append(
            f"Step {self._step_count}: {action_summary} → {result_message[:60]}"
        )
        # Keep only last 5
        if len(self._recent_actions) > 5:
            self._recent_actions = self._recent_actions[-5:]

        # ── Check if episode is done ──
        done = False

        if action.operation == "submit":
            done = True

        elif self._step_count >= self._task_data.max_steps:
            done = True
            result_message += f" | max steps ({self._task_data.max_steps}) reached"

        elif self._grader.all_fixed():
            # All issues resolved — encourage agent to submit
            result_message += " | all issues fixed! Use submit to complete."

        self._done = done

        # ── Build and return observation ──
        return self._build_observation(
            last_action_result=result_message,
            reward=reward,
            done=done,
            column_stats=column_stats_result,
        )

    # ─────────────────────────────────────────
    # state()
    # Returns episode metadata
    # ─────────────────────────────────────────

    def state(self) -> CRMState:
        """
        Return current episode metadata.
        Agents don't usually call this — it's for monitoring.
        """
        if self._task_data is None:
            return CRMState()

        progress = self._grader.progress_summary() if self._grader else {}

        return CRMState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            seed=self._seed,
            total_issues=progress.get("total_issues", 0),
            issues_fixed=progress.get("issues_fixed", 0),
            issues_wrongly_touched=0,
            max_steps=self._task_data.max_steps,
            is_complete=self._done,
        )

    # ─────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────

    def _build_observation(
        self,
        last_action_result: str,
        reward: Optional[float],
        done: bool,
        column_stats: Optional[Dict[str, Any]],
    ) -> CRMObservation:
        """
        Build a CRMObservation from current episode state.
        Called after every reset() and step().
        """
        progress = self._grader.progress_summary() if self._grader else {}
        fixed_keys = self._grader.fixed_issues if self._grader else set()

        hints = build_hints(self._task_data, fixed_keys)

        return CRMObservation(
            done=done,
            reward=reward,
            table_markdown=render_table_markdown(self._current_table),
            issues_remaining=hints,
            issues_fixed=progress.get("issues_fixed", 0),
            total_issues=progress.get("total_issues", 0),
            last_action_result=last_action_result,
            task_description=self._task_data.task_description,
            task_id=self._task_id,
            step_number=self._step_count,
            column_stats=column_stats,
            recent_actions     = self._recent_actions,   
        )