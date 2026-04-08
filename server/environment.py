# server/environment.py
# CRM Sanitizer — Core Environment Logic

import sys
import os
import uuid
import copy
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CRMAction, CRMObservation, CRMState
from tasks import TaskData, generate_task, get_column_stats
from grader import EpisodeGrader, clamp_score, SCORE_MIN, SCORE_MAX


def render_table_markdown(table: List[Dict[str, Any]]) -> str:
    if not table:
        return "| (empty table) |"
    columns = list(table[0].keys())
    header    = "| " + " | ".join(str(c) for c in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for row in table:
        cells = []
        for col in columns:
            val = row.get(col)
            cells.append("██ MISSING ██" if val is None else str(val))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator] + rows)


def build_hints(task_data: TaskData, fixed_issue_keys: set) -> List[str]:
    hint_level = task_data.hint_level
    if hint_level == "none":
        return []
    hints = []
    for issue in task_data.issues:
        issue_key = f"{issue.uid}::{issue.column}"
        if issue_key in fixed_issue_keys:
            continue
        if hint_level == "full":
            if issue.issue_type == "missing_value":
                hints.append(f"uid {issue.uid}: column '{issue.column}' has a missing value")
            elif issue.issue_type == "duplicate_row":
                hints.append(f"uid {issue.uid}: this row is a duplicate — remove it")
            elif issue.issue_type == "phone_format":
                hints.append(f"uid {issue.uid}: column 'phone' has inconsistent format")
            elif issue.issue_type == "city_format":
                hints.append(f"uid {issue.uid}: column 'city' has inconsistent casing")
            elif issue.issue_type == "date_format":
                hints.append(f"uid {issue.uid}: column 'join_date' has non-standard format")
            elif issue.issue_type == "negative_value":
                hints.append(f"uid {issue.uid}: column 'loyalty_points' has a negative value")
            elif issue.is_ambiguous:
                hints.append(f"uid {issue.uid}: column '{issue.column}' has an ambiguous value")
        elif hint_level == "partial":
            hint = f"column '{issue.column}' has issues"
            if hint not in hints:
                hints.append(hint)
    return hints


class CRMEnvironment:
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._task_data:     Optional[TaskData]             = None
        self._current_table: Optional[List[Dict[str, Any]]] = None
        self._grader:        Optional[EpisodeGrader]        = None
        self._episode_id:    Optional[str]                  = None
        self._step_count:    int                            = 0
        self._done:          bool                           = False
        self._seed:          int                            = 42
        self._task_id:       str                            = ""
        self._recent_actions: List[str]                     = []

    # ─────────────────────────────────────────
    # reset
    # ─────────────────────────────────────────

    def reset(
        self,
        task_id:    str           = "easy_basic_fix",
        seed:       Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> CRMObservation:
        self._seed           = seed if seed is not None else 42
        self._task_id        = task_id
        self._episode_id     = episode_id or str(uuid.uuid4())
        self._step_count     = 0
        self._done           = False
        self._recent_actions = []

        self._task_data     = generate_task(task_id, self._seed)
        self._current_table = copy.deepcopy(self._task_data.dirty_table)
        self._grader        = EpisodeGrader(self._task_data)

        return self._build_observation(
            last_action_result=f"Episode started. Task: {self._task_data.task_name}",
            reward=None,
            done=False,
            column_stats=None,
        )

    # ─────────────────────────────────────────
    # step
    # ─────────────────────────────────────────

    def step(self, action: CRMAction, **kwargs) -> CRMObservation:
        if self._task_data is None or self._grader is None:
            return CRMObservation(
                done=True, reward=0.01,
                table_markdown="",
                last_action_result="error: call reset() before step()",
                task_description="", task_id="", step_number=0,
            )

        if self._done:
            return CRMObservation(
                done=True, reward=0.01,
                table_markdown=render_table_markdown(self._current_table),
                last_action_result="error: episode already done — call reset()",
                task_description=self._task_data.task_description,
                task_id=self._task_id,
                step_number=self._step_count,
                issues_fixed=self._grader.progress_summary()["issues_fixed"],
                total_issues=len(self._task_data.issues),
            )

        self._step_count += 1

        if action.reason:
            print(
                f"[AGENT REASON] step={self._step_count} "
                f"op={action.operation} uid={action.row_uid} "
                f"col={action.column} reason={action.reason!r}",
                flush=True,
            )

        column_stats_result = None

        # Apply physical table change
        if action.operation == "remove_duplicate":
            self._current_table = [
                r for r in self._current_table if r["uid"] != action.row_uid
            ]

        elif action.operation == "fill_missing":
            for row in self._current_table:
                if row["uid"] == action.row_uid:
                    if row.get(action.column) is None:
                        row[action.column] = action.value
                    break

        elif action.operation in ("standardize_format", "fix_value", "flag_ambiguous"):
            for row in self._current_table:
                if row["uid"] == action.row_uid:
                    row[action.column] = action.value
                    break

        elif action.operation == "bulk_fix_column":
            for row in self._current_table:
                if action.column in row and row[action.column] is not None:
                    row[action.column] = action.value

        elif action.operation == "get_column_stats":
            column_stats_result = get_column_stats(self._current_table, action.column)

        # Score the action
        reward, result_message = self._grader.grade_action(
            operation     = action.operation,
            uid           = action.row_uid,
            column        = action.column,
            value         = action.value,
            current_table = self._current_table,
        )

        # Track recent actions for agent memory
        action_summary = f"{action.operation}(uid={action.row_uid},col={action.column!r})"
        self._recent_actions.append(
            f"Step {self._step_count}: {action_summary} → {result_message[:60]}"
        )
        if len(self._recent_actions) > 5:
            self._recent_actions = self._recent_actions[-5:]

        # Check episode done
        done = False
        if action.operation == "submit":
            done = True
        elif self._step_count >= self._task_data.max_steps:
            done = True
            result_message += f" | max steps ({self._task_data.max_steps}) reached"
        elif self._grader.all_fixed():
            result_message += " | all issues fixed! Use submit to complete."

        self._done = done

        # Pass raw reward as context; _build_observation always uses grader.final_score()
        return self._build_observation(
            last_action_result = result_message,
            reward             = reward,
            done               = done,
            column_stats       = column_stats_result,
        )

    # ─────────────────────────────────────────
    # state
    # ─────────────────────────────────────────

    def state(self) -> CRMState:
        if self._task_data is None:
            return CRMState()
        progress = self._grader.progress_summary() if self._grader else {}
        return CRMState(
            episode_id             = self._episode_id,
            step_count             = self._step_count,
            task_id                = self._task_id,
            seed                   = self._seed,
            total_issues           = progress.get("total_issues", 0),
            issues_fixed           = progress.get("issues_fixed", 0),
            issues_wrongly_touched = 0,
            max_steps              = self._task_data.max_steps,
            is_complete            = self._done,
        )

    # ─────────────────────────────────────────
    # internal
    # ─────────────────────────────────────────

    def _build_observation(
        self,
        last_action_result: str,
        reward:             Optional[float],
        done:               bool,
        column_stats:       Optional[Dict[str, Any]],
    ) -> CRMObservation:
        progress   = self._grader.progress_summary() if self._grader else {}
        fixed_keys = self._grader.fixed_issues if self._grader else set()
        hints      = build_hints(self._task_data, fixed_keys)

        # Reward must ALWAYS be strictly in (0.01, 0.99) — every step, not just done.
        # The raw action rewards (e.g. -0.40, 0.0, +0.60) are deltas, not valid scores.
        # We normalise against max_possible_reward via grader.final_score() every time.
        if self._grader is not None:
            final_reward = self._grader.final_score()  # already clamped to [0.01, 0.99]
        elif reward is not None:
            import math
            r = float(reward)
            if not math.isfinite(r) or r <= 0.0:
                final_reward = 0.01
            elif r >= 1.0:
                final_reward = 0.99
            else:
                final_reward = r
        else:
            final_reward = 0.01

        return CRMObservation(
            done               = done,
            reward             = final_reward,
            table_markdown     = render_table_markdown(self._current_table),
            issues_remaining   = hints,
            issues_fixed       = progress.get("issues_fixed", 0),
            total_issues       = progress.get("total_issues", 0),
            last_action_result = last_action_result,
            task_description   = self._task_data.task_description,
            task_id            = self._task_id,
            step_number        = self._step_count,
            column_stats       = column_stats,
            recent_actions     = self._recent_actions,
        )