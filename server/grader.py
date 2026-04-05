# server/grader.py
# CRM Sanitizer — Scoring Engine
#
# This file scores every action the agent takes.
# It compares what the agent did against the ground truth
# stored in TaskData and returns a reward signal.
#
# REWARD STRUCTURE:
#   +0.15  correctly fixed a real issue
#   -0.08  "fixed" something that was not broken
#   -0.03  redundant action (already fixed this issue before)
#   +0.60  submit bonus when ALL issues are resolved
#   -0.40  submit penalty when issues still remain
#
# FINAL SCORE:
#   total_reward / max_possible_reward
#   clamped to [0.0, 1.0]
#
# DETERMINISM GUARANTEE:
#   This grader has zero randomness.
#   Same actions on same dataset always produce same score.

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from tasks import IssueRecord, TaskData


# ─────────────────────────────────────────────
# REWARD CONSTANTS
# Change these in one place — affects everything
# ─────────────────────────────────────────────

REWARD_CORRECT_FIX    = +0.15   # agent fixed a real issue correctly
REWARD_WRONG_FIX      = -0.08   # agent "fixed" something not broken
REWARD_REDUNDANT      = -0.03   # agent fixed same issue twice
REWARD_SUBMIT_SUCCESS = +0.60   # submitted with all issues resolved
REWARD_SUBMIT_PARTIAL = -0.40   # submitted with issues still remaining
REWARD_EXPLORE        = +0.00   # get_column_stats: neutral, no penalty


# ─────────────────────────────────────────────
# VALUE NORMALIZERS
# Used to compare agent-provided values
# against ground truth in a forgiving way
# ─────────────────────────────────────────────

def normalize_phone(value: str) -> str:
    """
    Strip all non-digit characters from a phone number.
    (555) 123-4567  →  5551234567
    555.123.4567    →  5551234567
    Both are the same phone. Agent passes if digits match.
    """
    if value is None:
        return ""
    return re.sub(r"\D", "", str(value))


def normalize_city(value: str) -> str:
    """
    Lowercase and strip whitespace from city name.
    'New York' == 'new york' == 'NEW YORK'
    Agent passes if lowercase versions match.
    """
    if value is None:
        return ""
    return str(value).strip().lower()


def normalize_date(value: str) -> str:
    """
    Try to convert any date format to YYYY-MM-DD.
    Handles: MM/DD/YYYY, DD-MM-YYYY, MM-DD-YY
    Returns original string if parsing fails.
    """
    if value is None:
        return ""
    value = str(value).strip()

    # Already ISO format
    if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        return value

    # MM/DD/YYYY
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", value)
    if m:
        month, day, year = m.group(1), m.group(2), m.group(3)
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    # DD-MM-YYYY
    m = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{4})$", value)
    if m:
        day, month, year = m.group(1), m.group(2), m.group(3)
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    # MM-DD-YY
    m = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{2})$", value)
    if m:
        month, day, year = m.group(1), m.group(2), m.group(3)
        full_year = f"20{m.group(3)}"
        return f"{full_year}-{month.zfill(2)}-{day.zfill(2)}"

    return value  # return as-is if we can't parse it


def normalize_value(column: str, value: Any) -> str:
    """
    Route value to the correct normalizer based on column name.
    Everything else is compared as stripped lowercase string.
    """
    if value is None:
        return "__NONE__"

    if column == "phone":
        return normalize_phone(str(value))
    elif column == "city":
        return normalize_city(str(value))
    elif column == "join_date":
        return normalize_date(str(value))
    else:
        return str(value).strip().lower()


# ─────────────────────────────────────────────
# COLUMN STATS GENERATOR
# Returns when agent uses get_column_stats
# ─────────────────────────────────────────────

def get_column_stats(
    table: List[Dict[str, Any]],
    column: str
) -> Dict[str, Any]:
    """
    Returns a summary of a column to help the agent explore.
    This is what the agent sees when it uses get_column_stats.

    Returns:
        null_count:     how many rows have None/empty in this column
        unique_count:   how many distinct values exist
        sample_values:  up to 5 example values from the column
        dtype:          guessed data type (string/number/date)
        min_value:      minimum (for numeric columns)
        max_value:      maximum (for numeric columns)
    """
    if not table:
        return {"error": "table is empty"}

    # Check column exists
    if column not in table[0]:
        return {"error": f"column '{column}' does not exist"}

    values = [row.get(column) for row in table]

    # Count nulls
    null_count = sum(1 for v in values if v is None or str(v).strip() == "")

    # Unique non-null values
    non_null = [v for v in values if v is not None and str(v).strip() != ""]
    unique_vals = list(set(str(v) for v in non_null))
    unique_count = len(unique_vals)

    # Sample values (up to 5, sorted for consistency)
    sample_values = sorted(unique_vals)[:5]

    # Guess dtype
    dtype = "string"
    if column == "loyalty_points":
        dtype = "integer"
    elif column == "join_date":
        dtype = "date"
    elif column in ("uid",):
        dtype = "integer"

    # Min/max for numeric columns
    min_value = None
    max_value = None
    if dtype == "integer":
        numeric_vals = []
        for v in non_null:
            try:
                numeric_vals.append(int(v))
            except (ValueError, TypeError):
                pass
        if numeric_vals:
            min_value = min(numeric_vals)
            max_value = max(numeric_vals)

    return {
        "column":        column,
        "total_rows":    len(table),
        "null_count":    null_count,
        "unique_count":  unique_count,
        "sample_values": sample_values,
        "dtype":         dtype,
        "min_value":     min_value,
        "max_value":     max_value,
    }


# ─────────────────────────────────────────────
# MAIN GRADER CLASS
# One instance lives for the whole episode
# ─────────────────────────────────────────────

class EpisodeGrader:
    """
    Tracks and scores agent actions for one episode.

    Created once at reset().
    grade_action() called after every step().
    final_score() called when episode ends.
    """

    def __init__(self, task_data: TaskData):
        self.task_data = task_data

        # All issues in this episode
        self.all_issues: List[IssueRecord] = task_data.issues

        # Track which issues have been fixed (by uid+column key)
        self.fixed_issues: Set[str] = set()

        # Track ambiguous uids — agent can flag these legitimately
        self.ambiguous_uids: Set[int] = {
            issue.uid
            for issue in self.all_issues
            if issue.is_ambiguous
        }

        # Running reward total
        self.total_reward: float = 0.0

        # Max possible reward
        # = sum of correct fixes + submit bonus
        non_ambiguous = [i for i in self.all_issues if not i.is_ambiguous]
        ambiguous     = [i for i in self.all_issues if i.is_ambiguous]
        self.max_possible_reward: float = (
            len(non_ambiguous) * REWARD_CORRECT_FIX +
            len(ambiguous)     * REWARD_CORRECT_FIX +  # flagging counts
            REWARD_SUBMIT_SUCCESS
        )

        # Action history for redundancy detection
        # key: (uid, column, operation) → True if already done
        self.action_history: Dict[Tuple, bool] = {}

    def _issue_key(self, uid: int, column: str) -> str:
        """Unique string key for one issue."""
        return f"{uid}::{column}"

    def _find_issue(self, uid: int, column: str) -> Optional[IssueRecord]:
        """Find the issue record for a specific uid+column."""
        for issue in self.all_issues:
            if issue.uid == uid and issue.column == column:
                return issue
        return None

    def _is_already_fixed(self, uid: int, column: str) -> bool:
        return self._issue_key(uid, column) in self.fixed_issues

    def _mark_fixed(self, uid: int, column: str) -> None:
        self.fixed_issues.add(self._issue_key(uid, column))

    def issues_remaining_count(self) -> int:
        return len(self.all_issues) - len(self.fixed_issues)

    def all_fixed(self) -> bool:
        return len(self.fixed_issues) >= len(self.all_issues)

    # ─────────────────────────────────────────
    # ACTION GRADERS
    # One method per operation type
    # ─────────────────────────────────────────

    def grade_fill_missing(
        self,
        uid: int,
        column: str,
        value: str,
        current_table: List[Dict[str, Any]],
    ) -> Tuple[float, str]:
        """
        Grade a fill_missing action.
        Returns (reward, result_message)
        """
        action_key = (uid, column, "fill_missing")

        # Redundancy check
        if action_key in self.action_history:
            self.total_reward += REWARD_REDUNDANT
            return REWARD_REDUNDANT, f"redundant: uid {uid} column '{column}' already fixed"

        self.action_history[action_key] = True

        # Find the issue
        issue = self._find_issue(uid, column)

        if issue is None or issue.issue_type != "missing_value":
            # No missing value issue on this cell — wrong action
            self.total_reward += REWARD_WRONG_FIX
            return REWARD_WRONG_FIX, f"wrong: uid {uid} column '{column}' has no missing value"

        if self._is_already_fixed(uid, column):
            self.total_reward += REWARD_REDUNDANT
            return REWARD_REDUNDANT, f"redundant: already filled uid {uid} '{column}'"

        # Check if value is reasonable (non-empty)
        if not value or str(value).strip() == "":
            self.total_reward += REWARD_WRONG_FIX
            return REWARD_WRONG_FIX, f"wrong: fill value cannot be empty"

        # Accept any non-empty fill — agent earns reward
        # (We can't demand exact value for missing fields)
        self._mark_fixed(uid, column)
        self.total_reward += REWARD_CORRECT_FIX
        return REWARD_CORRECT_FIX, f"success: filled missing '{column}' for uid {uid}"

    def grade_remove_duplicate(
        self,
        uid: int,
        current_table: List[Dict[str, Any]],
    ) -> Tuple[float, str]:
        """
        Grade a remove_duplicate action.
        """
        action_key = (uid, "uid", "remove_duplicate")

        if action_key in self.action_history:
            self.total_reward += REWARD_REDUNDANT
            return REWARD_REDUNDANT, f"redundant: uid {uid} already removed"

        self.action_history[action_key] = True

        # Find the duplicate issue
        issue = self._find_issue(uid, "uid")

        if issue is None or issue.issue_type != "duplicate_row":
            # Not actually a duplicate
            self.total_reward += REWARD_WRONG_FIX
            return REWARD_WRONG_FIX, f"wrong: uid {uid} is not a duplicate"

        if self._is_already_fixed(uid, "uid"):
            self.total_reward += REWARD_REDUNDANT
            return REWARD_REDUNDANT, f"redundant: uid {uid} already removed"

        self._mark_fixed(uid, "uid")
        self.total_reward += REWARD_CORRECT_FIX
        return REWARD_CORRECT_FIX, f"success: removed duplicate uid {uid}"

    def grade_standardize_format(
        self,
        uid: int,
        column: str,
        value: str,
        current_table: List[Dict[str, Any]],
    ) -> Tuple[float, str]:
        """
        Grade a standardize_format action on a single row.
        Accepts any correct standardization (not just exact match).
        """
        action_key = (uid, column, "standardize_format")

        if action_key in self.action_history:
            self.total_reward += REWARD_REDUNDANT
            return REWARD_REDUNDANT, f"redundant: uid {uid} '{column}' already standardized"

        self.action_history[action_key] = True

        issue = self._find_issue(uid, column)

        if issue is None or issue.issue_type not in (
            "phone_format", "city_format", "date_format"
        ):
            self.total_reward += REWARD_WRONG_FIX
            return REWARD_WRONG_FIX, f"wrong: uid {uid} column '{column}' has no format issue"

        if self._is_already_fixed(uid, column):
            self.total_reward += REWARD_REDUNDANT
            return REWARD_REDUNDANT, f"redundant: uid {uid} '{column}' already fixed"

        # Compare normalized versions
        agent_normalized = normalize_value(column, value)
        clean_normalized = normalize_value(column, issue.clean_value)

        if agent_normalized == clean_normalized:
            self._mark_fixed(uid, column)
            self.total_reward += REWARD_CORRECT_FIX
            return REWARD_CORRECT_FIX, f"success: standardized '{column}' for uid {uid}"
        else:
            self.total_reward += REWARD_WRONG_FIX
            return (
                REWARD_WRONG_FIX,
                f"wrong: '{value}' is not a valid standardization for uid {uid} '{column}'"
            )

    def grade_fix_value(
        self,
        uid: int,
        column: str,
        value: str,
        current_table: List[Dict[str, Any]],
    ) -> Tuple[float, str]:
        """
        Grade a fix_value action.
        Used for out-of-range values like negative loyalty_points.
        """
        action_key = (uid, column, "fix_value")

        if action_key in self.action_history:
            self.total_reward += REWARD_REDUNDANT
            return REWARD_REDUNDANT, f"redundant: uid {uid} '{column}' already fixed"

        self.action_history[action_key] = True

        issue = self._find_issue(uid, column)

        if issue is None or issue.issue_type != "negative_value":
            self.total_reward += REWARD_WRONG_FIX
            return REWARD_WRONG_FIX, f"wrong: uid {uid} '{column}' has no value error"

        if self._is_already_fixed(uid, column):
            self.total_reward += REWARD_REDUNDANT
            return REWARD_REDUNDANT, f"redundant: uid {uid} '{column}' already fixed"

        # For loyalty_points: any non-negative integer is acceptable
        try:
            int_val = int(value)
            if int_val >= 0:
                self._mark_fixed(uid, column)
                self.total_reward += REWARD_CORRECT_FIX
                return REWARD_CORRECT_FIX, f"success: fixed '{column}' for uid {uid} to {int_val}"
            else:
                self.total_reward += REWARD_WRONG_FIX
                return REWARD_WRONG_FIX, f"wrong: loyalty_points must be >= 0, got {int_val}"
        except (ValueError, TypeError):
            self.total_reward += REWARD_WRONG_FIX
            return REWARD_WRONG_FIX, f"wrong: '{value}' is not a valid integer"

    def grade_bulk_fix_column(
        self,
        column: str,
        value: str,
        current_table: List[Dict[str, Any]],
    ) -> Tuple[float, str]:
        """
        Grade a bulk_fix_column action.
        Fixes all format issues in an entire column at once.
        Awards reward for each issue fixed.
        """
        action_key = ("ALL", column, "bulk_fix_column")

        if action_key in self.action_history:
            self.total_reward += REWARD_REDUNDANT
            return REWARD_REDUNDANT, f"redundant: bulk fix already applied to '{column}'"

        self.action_history[action_key] = True

        # Find all unfixed format issues in this column
        fixable_types = {
            "phone":     "phone_format",
            "city":      "city_format",
            "join_date": "date_format",
        }

        if column not in fixable_types:
            self.total_reward += REWARD_WRONG_FIX
            return (
                REWARD_WRONG_FIX,
                f"wrong: bulk_fix_column not applicable to '{column}'"
            )

        expected_issue_type = fixable_types[column]
        issues_in_column = [
            i for i in self.all_issues
            if i.column == column
            and i.issue_type == expected_issue_type
            and not self._is_already_fixed(i.uid, column)
        ]

        if not issues_in_column:
            self.total_reward += REWARD_WRONG_FIX
            return REWARD_WRONG_FIX, f"wrong: no unfixed format issues in '{column}'"

        # Award reward for each issue fixed
        fixed_count = 0
        for issue in issues_in_column:
            self._mark_fixed(issue.uid, column)
            self.total_reward += REWARD_CORRECT_FIX
            fixed_count += 1

        return (
            REWARD_CORRECT_FIX * fixed_count,
            f"success: bulk fixed {fixed_count} issues in '{column}'"
        )

    def grade_flag_ambiguous(
        self,
        uid: int,
        column: str,
        value: str,
    ) -> Tuple[float, str]:
        """
        Grade a flag_ambiguous action.

        STRICT CONTROL:
        - Only pre-defined ambiguous rows can be flagged
        - Flagging a non-ambiguous row → penalty
        - Flagging correctly → same reward as a correct fix
        """
        action_key = (uid, column, "flag_ambiguous")

        if action_key in self.action_history:
            self.total_reward += REWARD_REDUNDANT
            return REWARD_REDUNDANT, f"redundant: uid {uid} '{column}' already flagged"

        self.action_history[action_key] = True

        # Check if this uid is genuinely ambiguous
        if uid not in self.ambiguous_uids:
            self.total_reward += REWARD_WRONG_FIX
            return (
                REWARD_WRONG_FIX,
                f"wrong: uid {uid} is not an ambiguous case — "
                f"flagging non-ambiguous rows is penalized"
            )

        issue = self._find_issue(uid, column)
        if issue is None or not issue.is_ambiguous:
            self.total_reward += REWARD_WRONG_FIX
            return (
                REWARD_WRONG_FIX,
                f"wrong: uid {uid} column '{column}' is not ambiguous"
            )

        if self._is_already_fixed(uid, column):
            self.total_reward += REWARD_REDUNDANT
            return REWARD_REDUNDANT, f"redundant: uid {uid} '{column}' already handled"

        # Check if agent's chosen value is in acceptable list
        # OR if they are flagging without a specific value
        agent_normalized = normalize_value(column, value)
        acceptable_normalized = [
            normalize_value(column, v) for v in issue.acceptable_values
        ]

        flagged_marker = normalize_value(column, "FLAGGED")

        if agent_normalized in acceptable_normalized or agent_normalized == flagged_marker:
            self._mark_fixed(uid, column)
            self.total_reward += REWARD_CORRECT_FIX
            return (
                REWARD_CORRECT_FIX,
                f"success: correctly handled ambiguous case uid {uid} '{column}'"
            )
        else:
            self.total_reward += REWARD_WRONG_FIX
            return (
                REWARD_WRONG_FIX,
                f"wrong: value '{value}' is not acceptable for ambiguous uid {uid} '{column}'"
            )

    def grade_submit(self) -> Tuple[float, str]:
        """
        Grade the submit action.
        Big bonus if everything is fixed.
        Big penalty if issues remain.
        """
        remaining = self.issues_remaining_count()

        if remaining == 0:
            self.total_reward += REWARD_SUBMIT_SUCCESS
            return (
                REWARD_SUBMIT_SUCCESS,
                f"success: all issues resolved — perfect submission!"
            )
        else:
            self.total_reward += REWARD_SUBMIT_PARTIAL
            return (
                REWARD_SUBMIT_PARTIAL,
                f"partial: {remaining} issues still unresolved at submission"
            )

    # ─────────────────────────────────────────
    # MAIN ENTRY POINT
    # Environment calls this after every step
    # ─────────────────────────────────────────

    def grade_action(
        self,
        operation: str,
        uid: int,
        column: str,
        value: str,
        current_table: List[Dict[str, Any]],
    ) -> Tuple[float, str]:
        """
        Route action to the correct grader.
        Returns (reward, result_message).
        """
        if operation == "fill_missing":
            return self.grade_fill_missing(uid, column, value, current_table)

        elif operation == "remove_duplicate":
            return self.grade_remove_duplicate(uid, current_table)

        elif operation == "standardize_format":
            return self.grade_standardize_format(uid, column, value, current_table)

        elif operation == "fix_value":
            return self.grade_fix_value(uid, column, value, current_table)

        elif operation == "get_column_stats":
            # Exploration action — no reward, no penalty
            stats = get_column_stats(current_table, column)
            return REWARD_EXPLORE, f"stats: {stats}"

        elif operation == "bulk_fix_column":
            return self.grade_bulk_fix_column(column, value, current_table)

        elif operation == "flag_ambiguous":
            return self.grade_flag_ambiguous(uid, column, value)

        elif operation == "submit":
            return self.grade_submit()

        else:
            # Unknown operation — penalize
            self.total_reward += REWARD_WRONG_FIX
            return REWARD_WRONG_FIX, f"error: unknown operation '{operation}'"

    # ─────────────────────────────────────────
    # FINAL SCORE
    # Called at episode end
    # ─────────────────────────────────────────

    def final_score(self) -> float:
        """
        Returns normalized score between 0.0 and 1.0.
        total_reward / max_possible_reward, clamped.
        """
        if self.max_possible_reward <= 0:
            return 0.0

        raw = self.total_reward / self.max_possible_reward
        return float(max(0.0, min(1.0, raw)))

    def progress_summary(self) -> Dict[str, Any]:
        """
        Returns a human-readable summary of episode progress.
        Used for logging and debugging.
        """
        return {
            "total_issues":     len(self.all_issues),
            "issues_fixed":     len(self.fixed_issues),
            "issues_remaining": self.issues_remaining_count(),
            "total_reward":     round(self.total_reward, 4),
            "max_reward":       round(self.max_possible_reward, 4),
            "current_score":    round(self.final_score(), 4),
        }