# server/grader.py
# CRM Sanitizer — Scoring Engine
#
# REWARD STRUCTURE:
#   +0.15  correctly fixed a real issue
#   -0.08  fixed something not broken
#   -0.03  redundant action
#   +0.60  submit bonus when ALL issues resolved
#   -0.40  submit penalty when issues remain
#
# FINAL SCORE:
#   Strictly in open interval (0.01, 0.99)
#   Validator rejects exact 0.0 and exact 1.0

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from tasks import IssueRecord, TaskData

# ─────────────────────────────────────────────
# REWARD CONSTANTS
# ─────────────────────────────────────────────

REWARD_CORRECT_FIX    = +0.15
REWARD_WRONG_FIX      = -0.08
REWARD_REDUNDANT      = -0.03
REWARD_SUBMIT_SUCCESS = +0.60
REWARD_SUBMIT_PARTIAL = -0.40
REWARD_EXPLORE        = +0.00

# Strictly open interval — validator rejects 0.0 and 1.0
SCORE_MIN = 0.01
SCORE_MAX = 0.99


# ─────────────────────────────────────────────
# SCORE CLAMPING — used everywhere
# ─────────────────────────────────────────────

def clamp_score(raw: float) -> float:
    """
    Clamp any float to strictly open interval (0.01, 0.99).
    Handles NaN, Inf, negative, and > 1 inputs safely.
    """
    if not math.isfinite(raw):
        return SCORE_MIN
    return float(max(SCORE_MIN, min(SCORE_MAX, raw)))


# ─────────────────────────────────────────────
# VALUE NORMALIZERS
# ─────────────────────────────────────────────

def normalize_phone(value: str) -> str:
    if value is None:
        return ""
    return re.sub(r"\D", "", str(value))


def normalize_city(value: str) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def normalize_date(value: str) -> str:
    if value is None:
        return ""
    value = str(value).strip()

    if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        return value

    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", value)
    if m:
        return f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"

    m = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{4})$", value)
    if m:
        return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"

    m = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{2})$", value)
    if m:
        return f"20{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"

    return value


def normalize_value(column: str, value: Any) -> str:
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
# COLUMN STATS
# ─────────────────────────────────────────────

def get_column_stats(
    table: List[Dict[str, Any]],
    column: str
) -> Dict[str, Any]:
    if not table:
        return {"error": "table is empty"}
    if column not in table[0]:
        return {"error": f"column '{column}' does not exist"}

    values    = [row.get(column) for row in table]
    null_count = sum(1 for v in values if v is None or str(v).strip() == "")
    non_null  = [v for v in values if v is not None and str(v).strip() != ""]
    unique_vals = sorted(set(str(v) for v in non_null))
    unique_count = len(unique_vals)
    sample_values = unique_vals[:5]

    dtype = "string"
    if column == "loyalty_points":
        dtype = "integer"
    elif column == "join_date":
        dtype = "date"
    elif column == "uid":
        dtype = "integer"

    min_value = max_value = None
    if dtype == "integer":
        nums = []
        for v in non_null:
            try:
                nums.append(int(v))
            except (ValueError, TypeError):
                pass
        if nums:
            min_value = min(nums)
            max_value = max(nums)

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
# EPISODE GRADER
# ─────────────────────────────────────────────

class EpisodeGrader:
    """
    Scores every agent action for one episode.
    final_score() always returns strictly (0.01, 0.99).
    """

    def __init__(self, task_data: TaskData):
        self.task_data   = task_data
        self.all_issues  = task_data.issues
        self.fixed_issues: Set[str] = set()

        self.ambiguous_uids: Set[int] = {
            i.uid for i in self.all_issues if i.is_ambiguous
        }

        self.total_reward: float = 0.0

        non_ambiguous = [i for i in self.all_issues if not i.is_ambiguous]
        ambiguous     = [i for i in self.all_issues if i.is_ambiguous]
        self.max_possible_reward: float = (
            (len(non_ambiguous) + len(ambiguous)) * REWARD_CORRECT_FIX
            + REWARD_SUBMIT_SUCCESS
        )

        self.action_history: Dict[Tuple, bool] = {}

    # ── helpers ──

    def _key(self, uid: int, column: str) -> str:
        return f"{uid}::{column}"

    def _find_issue(self, uid: int, column: str) -> Optional[IssueRecord]:
        for issue in self.all_issues:
            if issue.uid == uid and issue.column == column:
                return issue
        return None

    def _fixed(self, uid: int, column: str) -> bool:
        return self._key(uid, column) in self.fixed_issues

    def _mark(self, uid: int, column: str) -> None:
        self.fixed_issues.add(self._key(uid, column))

    def issues_remaining_count(self) -> int:
        return len(self.all_issues) - len(self.fixed_issues)

    def all_fixed(self) -> bool:
        return len(self.fixed_issues) >= len(self.all_issues)

    # ── action graders ──

    def grade_fill_missing(
        self, uid: int, column: str, value: str,
        current_table: List[Dict[str, Any]]
    ) -> Tuple[float, str]:
        key = (uid, column, "fill_missing")
        if key in self.action_history:
            self.total_reward += REWARD_REDUNDANT
            return REWARD_REDUNDANT, f"redundant: uid {uid} '{column}' already attempted"
        self.action_history[key] = True

        issue = self._find_issue(uid, column)
        if issue is None or issue.issue_type != "missing_value":
            self.total_reward += REWARD_WRONG_FIX
            return REWARD_WRONG_FIX, f"wrong: uid {uid} '{column}' has no missing value"
        if self._fixed(uid, column):
            self.total_reward += REWARD_REDUNDANT
            return REWARD_REDUNDANT, f"redundant: uid {uid} '{column}' already filled"
        if not value or str(value).strip() == "":
            self.total_reward += REWARD_WRONG_FIX
            return REWARD_WRONG_FIX, "wrong: fill value cannot be empty"

        self._mark(uid, column)
        self.total_reward += REWARD_CORRECT_FIX
        return REWARD_CORRECT_FIX, f"success: filled '{column}' for uid {uid}"

    def grade_remove_duplicate(
        self, uid: int,
        current_table: List[Dict[str, Any]]
    ) -> Tuple[float, str]:
        key = (uid, "uid", "remove_duplicate")
        if key in self.action_history:
            self.total_reward += REWARD_REDUNDANT
            return REWARD_REDUNDANT, f"redundant: uid {uid} already removed"
        self.action_history[key] = True

        issue = self._find_issue(uid, "uid")
        if issue is None or issue.issue_type != "duplicate_row":
            self.total_reward += REWARD_WRONG_FIX
            return REWARD_WRONG_FIX, f"wrong: uid {uid} is not a duplicate"
        if self._fixed(uid, "uid"):
            self.total_reward += REWARD_REDUNDANT
            return REWARD_REDUNDANT, f"redundant: uid {uid} already removed"

        self._mark(uid, "uid")
        self.total_reward += REWARD_CORRECT_FIX
        return REWARD_CORRECT_FIX, f"success: removed duplicate uid {uid}"

    def grade_standardize_format(
        self, uid: int, column: str, value: str,
        current_table: List[Dict[str, Any]]
    ) -> Tuple[float, str]:
        key = (uid, column, "standardize_format")
        if key in self.action_history:
            self.total_reward += REWARD_REDUNDANT
            return REWARD_REDUNDANT, f"redundant: uid {uid} '{column}' already standardized"
        self.action_history[key] = True

        issue = self._find_issue(uid, column)
        if issue is None or issue.issue_type not in (
            "phone_format", "city_format", "date_format"
        ):
            self.total_reward += REWARD_WRONG_FIX
            return REWARD_WRONG_FIX, f"wrong: uid {uid} '{column}' has no format issue"
        if self._fixed(uid, column):
            self.total_reward += REWARD_REDUNDANT
            return REWARD_REDUNDANT, f"redundant: uid {uid} '{column}' already fixed"

        if normalize_value(column, value) == normalize_value(column, issue.clean_value):
            self._mark(uid, column)
            self.total_reward += REWARD_CORRECT_FIX
            return REWARD_CORRECT_FIX, f"success: standardized '{column}' for uid {uid}"
        else:
            self.total_reward += REWARD_WRONG_FIX
            return REWARD_WRONG_FIX, f"wrong: '{value}' invalid for uid {uid} '{column}'"

    def grade_fix_value(
        self, uid: int, column: str, value: str,
        current_table: List[Dict[str, Any]]
    ) -> Tuple[float, str]:
        key = (uid, column, "fix_value")
        if key in self.action_history:
            self.total_reward += REWARD_REDUNDANT
            return REWARD_REDUNDANT, f"redundant: uid {uid} '{column}' already fixed"
        self.action_history[key] = True

        issue = self._find_issue(uid, column)
        if issue is None or issue.issue_type != "negative_value":
            self.total_reward += REWARD_WRONG_FIX
            return REWARD_WRONG_FIX, f"wrong: uid {uid} '{column}' has no value error"
        if self._fixed(uid, column):
            self.total_reward += REWARD_REDUNDANT
            return REWARD_REDUNDANT, f"redundant: uid {uid} '{column}' already fixed"

        try:
            int_val = int(float(str(value)))
            if int_val >= 0:
                self._mark(uid, column)
                self.total_reward += REWARD_CORRECT_FIX
                return REWARD_CORRECT_FIX, f"success: fixed '{column}' uid {uid} → {int_val}"
            else:
                self.total_reward += REWARD_WRONG_FIX
                return REWARD_WRONG_FIX, f"wrong: loyalty_points must be >= 0, got {int_val}"
        except (ValueError, TypeError):
            self.total_reward += REWARD_WRONG_FIX
            return REWARD_WRONG_FIX, f"wrong: '{value}' is not a valid number"

    def grade_bulk_fix_column(
        self, column: str, value: str,
        current_table: List[Dict[str, Any]]
    ) -> Tuple[float, str]:
        key = ("ALL", column, "bulk_fix_column")
        if key in self.action_history:
            self.total_reward += REWARD_REDUNDANT
            return REWARD_REDUNDANT, f"redundant: bulk fix already applied to '{column}'"
        self.action_history[key] = True

        fixable = {
            "phone":     "phone_format",
            "city":      "city_format",
            "join_date": "date_format",
        }
        if column not in fixable:
            self.total_reward += REWARD_WRONG_FIX
            return REWARD_WRONG_FIX, f"wrong: bulk_fix_column not applicable to '{column}'"

        pending = [
            i for i in self.all_issues
            if i.column == column
            and i.issue_type == fixable[column]
            and not self._fixed(i.uid, column)
        ]
        if not pending:
            self.total_reward += REWARD_WRONG_FIX
            return REWARD_WRONG_FIX, f"wrong: no unfixed format issues in '{column}'"

        for issue in pending:
            self._mark(issue.uid, column)
            self.total_reward += REWARD_CORRECT_FIX

        total = REWARD_CORRECT_FIX * len(pending)
        return total, f"success: bulk fixed {len(pending)} issues in '{column}'"

    def grade_flag_ambiguous(
        self, uid: int, column: str, value: str
    ) -> Tuple[float, str]:
        key = (uid, column, "flag_ambiguous")
        if key in self.action_history:
            self.total_reward += REWARD_REDUNDANT
            return REWARD_REDUNDANT, f"redundant: uid {uid} '{column}' already flagged"
        self.action_history[key] = True

        if uid not in self.ambiguous_uids:
            self.total_reward += REWARD_WRONG_FIX
            return REWARD_WRONG_FIX, f"wrong: uid {uid} is not ambiguous"

        issue = self._find_issue(uid, column)
        if issue is None or not issue.is_ambiguous:
            self.total_reward += REWARD_WRONG_FIX
            return REWARD_WRONG_FIX, f"wrong: uid {uid} '{column}' is not ambiguous"
        if self._fixed(uid, column):
            self.total_reward += REWARD_REDUNDANT
            return REWARD_REDUNDANT, f"redundant: uid {uid} '{column}' already handled"

        agent_norm = normalize_value(column, value)
        acceptable = [normalize_value(column, v) for v in issue.acceptable_values]
        flagged    = normalize_value(column, "FLAGGED")

        if agent_norm in acceptable or agent_norm == flagged:
            self._mark(uid, column)
            self.total_reward += REWARD_CORRECT_FIX
            return REWARD_CORRECT_FIX, f"success: ambiguous case uid {uid} '{column}' handled"
        else:
            self.total_reward += REWARD_WRONG_FIX
            return REWARD_WRONG_FIX, f"wrong: '{value}' not acceptable for uid {uid} '{column}'"

    def grade_submit(self) -> Tuple[float, str]:
        remaining = self.issues_remaining_count()
        if remaining == 0:
            self.total_reward += REWARD_SUBMIT_SUCCESS
            return REWARD_SUBMIT_SUCCESS, "success: all issues resolved!"
        else:
            self.total_reward += REWARD_SUBMIT_PARTIAL
            return REWARD_SUBMIT_PARTIAL, f"partial: {remaining} issues still unresolved"

    # ── main router ──

    def grade_action(
        self,
        operation: str,
        uid: int,
        column: str,
        value: str,
        current_table: List[Dict[str, Any]],
    ) -> Tuple[float, str]:

        # Protect uid column
        if column == "uid" and operation not in (
            "remove_duplicate", "submit", "get_column_stats"
        ):
            self.total_reward += REWARD_WRONG_FIX
            return (
                REWARD_WRONG_FIX,
                "error: 'uid' is immutable. Fix data columns: "
                "name, email, phone, company, city, join_date, loyalty_points"
            )

        if operation == "fill_missing":
            return self.grade_fill_missing(uid, column, value, current_table)
        elif operation == "remove_duplicate":
            return self.grade_remove_duplicate(uid, current_table)
        elif operation == "standardize_format":
            return self.grade_standardize_format(uid, column, value, current_table)
        elif operation == "fix_value":
            return self.grade_fix_value(uid, column, value, current_table)
        elif operation == "get_column_stats":
            stats_key = ("STATS", column, "get_column_stats")
            if stats_key in self.action_history:
                self.total_reward += REWARD_REDUNDANT
                return REWARD_REDUNDANT, f"redundant: already checked '{column}' stats"
            self.action_history[stats_key] = True
            stats = get_column_stats(current_table, column)
            return REWARD_EXPLORE, f"stats: {stats}"
        elif operation == "bulk_fix_column":
            return self.grade_bulk_fix_column(column, value, current_table)
        elif operation == "flag_ambiguous":
            return self.grade_flag_ambiguous(uid, column, value)
        elif operation == "submit":
            return self.grade_submit()
        else:
            self.total_reward += REWARD_WRONG_FIX
            return REWARD_WRONG_FIX, f"error: unknown operation '{operation}'"

    # ── scoring ──

    def final_score(self) -> float:
        """
        Returns score strictly in open interval (0.01, 0.99).
        Validator rejects exact 0.0 and exact 1.0.
        """
        if self.max_possible_reward <= 0:
            return SCORE_MIN
        try:
            raw = self.total_reward / self.max_possible_reward
        except ZeroDivisionError:
            return SCORE_MIN
        return clamp_score(raw)

    def progress_summary(self) -> Dict[str, Any]:
        return {
            "total_issues":     len(self.all_issues),
            "issues_fixed":     len(self.fixed_issues),
            "issues_remaining": self.issues_remaining_count(),
            "total_reward":     round(self.total_reward, 4),
            "max_reward":       round(self.max_possible_reward, 4),
            "current_score":    round(self.final_score(), 4),
        }