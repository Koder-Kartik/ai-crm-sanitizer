# inference.py
# CRM Sanitizer — Baseline Inference Script
#
# SETUP (PowerShell):
#   $env:HF_TOKEN="hf_your_token_here"
#   $env:API_BASE_URL="https://router.huggingface.co/v1"
#   $env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
#   $env:SERVER_URL="http://localhost:7860"
#
# Run ALL tasks:
#   python inference.py
#
# Run ONE task at a time (saves tokens):
#   python inference.py --task easy
#   python inference.py --task medium
#   python inference.py --task hard

import argparse
import json
import os
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ─────────────────────────────────────────────
# CONFIGURATION
# All values from environment variables
# ─────────────────────────────────────────────

API_BASE_URL = os.environ.get(
    "API_BASE_URL",
    "https://router.huggingface.co/v1"      # HF Router default
)

MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "Qwen/Qwen2.5-72B-Instruct"             # Best free HF model
)

API_KEY = os.environ.get(
    "HF_TOKEN",
    os.environ.get("OPENAI_API_KEY", "")
)

SERVER_URL        = os.environ.get("SERVER_URL", "http://localhost:7860")
BENCHMARK         = "crm-sanitizer-v1"
BASELINE_SEED     = 42
SUCCESS_THRESHOLD = 0.40
TEMPERATURE       = 0.01
MAX_TOKENS        = 512

MAX_STEPS = {
    "easy_basic_fix":      15,
    "medium_format_dedup": 20,
    "hard_full_audit":     30,
}

ALL_TASKS = ["easy_basic_fix", "medium_format_dedup", "hard_full_audit"]


# ─────────────────────────────────────────────
# REQUIRED LOG FUNCTIONS
# Judges parse these exactly — do not change
# ─────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_str = error if error else "null"
    print(
        f"[STEP] step={step} "
        f"action={action} "
        f"reward={reward:.2f} "
        f"done={str(done).lower()} "
        f"error={error_str}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    # Spec format: rewards=0.15,0.15,0.60  (no brackets, 2dp)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} "
        f"rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a CRM data cleaning agent. Fix data quality issues one at a time.

RESPOND WITH ONLY RAW JSON. NO MARKDOWN. NO EXPLANATION. JUST THE JSON OBJECT.

JSON FORMAT:
{"operation":"OPERATION_NAME","column":"COLUMN","row_uid":UID,"value":"VALUE","reason":"REASON"}

OPERATIONS:
- fill_missing: fill a null/MISSING cell
- remove_duplicate: remove a duplicate row
- standardize_format: fix phone/date/city format for ONE row
- fix_value: fix wrong value (negative numbers etc)
- bulk_fix_column: fix all format issues in entire column at once
- get_column_stats: explore a column (use before fixing)
- flag_ambiguous: flag a row with genuinely ambiguous data
- submit: call this when ALL issues are fixed

RULES:
1. Use the uid number from the uid column (e.g. 1001, 2003, 3015)
2. row_uid=-1 only for bulk_fix_column and get_column_stats
3. Phones must be: (XXX) XXX-XXXX
4. Dates must be: YYYY-MM-DD
5. Cities must be: Title Case (e.g. New York not new york)
6. loyalty_points must be >= 0
7. MISSING in a cell = null value, fill it
8. Submit only when ALL issues are fixed
9. Do NOT repeat an action you already took on the same uid+column"""


# ─────────────────────────────────────────────
# VALID OPERATIONS
# ─────────────────────────────────────────────

VALID_OPERATIONS = {
    "fill_missing", "remove_duplicate", "standardize_format",
    "fix_value", "get_column_stats", "bulk_fix_column",
    "flag_ambiguous", "submit",
}


# ─────────────────────────────────────────────
# ACTION PARSER
# ─────────────────────────────────────────────

def parse_action(text: str) -> Dict[str, Any]:
    safe_default = {
        "operation": "submit",
        "column":    "",
        "row_uid":   -1,
        "value":     "",
        "reason":    "parse failed",
    }

    if not text or not text.strip():
        return safe_default

    text = text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not json_match:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)

    if json_match:
        text = json_match.group(0)
    else:
        print(f"[DEBUG] No JSON in: {text[:80]!r}", flush=True)
        return safe_default

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        text = text.replace("'", '"')
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            print(f"[DEBUG] JSON parse failed: {text[:80]!r}", flush=True)
            return safe_default

    operation = str(data.get("operation", "submit")).strip().lower()
    if operation not in VALID_OPERATIONS:
        for valid_op in VALID_OPERATIONS:
            if valid_op in operation or operation in valid_op:
                operation = valid_op
                break
        else:
            operation = "submit"

    try:
        row_uid = int(data.get("row_uid", data.get("uid", -1)))
    except (ValueError, TypeError):
        row_uid = -1

    return {
        "operation": operation,
        "column":    str(data.get("column", "")).strip(),
        "row_uid":   row_uid,
        "value":     str(data.get("value", "")).strip(),
        "reason":    str(data.get("reason", "")).strip(),
    }


# ─────────────────────────────────────────────
# OBSERVATION FORMATTER
# ─────────────────────────────────────────────

def format_observation(obs: Any, history: List[str]) -> str:
    lines = []

    # Guard all attribute accesses with getattr fallbacks
    issues_fixed  = getattr(obs, "issues_fixed",  0) or 0
    total_issues  = getattr(obs, "total_issues",  0) or 0
    remaining     = total_issues - issues_fixed

    lines.append(
        f"Progress: {issues_fixed} fixed, "
        f"{remaining} remaining out of {total_issues} total"
    )

    last_action_result = getattr(obs, "last_action_result", None)
    if last_action_result and "Episode started" not in last_action_result:
        lines.append(f"Last action result: {last_action_result}")

    # Recent actions memory — stops agent repeating itself
    recent_actions = getattr(obs, "recent_actions", None)
    if recent_actions:
        lines.append("\nACTIONS ALREADY TAKEN (DO NOT REPEAT THESE):")
        for a in recent_actions:
            lines.append(f"  ✓ {a}")

    issues_remaining = getattr(obs, "issues_remaining", None)
    if issues_remaining:
        lines.append("\nISSUES TO FIX:")
        for hint in issues_remaining:
            lines.append(f"  - {hint}")
    else:
        lines.append("\nNo hints. Find issues by reading the table carefully.")

    column_stats = getattr(obs, "column_stats", None)
    if column_stats and not column_stats.get("error"):
        s = column_stats
        lines.append(
            f"\nColumn '{s.get('column','')}' stats: "
            f"nulls={s.get('null_count',0)}, "
            f"samples={s.get('sample_values',[])[:3]}"
        )

    if history:
        lines.append("\nRecent actions:")
        for h in history[-3:]:
            lines.append(f"  {h}")

    table_markdown = getattr(obs, "table_markdown", "")
    lines.append(f"\nCURRENT TABLE:\n{table_markdown}")

    if remaining == 0:
        lines.append("\nALL ISSUES FIXED. Call submit now.")
    elif issues_remaining:
        lines.append("\nFix the next issue from the list above.")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────

def call_llm(client: OpenAI, step: int, observation_text: str) -> str:
    user_prompt = (
        f"Step {step}. Current state:\n\n"
        f"{observation_text}\n\n"
        f"Respond with ONE JSON action only:"
    )

    try:
        completion = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature = TEMPERATURE,
            max_tokens  = MAX_TOKENS,
            stream      = False,
        )
        text = (completion.choices[0].message.content or "").strip()
        print(f"[DEBUG] LLM: {text[:120]!r}", flush=True)
        return text if text else '{"operation":"submit","column":"","row_uid":-1,"value":""}'

    except Exception as exc:
        print(f"[DEBUG] LLM failed step {step}: {exc}", flush=True)
        return '{"operation":"submit","column":"","row_uid":-1,"value":""}'


# ─────────────────────────────────────────────
# SINGLE TASK RUNNER
# ─────────────────────────────────────────────

def run_task(client: OpenAI, task_id: str, seed: int = BASELINE_SEED) -> float:
    # Import here so an ImportError gives a clear message instead of crashing
    try:
        from client import CRMSanitizerEnv, CRMAction
    except ImportError as e:
        print(f"[ERROR] Could not import client module: {e}", flush=True)
        print("[ERROR] Make sure client.py is in the same directory.", flush=True)
        log_end(success=False, steps=0, score=0.01, rewards=[])
        return 0.01

    max_steps   = MAX_STEPS[task_id]
    rewards: List[float] = []
    steps_taken = 0
    score       = 0.01
    success     = False
    history: List[str] = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        with CRMSanitizerEnv(base_url=SERVER_URL) as env:

            result = env.reset(task_id=task_id, seed=seed)
            obs    = result.observation

            total_issues = getattr(obs, "total_issues", 0) or 0
            print(
                f"[DEBUG] '{task_id}' started. "
                f"Issues: {total_issues}. "
                f"Max steps: {max_steps}",
                flush=True,
            )

            for step in range(1, max_steps + 1):

                if getattr(result, "done", False):
                    break

                obs_text    = format_observation(obs, history)
                raw         = call_llm(client, step, obs_text)
                action_dict = parse_action(raw)

                action = CRMAction(
                    operation = action_dict["operation"],
                    column    = action_dict["column"],
                    row_uid   = action_dict["row_uid"],
                    value     = action_dict["value"],
                    reason    = action_dict["reason"],
                )

                result = env.step(action)
                obs    = result.observation

                reward      = getattr(result, "reward", 0.0) or 0.0
                done        = getattr(result, "done", False)
                rewards.append(reward)
                steps_taken = step

                last_result = getattr(obs, "last_action_result", "")
                action_summary = (
                    f"{action.operation}("
                    f"uid={action.row_uid},"
                    f"col={action.column!r},"
                    f"val={action.value!r})"
                )

                log_step(
                    step   = step,
                    action = action_summary,
                    reward = reward,
                    done   = done,
                    error  = None,
                )

                history.append(
                    f"Step {step}: {action.operation} "
                    f"uid={action.row_uid} "
                    f"\u2192 reward={reward:+.2f} "
                    f"({last_result})"
                )

                if done:
                    break

            obs_total = getattr(obs, "total_issues", None) or total_issues or 1
            max_reward   = (obs_total * 0.15) + 0.60
            total_reward = sum(rewards)
            # Clamp to strictly open interval (0.01, 0.99)
            # Validator rejects exact 0.0 and exact 1.0
            score        = max(0.01, min(0.99, total_reward / max_reward))
            success      = score >= SUCCESS_THRESHOLD

    except ConnectionError as e:
        print(f"[ERROR] Cannot connect: {e}", flush=True)
        print(f"[ERROR] Start server: cd server && python app.py", flush=True)

    except Exception as e:
        print(f"[ERROR] Task {task_id}: {e}", flush=True)
        traceback.print_exc()

    finally:
        log_end(
            success = success,
            steps   = steps_taken,
            score   = score,
            rewards = rewards,
        )

    return score


# ─────────────────────────────────────────────
# SERVER READINESS CHECK
# ─────────────────────────────────────────────

def wait_for_server(server_url: str, max_wait: int = 60) -> bool:
    """
    Polls the server until it responds or timeout is reached.
    Falls back gracefully if the client module lacks wait_until_ready.
    """
    try:
        from client import CRMSanitizerEnv
        probe = CRMSanitizerEnv(base_url=server_url)
        # Use built-in method if available, otherwise poll manually
        if hasattr(probe, "wait_until_ready"):
            ready = probe.wait_until_ready(max_wait=max_wait)
            probe.close()
            return ready
        else:
            probe.close()
            # Manual poll fallback
            import urllib.request
            deadline = time.time() + max_wait
            while time.time() < deadline:
                try:
                    urllib.request.urlopen(server_url, timeout=3)
                    return True
                except Exception:
                    time.sleep(2)
            return False
    except ImportError as e:
        print(f"[ERROR] Could not import client module: {e}", flush=True)
        return False
    except Exception as e:
        print(f"[ERROR] Server check failed: {e}", flush=True)
        return False


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:

    # ── Parse arguments ──
    parser = argparse.ArgumentParser(
        description="CRM Sanitizer baseline inference script"
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help=(
            "Which task to run. "
            "easy=easy_basic_fix, "
            "medium=medium_format_dedup, "
            "hard=hard_full_audit, "
            "all=run all three (default)"
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BASELINE_SEED,
        help=f"Random seed (default: {BASELINE_SEED})"
    )
    args = parser.parse_args()

    # Map short names to task IDs
    task_map = {
        "easy":   "easy_basic_fix",
        "medium": "medium_format_dedup",
        "hard":   "hard_full_audit",
    }

    if args.task == "all":
        tasks_to_run = ALL_TASKS
    else:
        tasks_to_run = [task_map[args.task]]

    # ── Header ──
    print("=" * 60, flush=True)
    print("CRM Sanitizer — Baseline Inference", flush=True)
    print(f"Model:  {MODEL_NAME}", flush=True)
    print(f"Server: {SERVER_URL}", flush=True)
    print(f"Seed:   {args.seed}", flush=True)
    print(f"Tasks:  {tasks_to_run}", flush=True)
    print("=" * 60, flush=True)

    if not API_KEY:
        print(
            "[ERROR] HF_TOKEN not set.\n"
            "Run: $env:HF_TOKEN='hf_your_token_here'",
            flush=True,
        )
        sys.exit(1)

    print(f"\n[INFO] Endpoint: {API_BASE_URL}", flush=True)
    print(f"[INFO] Model: {MODEL_NAME}\n", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # ── Check server ──
    print("[INFO] Checking server...", flush=True)
    if not wait_for_server(SERVER_URL, max_wait=60):
        print(f"[ERROR] Server not responding at {SERVER_URL}", flush=True)
        sys.exit(1)
    print("[INFO] Server ready.\n", flush=True)

    # ── Run tasks ──
    scores: Dict[str, float] = {}

    for task_id in tasks_to_run:
        print(f"\n{'─' * 60}", flush=True)
        print(f"Running: {task_id}", flush=True)
        print(f"{'─' * 60}", flush=True)

        score           = run_task(client, task_id, seed=args.seed)
        scores[task_id] = score

        print(f"\n[RESULT] {task_id}: {score:.4f}", flush=True)

        # Pause between tasks to avoid rate limiting
        if task_id != tasks_to_run[-1]:
            print("[INFO] Pausing 5s between tasks...", flush=True)
            time.sleep(5)

    # ── Summary ──
    print(f"\n{'=' * 60}", flush=True)
    print("BASELINE SCORES", flush=True)
    print(f"{'=' * 60}", flush=True)

    for task_id, score in scores.items():
        bar    = "█" * int(score * 20)
        spaces = "░" * (20 - int(score * 20))
        status = "✓ PASS" if score >= SUCCESS_THRESHOLD else "✗ FAIL"
        print(
            f"{task_id:<25} [{bar}{spaces}] "
            f"{score:.4f}  {status}",
            flush=True,
        )

    if len(scores) > 1:
        avg = sum(scores.values()) / len(scores)
        print(f"\n{'─' * 60}", flush=True)
        print(f"{'AVERAGE':<25}  {avg:.4f}", flush=True)

    print(f"{'=' * 60}", flush=True)

    all_passed = all(s >= SUCCESS_THRESHOLD for s in scores.values())
    if all_passed:
        print("\n✓ All tasks passed.", flush=True)
        sys.exit(0)
    else:
        failed = [t for t, s in scores.items() if s < SUCCESS_THRESHOLD]
        print(f"\n✗ Failed: {failed}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL] Unhandled exception: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)