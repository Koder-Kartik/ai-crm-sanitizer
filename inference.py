# inference.py
# CRM Sanitizer — Baseline Inference Script
#
# SETUP (PowerShell):
#   $env:HF_TOKEN="hf_your_token_here"
#   $env:API_BASE_URL="https://router.huggingface.co/v1"
#   $env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
#   $env:SERVER_URL="http://localhost:7860"
#
# Run ALL tasks:    python inference.py
# Run ONE task:     python inference.py --task easy
#                   python inference.py --task medium
#                   python inference.py --task hard

import argparse
import json
import os
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

# ── Guard: openai must be installed ──
try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai package not installed. Run: pip install openai", flush=True)
    sys.exit(1)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

API_BASE_URL = os.environ.get(
    "API_BASE_URL",
    "https://router.huggingface.co/v1"
)

MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "Qwen/Qwen2.5-72B-Instruct"
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
# LOG FUNCTIONS — judges parse these exactly
# ─────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step:   int,
    action: str,
    reward: float,
    done:   bool,
    error:  Optional[str],
) -> None:
    print(
        f"[STEP] step={step} "
        f"action={action} "
        f"reward={reward:.2f} "
        f"done={str(done).lower()} "
        f"error={error if error else 'null'}",
        flush=True,
    )


def log_end(
    success: bool,
    steps:   int,
    score:   float,
    rewards: List[float],
) -> None:
    # Spec: rewards=0.15,0.15,0.60  (no brackets, 2dp)
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
- get_column_stats: explore a column (use before fixing hard task)
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
9. Do NOT repeat an action on the same uid+column"""


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
    safe = {"operation": "submit", "column": "", "row_uid": -1, "value": "", "reason": ""}

    if not text or not text.strip():
        return safe

    text = text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        print(f"[DEBUG] No JSON in: {text[:80]!r}", flush=True)
        return safe

    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        fixed = m.group(0).replace("'", '"')
        fixed = re.sub(r",\s*}", "}", fixed)
        try:
            data = json.loads(fixed)
        except json.JSONDecodeError:
            print(f"[DEBUG] JSON parse failed: {m.group(0)[:80]!r}", flush=True)
            return safe

    op = str(data.get("operation", "submit")).strip().lower()
    if op not in VALID_OPERATIONS:
        for v in VALID_OPERATIONS:
            if v in op or op in v:
                op = v
                break
        else:
            op = "submit"

    try:
        uid = int(data.get("row_uid", data.get("uid", -1)))
    except (ValueError, TypeError):
        uid = -1

    return {
        "operation": op,
        "column":    str(data.get("column", "")).strip(),
        "row_uid":   uid,
        "value":     str(data.get("value", "")).strip(),
        "reason":    str(data.get("reason", "")).strip(),
    }


# ─────────────────────────────────────────────
# OBSERVATION FORMATTER
# ─────────────────────────────────────────────

def format_observation(obs, history: List[str]) -> str:
    lines = []
    remaining = (obs.total_issues or 0) - (obs.issues_fixed or 0)

    lines.append(
        f"Progress: {obs.issues_fixed or 0} fixed, "
        f"{remaining} remaining out of {obs.total_issues or 0} total"
    )

    if obs.last_action_result and "Episode started" not in obs.last_action_result:
        lines.append(f"Last result: {obs.last_action_result}")

    if hasattr(obs, "recent_actions") and obs.recent_actions:
        lines.append("\nACTIONS ALREADY TAKEN — DO NOT REPEAT:")
        for a in obs.recent_actions:
            lines.append(f"  ✓ {a}")

    if obs.issues_remaining:
        lines.append("\nISSUES TO FIX:")
        for h in obs.issues_remaining:
            lines.append(f"  - {h}")
    else:
        lines.append("\nNo hints. Read the table carefully to find issues.")

    if obs.column_stats and not obs.column_stats.get("error"):
        s = obs.column_stats
        lines.append(
            f"\nColumn '{s.get('column','')}' stats: "
            f"nulls={s.get('null_count',0)}, "
            f"samples={s.get('sample_values',[])[:3]}"
        )

    if history:
        lines.append("\nRecent steps:")
        for h in history[-3:]:
            lines.append(f"  {h}")

    lines.append(f"\nTABLE:\n{obs.table_markdown}")

    if remaining == 0:
        lines.append("\nALL ISSUES FIXED. Call submit now.")
    elif obs.issues_remaining:
        lines.append("\nFix the next issue from the list above.")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────

def call_llm(client: OpenAI, step: int, obs_text: str) -> str:
    try:
        completion = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": (
                    f"Step {step}. Current state:\n\n{obs_text}\n\n"
                    f"Respond with ONE JSON action only:"
                )},
            ],
            temperature = TEMPERATURE,
            max_tokens  = MAX_TOKENS,
            stream      = False,
        )
        text = (completion.choices[0].message.content or "").strip()
        print(f"[DEBUG] LLM: {text[:120]!r}", flush=True)
        return text or '{"operation":"submit","column":"","row_uid":-1,"value":""}'
    except Exception as exc:
        print(f"[DEBUG] LLM failed step {step}: {exc}", flush=True)
        return '{"operation":"submit","column":"","row_uid":-1,"value":""}'


# ─────────────────────────────────────────────
# SCORE HELPER — strictly open interval
# ─────────────────────────────────────────────

def safe_score(total_reward: float, total_issues: int) -> float:
    import math
    max_reward = (max(total_issues, 1) * 0.15) + 0.60
    if max_reward <= 0:
        return 0.01
    raw = total_reward / max_reward
    if not math.isfinite(raw):
        return 0.01
    return float(max(0.01, min(0.99, raw)))


# ─────────────────────────────────────────────
# SINGLE TASK RUNNER
# ─────────────────────────────────────────────

def run_task(client: OpenAI, task_id: str, seed: int = BASELINE_SEED) -> float:
    # Import here so path issues surface clearly
    try:
        from client import CRMSanitizerEnv, CRMAction
    except ImportError as e:
        print(f"[ERROR] Cannot import client: {e}", flush=True)
        print("[ERROR] Make sure client.py is in the same directory as inference.py", flush=True)
        return 0.01

    max_steps   = MAX_STEPS[task_id]
    rewards:    List[float] = []
    steps_taken = 0
    score       = 0.01
    success     = False
    history:    List[str]  = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    env = None
    try:
        env = CRMSanitizerEnv(base_url=SERVER_URL)
        result = env.reset(task_id=task_id, seed=seed)
        obs    = result.observation

        print(
            f"[DEBUG] '{task_id}' started. "
            f"Issues: {obs.total_issues}. "
            f"Max steps: {max_steps}",
            flush=True,
        )

        for step in range(1, max_steps + 1):
            if result.done:
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

            reward = float(result.reward or 0.0)
            done   = bool(result.done)
            rewards.append(reward)
            steps_taken = step

            log_step(
                step   = step,
                action = (
                    f"{action.operation}("
                    f"uid={action.row_uid},"
                    f"col={action.column!r},"
                    f"val={action.value!r})"
                ),
                reward = reward,
                done   = done,
                error  = None,
            )

            history.append(
                f"Step {step}: {action.operation} uid={action.row_uid} "
                f"→ reward={reward:+.2f} ({obs.last_action_result})"
            )

            if done:
                break

        score   = safe_score(sum(rewards), obs.total_issues or 1)
        success = score >= SUCCESS_THRESHOLD

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.", flush=True)
        score = safe_score(sum(rewards), 1)

    except Exception as e:
        print(f"[ERROR] Task {task_id} failed: {e}", flush=True)
        traceback.print_exc()
        score = 0.01

    finally:
        # Always close env cleanly
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

        # Always emit [END] — even on exception
        log_end(
            success = success,
            steps   = steps_taken,
            score   = score,
            rewards = rewards,
        )

    return score


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CRM Sanitizer baseline inference")
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Task to run: easy | medium | hard | all (default: all)",
    )
    parser.add_argument(
        "--seed", type=int, default=BASELINE_SEED,
        help=f"Random seed (default: {BASELINE_SEED})",
    )
    args = parser.parse_args()

    task_map = {
        "easy":   "easy_basic_fix",
        "medium": "medium_format_dedup",
        "hard":   "hard_full_audit",
    }
    tasks_to_run = ALL_TASKS if args.task == "all" else [task_map[args.task]]

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

    # Create OpenAI client
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"[ERROR] Failed to create OpenAI client: {e}", flush=True)
        sys.exit(1)

    # Check server
    print("[INFO] Checking server...", flush=True)
    try:
        from client import CRMSanitizerEnv
        probe = CRMSanitizerEnv(base_url=SERVER_URL)
        if not probe.wait_until_ready(max_wait=60):
            print(f"[ERROR] Server not responding at {SERVER_URL}", flush=True)
            print("[ERROR] Start server: cd server && python app.py", flush=True)
            sys.exit(1)
        probe.close()
        print("[INFO] Server ready.\n", flush=True)
    except Exception as e:
        print(f"[ERROR] Server check failed: {e}", flush=True)
        sys.exit(1)

    # Run tasks
    scores: Dict[str, float] = {}

    for i, task_id in enumerate(tasks_to_run):
        print(f"\n{'─' * 60}", flush=True)
        print(f"Running: {task_id}", flush=True)
        print(f"{'─' * 60}", flush=True)

        try:
            score = run_task(client, task_id, seed=args.seed)
        except Exception as e:
            print(f"[ERROR] Unexpected error running {task_id}: {e}", flush=True)
            traceback.print_exc()
            score = 0.01

        scores[task_id] = score
        print(f"\n[RESULT] {task_id}: {score:.4f}", flush=True)

        # Pause between tasks to avoid rate limiting
        if i < len(tasks_to_run) - 1:
            print("[INFO] Pausing 5s...", flush=True)
            time.sleep(5)

    # Summary
    print(f"\n{'=' * 60}", flush=True)
    print("BASELINE SCORES", flush=True)
    print(f"{'=' * 60}", flush=True)

    for task_id, score in scores.items():
        bar    = "█" * int(score * 20)
        spaces = "░" * (20 - int(score * 20))
        status = "✓ PASS" if score >= SUCCESS_THRESHOLD else "✗ FAIL"
        print(f"{task_id:<25} [{bar}{spaces}] {score:.4f}  {status}", flush=True)

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
    main()