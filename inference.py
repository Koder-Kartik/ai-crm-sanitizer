# inference.py
# CRM Sanitizer — Baseline Inference Script
#
# Uses OpenAI client pointed at Hugging Face Inference API
# This is the correct approach for HF-compatible endpoints
#
# SETUP:
#   export HF_TOKEN=hf_your_token_here
#   export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
#   export API_BASE_URL=https://api-inference.huggingface.co/v1
#
# Then run:
#   python inference.py

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
# ─────────────────────────────────────────────

API_BASE_URL = os.environ.get(
    "API_BASE_URL",
    "https://api-inference.huggingface.co/v1"   # HF OpenAI-compatible endpoint
)

MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "Qwen/Qwen2.5-72B-Instruct"   # Best free model on HF with chat support
)

API_KEY = os.environ.get(
    "HF_TOKEN",
    os.environ.get("OPENAI_API_KEY", "")
)

SERVER_URL       = os.environ.get("SERVER_URL", "http://localhost:7860")
BENCHMARK        = "crm-sanitizer-v1"
BASELINE_SEED    = 42
SUCCESS_THRESHOLD = 0.40
TEMPERATURE      = 0.01   # near-zero but not zero — HF models sometimes reject 0.0
MAX_TOKENS       = 512    # keep small — HF free tier has limits

MAX_STEPS = {
    "easy_basic_fix":      15,
    "medium_format_dedup": 20,
    "hard_full_audit":     30,
}


# ─────────────────────────────────────────────
# REQUIRED LOG FUNCTIONS
# DO NOT change these — judges parse them exactly
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
        f"action={action!r} "
        f"reward={reward:.4f} "
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
    rewards_str = json.dumps([round(r, 4) for r in rewards])
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} "
        f"score={score:.4f} "
        f"rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# Kept short — HF free models have context limits
# Very explicit JSON instructions — no assumptions
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a CRM data cleaning agent.
You will see a table with customer data that has quality issues.
Your job is to fix all issues by responding with ONE JSON action at a time.

RESPOND WITH ONLY A JSON OBJECT. NO OTHER TEXT. NO EXPLANATION.
DO NOT wrap in markdown. DO NOT add ```json. JUST the raw JSON.

AVAILABLE ACTIONS:

Fix missing value:
{"operation":"fill_missing","column":"email","row_uid":1003,"value":"john@co.com","reason":"was null"}

Remove duplicate row:
{"operation":"remove_duplicate","column":"","row_uid":1099,"value":"","reason":"duplicate"}

Fix phone format (use format: (XXX) XXX-XXXX):
{"operation":"standardize_format","column":"phone","row_uid":1008,"value":"(555) 123-4567","reason":"bad format"}

Fix wrong value:
{"operation":"fix_value","column":"loyalty_points","row_uid":1006,"value":"250","reason":"was negative"}

Check a column before fixing:
{"operation":"get_column_stats","column":"phone","row_uid":-1,"value":"","reason":"exploring"}

Fix entire column format at once:
{"operation":"bulk_fix_column","column":"city","row_uid":-1,"value":"New York","reason":"all cities need fix"}

Flag ambiguous data:
{"operation":"flag_ambiguous","column":"email","row_uid":1042,"value":"FLAGGED","reason":"two valid options"}

Submit when done:
{"operation":"submit","column":"","row_uid":-1,"value":"","reason":"done"}

RULES:
- Only ONE JSON object per response
- Use row uid numbers from the uid column
- Missing values shown as: MISSING
- Phone format must be: (XXX) XXX-XXXX
- Date format must be: YYYY-MM-DD
- City names must be Title Case
- loyalty_points must be >= 0
- Submit only when all issues are fixed
"""


# ─────────────────────────────────────────────
# ROBUST ACTION PARSER
# Handles all the ways HF models format output
# Never crashes — always returns valid action
# ─────────────────────────────────────────────

# All valid operations — anything else gets rejected
VALID_OPERATIONS = {
    "fill_missing",
    "remove_duplicate",
    "standardize_format",
    "fix_value",
    "get_column_stats",
    "bulk_fix_column",
    "flag_ambiguous",
    "submit",
}

def parse_action(text: str) -> Dict[str, Any]:
    """
    Parse LLM text into action dict.
    Handles every messy format HF models produce.
    """
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

    # ── Step 1: Strip markdown fences ──
    # Some models wrap in ```json ... ``` or ``` ... ```
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```",             "", text)
    text = text.strip()

    # ── Step 2: Extract first JSON object ──
    # Models sometimes add explanation text before or after
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not json_match:
        # Try broader match with nested braces
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
    
    if json_match:
        text = json_match.group(0)
    else:
        print(f"[DEBUG] No JSON found in: {text[:100]!r}", flush=True)
        return safe_default

    # ── Step 3: Parse JSON ──
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try fixing common JSON errors
        # Single quotes instead of double quotes
        text = text.replace("'", '"')
        # Trailing commas
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            print(f"[DEBUG] JSON parse failed: {text[:100]!r}", flush=True)
            return safe_default

    # ── Step 4: Extract and validate fields ──
    operation = str(data.get("operation", "submit")).strip().lower()

    # If model returned invalid operation name try to fix it
    if operation not in VALID_OPERATIONS:
        # Try partial match
        for valid_op in VALID_OPERATIONS:
            if valid_op in operation or operation in valid_op:
                operation = valid_op
                break
        else:
            operation = "submit"

    # Safe extraction of all fields
    try:
        row_uid = int(data.get("row_uid", data.get("uid", -1)))
    except (ValueError, TypeError):
        row_uid = -1

    return {
        "operation": operation,
        "column":    str(data.get("column",  "")).strip(),
        "row_uid":   row_uid,
        "value":     str(data.get("value",   "")).strip(),
        "reason":    str(data.get("reason",  "")).strip(),
    }


# ─────────────────────────────────────────────
# OBSERVATION FORMATTER
# Kept concise — HF models have smaller context
# ─────────────────────────────────────────────

def format_observation(obs) -> str:
    """
    Format observation into concise text for HF models.
    Smaller than before — fits in limited context windows.
    """
    lines = []

    lines.append(f"TASK: {obs.task_description}")
    lines.append(f"Progress: {obs.issues_fixed}/{obs.total_issues} fixed | Step {obs.step_number}")

    if obs.last_action_result:
        lines.append(f"Last result: {obs.last_action_result}")

    # Hints
    if obs.issues_remaining:
        lines.append("\nISSUES TO FIX:")
        for hint in obs.issues_remaining:
            lines.append(f"  - {hint}")

    # Column stats
    if obs.column_stats and not obs.column_stats.get("error"):
        s = obs.column_stats
        lines.append(
            f"\nSTATS for '{s.get('column','')}': "
            f"nulls={s.get('null_count',0)} "
            f"unique={s.get('unique_count',0)} "
            f"samples={s.get('sample_values',[])}"
        )

    lines.append(f"\nTABLE:\n{obs.table_markdown}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# LLM CALL
# OpenAI client pointed at HF endpoint
# ─────────────────────────────────────────────

def call_llm(
    client: OpenAI,
    step: int,
    observation_text: str,
    history: List[str],
) -> str:
    """
    Call LLM via OpenAI client pointed at HF endpoint.
    Returns raw text. Falls back to submit on any error.
    """
    # Keep history short — HF models have context limits
    history_text = ""
    if history:
        recent = history[-3:]   # only last 3 steps
        history_text = "\nRECENT ACTIONS:\n" + "\n".join(recent)

    user_prompt = (
        f"Step {step}.\n\n"
        f"{observation_text}"
        f"{history_text}\n\n"
        f"Respond with ONE JSON action only:"
    )

    try:
        completion = client.chat.completions.create(
            model      = MODEL_NAME,
            messages   = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature = TEMPERATURE,
            max_tokens  = MAX_TOKENS,
            stream      = False,
        )

        text = (completion.choices[0].message.content or "").strip()
        print(f"[DEBUG] LLM raw response: {text[:150]!r}", flush=True)
        return text if text else '{"operation":"submit","column":"","row_uid":-1,"value":""}'

    except Exception as exc:
        print(f"[DEBUG] LLM call failed at step {step}: {exc}", flush=True)
        return '{"operation":"submit","column":"","row_uid":-1,"value":""}'


# ─────────────────────────────────────────────
# SINGLE TASK RUNNER
# ─────────────────────────────────────────────

def run_task(
    client: OpenAI,
    task_id: str,
    seed: int = BASELINE_SEED,
) -> float:
    """
    Run one complete episode. Returns score 0.0-1.0.
    """
    from client import CRMSanitizerEnv, CRMAction

    max_steps   = MAX_STEPS[task_id]
    rewards     = []
    steps_taken = 0
    score       = 0.0
    success     = False
    history     = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        with CRMSanitizerEnv(base_url=SERVER_URL) as env:

            result = env.reset(task_id=task_id, seed=seed)
            obs    = result.observation

            print(
                f"[DEBUG] Task '{task_id}' started. "
                f"Total issues: {obs.total_issues}. "
                f"Max steps: {max_steps}",
                flush=True,
            )

            for step in range(1, max_steps + 1):

                if result.done:
                    break

                obs_text     = format_observation(obs)
                raw_response = call_llm(client, step, obs_text, history)
                action_dict  = parse_action(raw_response)

                action = CRMAction(
                    operation = action_dict["operation"],
                    column    = action_dict["column"],
                    row_uid   = action_dict["row_uid"],
                    value     = action_dict["value"],
                    reason    = action_dict["reason"],
                )

                result = env.step(action)
                obs    = result.observation

                reward = result.reward or 0.0
                done   = result.done
                rewards.append(reward)
                steps_taken = step

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
                    f"uid={action.row_uid} col={action.column!r} "
                    f"→ {obs.last_action_result} "
                    f"(reward {reward:+.2f})"
                )

                if done:
                    break

            # Final score calculation
            total_issues = obs.total_issues or 1
            max_reward   = (total_issues * 0.15) + 0.60
            total_reward = sum(rewards)
            score        = max(0.0, min(1.0, total_reward / max_reward))
            success      = score >= SUCCESS_THRESHOLD

    except ConnectionError as e:
        print(f"[ERROR] Cannot connect to server: {e}", flush=True)
        print(f"[ERROR] Start server: cd server && python app.py", flush=True)

    except Exception as e:
        print(f"[ERROR] Task {task_id} failed: {e}", flush=True)
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
# MAIN
# ─────────────────────────────────────────────

def main() -> None:

    print("=" * 60, flush=True)
    print("CRM Sanitizer — Baseline Inference", flush=True)
    print(f"Model:  {MODEL_NAME}", flush=True)
    print(f"Server: {SERVER_URL}", flush=True)
    print(f"Seed:   {BASELINE_SEED}", flush=True)
    print("=" * 60, flush=True)

    if not API_KEY:
        print(
            "[ERROR] HF_TOKEN not set.\n"
            "Run: export HF_TOKEN=hf_your_token_here",
            flush=True,
        )
        sys.exit(1)

    # Show which endpoint we are using
    print(f"\n[INFO] API endpoint: {API_BASE_URL}", flush=True)
    print(f"[INFO] Model: {MODEL_NAME}\n", flush=True)

    # Create OpenAI client pointed at HF
    client = OpenAI(
        base_url = API_BASE_URL,
        api_key  = API_KEY,
    )

    # Wait for environment server
    print("[INFO] Checking server health...", flush=True)
    from client import CRMSanitizerEnv
    probe = CRMSanitizerEnv(base_url=SERVER_URL)
    if not probe.wait_until_ready(max_wait=60):
        print(
            f"[ERROR] Server at {SERVER_URL} not responding.\n"
            f"Start it: cd server && python app.py",
            flush=True,
        )
        sys.exit(1)
    probe.close()
    print("[INFO] Server is ready.\n", flush=True)

    # Run all three tasks
    tasks  = ["easy_basic_fix", "medium_format_dedup", "hard_full_audit"]
    scores = {}

    for task_id in tasks:
        print(f"\n{'─' * 60}", flush=True)
        print(f"Running task: {task_id}", flush=True)
        print(f"{'─' * 60}", flush=True)

        score          = run_task(client, task_id, seed=BASELINE_SEED)
        scores[task_id] = score

        print(f"\n[RESULT] {task_id}: {score:.4f}", flush=True)
        time.sleep(3)   # small pause between tasks

    # Summary
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

    avg = sum(scores.values()) / len(scores)
    print(f"\n{'─' * 60}", flush=True)
    print(f"{'AVERAGE SCORE':<25}  {avg:.4f}", flush=True)
    print(f"{'=' * 60}", flush=True)

    if all(s >= SUCCESS_THRESHOLD for s in scores.values()):
        print("\n✓ All tasks passed.", flush=True)
        sys.exit(0)
    else:
        failed = [t for t, s in scores.items() if s < SUCCESS_THRESHOLD]
        print(f"\n✗ Failed: {failed}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()