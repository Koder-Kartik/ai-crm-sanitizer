# 🧹 CRM Sanitizer

> **CRM Sanitizer** is an OpenEnv benchmark where AI agents clean, audit,
> and reason over messy customer data — the task every revenue operations
> team does manually, every single day.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-1.0.0-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Why CRM Sanitizer?

Every company that has ever used Salesforce, HubSpot, or any CRM tool
has the same problem: the data gets messy. Fast.

- Sales reps enter phone numbers in five different formats
- The same customer gets imported twice from two different sources
- Date fields come from three different countries with three different conventions
- City names are typed by hand: "new york", "NEW YORK", "New york", "newyork"
- Loyalty points go negative after a bad data migration

This is not a toy problem. Data cleaning consumes an estimated **80% of a
data analyst's time** in real organizations. CRM Sanitizer turns this
daily pain point into a rigorous benchmark for evaluating how well AI
agents can reason over structured, imperfect data.

---

## Environment Overview

CRM Sanitizer is a fully OpenEnv-compliant environment. An AI agent
receives a dirty CRM table rendered as Markdown, takes cleaning actions
one step at a time, and is rewarded for fixing real data quality issues.

### What Makes It Different

- **Procedural generation** — every `reset()` call with a new seed
  produces a fresh dirty dataset. Agents must generalize, not memorize.
- **Tiered difficulty** — hints decrease as tasks get harder, forcing
  agents to discover issues independently on the hardest task.
- **Ambiguity handling** — the hard task includes genuinely ambiguous
  cases with no single correct answer, testing decision-making under
  uncertainty.
- **Permanent row IDs** — all actions reference stable `uid` values,
  not row indices, preventing index-shifting bugs after deletions.

---

## CRM Schema

Every table in CRM Sanitizer uses this schema:

| Column | Type | Description |
|--------|------|-------------|
| `uid` | integer | Permanent unique ID — never changes |
| `name` | string | Customer full name |
| `email` | string | Email address |
| `phone` | string | Phone number |
| `company` | string | Company name |
| `city` | string | City name |
| `join_date` | string | Date customer joined (ISO: YYYY-MM-DD) |
| `loyalty_points` | integer | Reward points (must be >= 0) |

---

## Action Space

The agent can take 8 actions per step:

| Operation | Description | Example |
|-----------|-------------|---------|
| `fill_missing` | Fill a null cell with a value | Fill missing email for uid 1003 |
| `remove_duplicate` | Remove a duplicate row by uid | Remove duplicate uid 1099 |
| `standardize_format` | Fix format on one row | Fix phone format for uid 1008 |
| `fix_value` | Correct a wrong value | Fix negative loyalty_points for uid 1006 |
| `get_column_stats` | Explore a column before acting | Get stats on phone column |
| `bulk_fix_column` | Fix all format issues in a column | Standardize all city names |
| `flag_ambiguous` | Flag a row with no single correct answer | Flag conflicting emails on uid 1042 |
| `submit` | Declare task complete | Submit when all issues resolved |

### Action Model
```python
class CRMAction(Action):
    operation: str    # one of the 8 operations above
    column:    str    # target column name
    row_uid:   int    # permanent uid (-1 for column-level ops)
    value:     str    # new value or operation parameter
    reason:    str    # optional explanation (logged, not graded)
```

---

## Observation Space

After every action the agent receives:

| Field | Type | Description |
|-------|------|-------------|
| `table_markdown` | string | Current CRM table as Markdown |
| `issues_remaining` | list | Hints about remaining issues (tiered by difficulty) |
| `issues_fixed` | int | Number of issues correctly fixed so far |
| `total_issues` | int | Total issues in this episode |
| `last_action_result` | string | Feedback from the last action |
| `task_description` | string | Human-readable task description |
| `step_number` | int | Current step number |
| `column_stats` | dict or null | Stats returned by get_column_stats |
| `done` | bool | True when episode is finished |
| `reward` | float | Reward from last action |

---

## Tasks

### Task 1 — Easy: Basic CRM Audit & Fix

| Property | Value |
|----------|-------|
| Rows | 10 |
| Issues | 3 missing values, 1 phone format, 1 duplicate |
| Hints | Full — exact issue descriptions provided |
| Max steps | 15 |
| Expected score (strong LLM) | 0.80 — 1.00 |

The agent is told exactly what is wrong. It must execute the correct
sequence of actions to fix each issue and submit.

---

### Task 2 — Medium: Format Standardization & Deduplication

| Property | Value |
|----------|-------|
| Rows | 25 |
| Issues | 4 date formats, 4 city inconsistencies, 3 duplicates |
| Hints | Partial — only affected column names provided |
| Max steps | 20 |
| Expected score (strong LLM) | 0.55 — 0.80 |

The agent knows which columns have problems but must identify
the specific rows itself. Tests systematic column-level reasoning.

---

### Task 3 — Hard: Full Audit, No Hints

| Property | Value |
|----------|-------|
| Rows | 40 |
| Issues | 4 missing values, 4 phone formats, 4 city inconsistencies, 3 date formats, 2 negative values, 3 duplicates, 2 ambiguous cases |
| Hints | None — agent must discover everything |
| Max steps | 30 |
| Expected score (strong LLM) | 0.30 — 0.55 |

No hints. The agent must use `get_column_stats` to explore,
identify all issues independently, handle two ambiguous cases
with no single correct answer, and submit.

The two ambiguous cases:
- **Conflicting email** — same customer uid appears with two
  different plausible email addresses
- **Zero loyalty points** — could be legitimately zero or a
  missing value; both interpretations are valid

---

## Reward Function

| Event | Reward |
|-------|--------|
| Correctly fix a real issue | +0.15 |
| Fix something that was not broken | -0.08 |
| Redundant action (already fixed) | -0.03 |
| Submit with all issues resolved | +0.60 |
| Submit with issues remaining | -0.40 |
| get_column_stats (exploration) | 0.00 |

**Final score formula:**

final_score = clamp(total_reward / max_possible_reward, 0.0, 1.0)
max_possible_reward = (num_issues × 0.15) + 0.60

Documented per task:

| Task | Issues | Max Reward |
|------|--------|------------|
| Easy | 5 | 1.35 |
| Medium | 11 | 2.25 |
| Hard | 22 | 3.90 |

---

## Baseline Scores

Baseline agent: `Qwen/Qwen2.5-72B-Instruct`
Seed: `42` (fixed for reproducibility)

| Task | Score | Status |
|------|-------|--------|
| easy_basic_fix | 1.00 | ✓ PASS |
| medium_format_dedup | 1.00 | ✓ PASS |
| hard_full_audit | 0.35 | ✓ PASS |
| **Average** | **0.78** | ✓ |

---

## Setup & Usage

### Requirements

- Python 3.11+
- Docker
- A Hugging Face account with inference permissions

### Install Dependencies
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/crm-sanitizer
cd crm-sanitizer
pip install -r requirements.txt
```

### Run Server Locally
```bash
cd server
python app.py
```

Server starts at `http://localhost:7860`

- Web UI: `http://localhost:7860/web`
- API Docs: `http://localhost:7860/docs`
- Health: `http://localhost:7860/health`

### Run With Docker
```bash
# Build
docker build -t crm-sanitizer -f server/Dockerfile .

# Run
docker run -p 7860:7860 crm-sanitizer
```

### Run Baseline Inference
```bash
# Set credentials
export HF_TOKEN=hf_your_token_here
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1

# Run
python inference.py
```

### Use the Python Client
```python
from client import CRMSanitizerEnv, CRMAction

with CRMSanitizerEnv(base_url="http://localhost:7860") as env:

    # Start episode
    result = env.reset(task_id="easy_basic_fix", seed=42)
    obs = result.observation

    print(obs.table_markdown)
    print(obs.issues_remaining)

    # Take action
    result = env.step(CRMAction(
        operation = "fill_missing",
        column    = "email",
        row_uid   = 1001,
        value     = "john@company.com",
        reason    = "email was null",
    ))

    print(f"Reward: {result.reward}")
    print(f"Done: {result.done}")
```

---

## Project Structure

crm-sanitizer/
├── inference.py          ← Baseline inference script
├── openenv.yaml          ← OpenEnv manifest
├── requirements.txt      ← Python dependencies
├── models.py             ← Typed Action, Observation, State models
├── client.py             ← Python client
├── README.md             ← This file
└── server/
├── app.py            ← FastAPI server + Web UI
├── environment.py    ← reset() / step() / state() logic
├── tasks.py          ← Procedural dataset generator
├── grader.py         ← Scoring engine
└── Dockerfile        ← Container definition

---

## Validate Before Submitting
```bash
pip install openenv-core
openenv validate
```

Expected output:
✓ openenv.yaml found and valid
✓ Typed models found
✓ step/reset/state endpoints respond
✓ All 3 tasks have graders
✓ Baseline scores in range [0.0, 1.0]
All checks passed.

---

## License

MIT License. See LICENSE file.

---

## Author

Built for the OpenEnv Environment Hackathon.
Domain: Customer CRM Data Quality
Environment: CRM Sanitizer v1.0.0