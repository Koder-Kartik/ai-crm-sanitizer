# server/tasks.py
# CRM Sanitizer — Procedural Dataset Generator
#
# This file generates dirty CRM datasets for all three tasks.
# Every dataset is seeded — same seed always produces same dataset.
# The ground truth clean table is stored internally for grading.
#
# HOW IT WORKS:
#   1. Start with a clean fictional CRM table
#   2. Inject specific issues based on task difficulty
#   3. Store what the clean version should look like
#   4. Return both dirty table and ground truth to the environment

import random
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# CRM SCHEMA
# Every row in our CRM table has these fields
# ─────────────────────────────────────────────

CRM_COLUMNS = [
    "uid",           # permanent unique ID — never changes
    "name",          # customer full name
    "email",         # email address
    "phone",         # phone number
    "company",       # company name
    "city",          # city name
    "join_date",     # date customer joined
    "loyalty_points" # reward points (must be >= 0)
]


# ─────────────────────────────────────────────
# DATA POOLS
# Realistic fake data the generator picks from
# ─────────────────────────────────────────────

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer",
    "Michael", "Linda", "William", "Barbara", "David", "Susan",
    "Richard", "Jessica", "Joseph", "Sarah", "Thomas", "Karen",
    "Charles", "Lisa", "Christopher", "Nancy", "Daniel", "Betty",
    "Matthew", "Margaret", "Anthony", "Sandra", "Mark", "Ashley"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
    "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
    "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore",
    "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
    "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson"
]

COMPANIES = [
    "Acme Corp", "Globex Inc", "Initech", "Umbrella Ltd", "Hooli",
    "Pied Piper", "Dunder Mifflin", "Vandelay Industries", "Prestige Worldwide",
    "Sterling Cooper", "Massive Dynamic", "Soylent Corp", "Gekko & Co",
    "Bluth Company", "Cyberdyne Systems", "Oscorp Industries", "Wayne Enterprises",
    "Stark Industries", "LexCorp", "Momcorp"
]

CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte",
    "Indianapolis", "San Francisco", "Seattle", "Denver", "Nashville"
]

# Messy versions of city names — used to inject inconsistency
CITY_MESS = {
    "New York":      ["new york", "NEW YORK", "New york", "newyork"],
    "Los Angeles":   ["los angeles", "LA", "L.A.", "los Angeles"],
    "Chicago":       ["chicago", "CHICAGO", "Chigago"],
    "Houston":       ["houston", "HOUSTON", "Huston"],
    "Phoenix":       ["phoenix", "PHOENIX", "Pheonix"],
    "San Francisco": ["san francisco", "SF", "S.F.", "san Francisco"],
    "Seattle":       ["seattle", "SEATTLE", "Seatle"],
    "Denver":        ["denver", "DENVER", "Denvor"],
    "Nashville":     ["nashville", "NASHVILLE", "Nasville"],
    "Dallas":        ["dallas", "DALLAS", "Dalas"],
}

# Phone formats — standard and messy versions
PHONE_FORMATS_CLEAN = "({area}) {exchange}-{number}"

def make_phone_clean(rng: random.Random) -> str:
    area     = rng.randint(200, 999)
    exchange = rng.randint(200, 999)
    number   = rng.randint(1000, 9999)
    return f"({area}) {exchange}-{number}"

def make_phone_messy(rng: random.Random) -> str:
    area     = rng.randint(200, 999)
    exchange = rng.randint(200, 999)
    number   = rng.randint(1000, 9999)
    # Pick a random messy format
    fmt = rng.choice([
        f"{area}-{exchange}-{number}",
        f"{area}.{exchange}.{number}",
        f"+1{area}{exchange}{number}",
        f"{area}{exchange}{number}",
        f"({area}){exchange}-{number}",
    ])
    return fmt

# Date formats
def make_date_clean(rng: random.Random) -> str:
    year  = rng.randint(2015, 2023)
    month = rng.randint(1, 12)
    day   = rng.randint(1, 28)
    return f"{year}-{month:02d}-{day:02d}"  # ISO format: 2021-03-15

def make_date_messy(rng: random.Random) -> str:
    year  = rng.randint(2015, 2023)
    month = rng.randint(1, 12)
    day   = rng.randint(1, 28)
    fmt = rng.choice([
        f"{month}/{day}/{year}",           # US format: 3/15/2021
        f"{day}-{month}-{year}",           # EU format: 15-3-2021
        f"{month}-{day}-{str(year)[2:]}",  # Short: 3-15-21
        f"{day}/{month}/{year}",           # EU slash: 15/3/2021
    ])
    return fmt

def make_email(name: str, company: str, rng: random.Random) -> str:
    first = name.split()[0].lower()
    last  = name.split()[1].lower() if len(name.split()) > 1 else "user"
    domain = company.lower().replace(" ", "").replace("&", "and")[:10]
    suffix = rng.choice(["com", "net", "org", "io"])
    sep    = rng.choice([".", "_", ""])
    return f"{first}{sep}{last}@{domain}.{suffix}"


# ─────────────────────────────────────────────
# ISSUE TRACKING
# Every injected problem is tracked precisely
# so the grader knows exactly what needs fixing
# ─────────────────────────────────────────────

@dataclass
class IssueRecord:
    """
    Represents one data quality issue in the dataset.
    The grader uses these records to score the agent.
    """
    uid: int                    # which row has the issue
    column: str                 # which column has the issue
    issue_type: str             # what kind of issue it is
    dirty_value: Any            # what the value looks like now
    clean_value: Any            # what it should be
    is_ambiguous: bool = False  # True for the 2 hard task ambiguous cases
    acceptable_values: List[Any] = field(default_factory=list)
    # ^ for ambiguous cases: list of values that are all acceptable


# ─────────────────────────────────────────────
# CLEAN DATASET GENERATOR
# Creates a perfect CRM table first
# Issues are injected on top of this
# ─────────────────────────────────────────────

def generate_clean_table(
    num_rows: int,
    rng: random.Random,
    start_uid: int = 1001
) -> List[Dict[str, Any]]:
    """
    Generate a clean, perfect CRM table with no issues.
    This is the ground truth we inject problems into.
    """
    rows = []
    used_emails = set()

    for i in range(num_rows):
        uid     = start_uid + i
        name    = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        company = rng.choice(COMPANIES)

        # Make unique email
        email = make_email(name, company, rng)
        counter = 0
        while email in used_emails:
            email = f"{email.split('@')[0]}{counter}@{email.split('@')[1]}"
            counter += 1
        used_emails.add(email)

        row = {
            "uid":            uid,
            "name":           name,
            "email":          email,
            "phone":          make_phone_clean(rng),
            "company":        company,
            "city":           rng.choice(CITIES),
            "join_date":      make_date_clean(rng),
            "loyalty_points": rng.randint(0, 5000),
        }
        rows.append(row)

    return rows


# ─────────────────────────────────────────────
# ISSUE INJECTORS
# Each function injects one specific type of problem
# Returns the modified row and an IssueRecord
# ─────────────────────────────────────────────

def inject_missing_value(
    row: Dict[str, Any],
    column: str,
    rng: random.Random
) -> Tuple[Dict[str, Any], IssueRecord]:
    """Replace a value with None (missing)."""
    dirty_row = copy.deepcopy(row)
    clean_value = dirty_row[column]
    dirty_row[column] = None

    issue = IssueRecord(
        uid=row["uid"],
        column=column,
        issue_type="missing_value",
        dirty_value=None,
        clean_value=clean_value,
    )
    return dirty_row, issue


def inject_phone_format(
    row: Dict[str, Any],
    rng: random.Random
) -> Tuple[Dict[str, Any], IssueRecord]:
    """Replace a clean phone with a messy format."""
    dirty_row = copy.deepcopy(row)
    clean_value = dirty_row["phone"]

    # Parse clean phone numbers like "(555) 123-4567"
    # Extract digits and rebuild in a messy format
    digits = "".join(c for c in clean_value if c.isdigit())
    if len(digits) >= 10:
        area     = digits[0:3]
        exchange = digits[3:6]
        number   = digits[6:10]
        messy = rng.choice([
            f"{area}-{exchange}-{number}",
            f"{area}.{exchange}.{number}",
            f"+1{area}{exchange}{number}",
            f"({area}){exchange}-{number}",
        ])
    else:
        messy = make_phone_messy(rng)

    dirty_row["phone"] = messy
    issue = IssueRecord(
        uid=row["uid"],
        column="phone",
        issue_type="phone_format",
        dirty_value=messy,
        clean_value=clean_value,
    )
    return dirty_row, issue


def inject_city_case(
    row: Dict[str, Any],
    rng: random.Random
) -> Tuple[Dict[str, Any], IssueRecord]:
    """Replace a clean city name with an inconsistent version."""
    dirty_row = copy.deepcopy(row)
    city = dirty_row["city"]
    clean_value = city

    if city in CITY_MESS:
        messy = rng.choice(CITY_MESS[city])
    else:
        messy = city.lower()

    dirty_row["city"] = messy
    issue = IssueRecord(
        uid=row["uid"],
        column="city",
        issue_type="city_format",
        dirty_value=messy,
        clean_value=clean_value,
    )
    return dirty_row, issue


def inject_date_format(
    row: Dict[str, Any],
    rng: random.Random
) -> Tuple[Dict[str, Any], IssueRecord]:
    """Replace a clean ISO date with a messy format."""
    dirty_row = copy.deepcopy(row)
    clean_value = dirty_row["join_date"]

    # Parse ISO date and reformat
    try:
        parts = clean_value.split("-")
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        messy = rng.choice([
            f"{month}/{day}/{year}",
            f"{day}-{month}-{year}",
            f"{month}-{day}-{str(year)[2:]}",
        ])
    except Exception:
        messy = make_date_messy(rng)

    dirty_row["join_date"] = messy
    issue = IssueRecord(
        uid=row["uid"],
        column="join_date",
        issue_type="date_format",
        dirty_value=messy,
        clean_value=clean_value,
    )
    return dirty_row, issue


def inject_negative_points(
    row: Dict[str, Any],
    rng: random.Random
) -> Tuple[Dict[str, Any], IssueRecord]:
    """Set loyalty_points to a negative value (out of range)."""
    dirty_row = copy.deepcopy(row)
    clean_value = dirty_row["loyalty_points"]
    messy = rng.choice([-100, -500, -1, -999])

    dirty_row["loyalty_points"] = messy
    issue = IssueRecord(
        uid=row["uid"],
        column="loyalty_points",
        issue_type="negative_value",
        dirty_value=messy,
        clean_value=clean_value,
    )
    return dirty_row, issue


def inject_duplicate_row(
    rows: List[Dict[str, Any]],
    source_uid: int,
    new_uid: int,
    rng: random.Random
) -> Tuple[Dict[str, Any], IssueRecord]:
    """
    Create a duplicate of an existing row with a new uid.
    The duplicate should be removed — the original is kept.
    """
    source = next(r for r in rows if r["uid"] == source_uid)
    duplicate = copy.deepcopy(source)
    duplicate["uid"] = new_uid

    # Slightly vary the name to make it realistic
    # (real duplicates often have small typos)
    name_parts = duplicate["name"].split()
    if len(name_parts) == 2:
        variations = [
            duplicate["name"],
            duplicate["name"].lower(),
            f"{name_parts[0][0]}. {name_parts[1]}",
        ]
        duplicate["name"] = rng.choice(variations)

    issue = IssueRecord(
        uid=new_uid,
        column="uid",
        issue_type="duplicate_row",
        dirty_value=new_uid,
        clean_value="REMOVE",  # grader expects this row to be deleted
    )
    return duplicate, issue


def inject_ambiguous_email(
    row: Dict[str, Any],
    rng: random.Random
) -> Tuple[Dict[str, Any], IssueRecord]:
    """
    HARD TASK ONLY — Ambiguous Case 1
    Row has two plausible emails. Agent must pick one
    OR flag as ambiguous. Both actions are acceptable.
    """
    dirty_row = copy.deepcopy(row)
    original_email = dirty_row["email"]

    # Create a second plausible email for the same person
    name_parts = dirty_row["name"].lower().split()
    first = name_parts[0] if name_parts else "user"
    last  = name_parts[1] if len(name_parts) > 1 else "x"
    alt_email = f"{last}.{first}@gmail.com"

    # Store both in a string to signal conflict
    dirty_row["email"] = f"{original_email} | {alt_email}"

    issue = IssueRecord(
        uid=row["uid"],
        column="email",
        issue_type="ambiguous_email",
        dirty_value=dirty_row["email"],
        clean_value=original_email,        # preferred answer
        is_ambiguous=True,
        acceptable_values=[
            original_email,                # keep original
            alt_email,                     # use alternative
            "FLAGGED",                     # flag_ambiguous action
        ]
    )
    return dirty_row, issue


def inject_ambiguous_age_zero(
    row: Dict[str, Any],
    rng: random.Random
) -> Tuple[Dict[str, Any], IssueRecord]:
    """
    HARD TASK ONLY — Ambiguous Case 2
    loyalty_points = 0. Could be legitimately zero
    OR could be a missing value. Both interpretations valid.
    """
    dirty_row = copy.deepcopy(row)
    original_points = dirty_row["loyalty_points"]
    dirty_row["loyalty_points"] = 0

    issue = IssueRecord(
        uid=row["uid"],
        column="loyalty_points",
        issue_type="ambiguous_zero",
        dirty_value=0,
        clean_value=original_points,
        is_ambiguous=True,
        acceptable_values=[
            0,           # keep as zero (valid choice)
            None,        # treat as missing (valid choice)
            "FLAGGED",   # flag_ambiguous (valid choice)
        ]
    )
    return dirty_row, issue


# ─────────────────────────────────────────────
# TASK GENERATORS
# One function per task — each returns
# (dirty_table, clean_table, issues, task_config)
# ─────────────────────────────────────────────

@dataclass
class TaskData:
    """
    Everything the environment needs to run one episode.
    Generated once on reset(), stored for the whole episode.
    """
    task_id: str
    task_name: str
    task_description: str
    dirty_table: List[Dict[str, Any]]   # what agent sees
    clean_table: List[Dict[str, Any]]   # ground truth
    issues: List[IssueRecord]           # every injected problem
    max_steps: int
    hint_level: str                     # "full", "partial", or "none"
    seed: int


def generate_easy_task(seed: int) -> TaskData:
    """
    EASY TASK — Basic CRM Audit & Fix
    - 10 rows
    - 3 missing values, 1 duplicate, 1 phone format issue
    - Full hints provided (agent sees exact list of issues)
    - Max 15 steps
    """
    rng = random.Random(seed)
    clean_table = generate_clean_table(10, rng, start_uid=1001)
    dirty_table = copy.deepcopy(clean_table)
    issues: List[IssueRecord] = []

    # Issue 1: Missing email on row 0
    dirty_table[0], issue = inject_missing_value(dirty_table[0], "email", rng)
    issues.append(issue)

    # Issue 2: Missing phone on row 3
    dirty_table[3], issue = inject_missing_value(dirty_table[3], "phone", rng)
    issues.append(issue)

    # Issue 3: Missing loyalty_points on row 6
    dirty_table[6], issue = inject_missing_value(dirty_table[6], "loyalty_points", rng)
    issues.append(issue)

    # Issue 4: Messy phone format on row 8
    dirty_table[8], issue = inject_phone_format(dirty_table[8], rng)
    issues.append(issue)

    # Issue 5: Duplicate of row 2, appended at end
    dup_uid = 1099
    duplicate, issue = inject_duplicate_row(dirty_table, 1003, dup_uid, rng)
    dirty_table.append(duplicate)
    issues.append(issue)

    return TaskData(
        task_id="easy_basic_fix",
        task_name="Easy — Basic CRM Audit & Fix",
        task_description=(
            "You are a CRM data agent. This 10-row customer table has "
            "3 missing values, 1 phone format issue, and 1 duplicate entry. "
            "Fix all issues and submit when done. "
            "Hints: The issues_remaining field lists every problem."
        ),
        dirty_table=dirty_table,
        clean_table=clean_table,
        issues=issues,
        max_steps=15,
        hint_level="full",
        seed=seed,
    )


def generate_medium_task(seed: int) -> TaskData:
    """
    MEDIUM TASK — Format Standardization & Deduplication
    - 25 rows
    - Mixed date formats, inconsistent cities, 3 duplicates
    - Partial hints: only column names given, not specific rows
    - Max 20 steps
    """
    rng = random.Random(seed)
    clean_table = generate_clean_table(25, rng, start_uid=2001)
    dirty_table = copy.deepcopy(clean_table)
    issues: List[IssueRecord] = []

    # Issues: messy dates on rows 1, 5, 11, 18
    for idx in [1, 5, 11, 18]:
        dirty_table[idx], issue = inject_date_format(dirty_table[idx], rng)
        issues.append(issue)

    # Issues: inconsistent city names on rows 2, 7, 14, 20
    for idx in [2, 7, 14, 20]:
        dirty_table[idx], issue = inject_city_case(dirty_table[idx], rng)
        issues.append(issue)

    # Issues: 3 duplicate rows
    for source_uid, new_uid in [(2003, 2091), (2010, 2092), (2017, 2093)]:
        dup, issue = inject_duplicate_row(dirty_table, source_uid, new_uid, rng)
        dirty_table.append(dup)
        issues.append(issue)

    return TaskData(
        task_id="medium_format_dedup",
        task_name="Medium — Format Standardization & Deduplication",
        task_description=(
            "You are a CRM data agent. This 25-row customer table has "
            "mixed date formats, inconsistent city name casing, and duplicate entries. "
            "Fix all issues and submit when done. "
            "Hint: Affected columns are listed in issues_remaining. "
            "You must identify which specific rows are affected."
        ),
        dirty_table=dirty_table,
        clean_table=clean_table,
        issues=issues,
        max_steps=20,
        hint_level="partial",
        seed=seed,
    )


def generate_hard_task(seed: int) -> TaskData:
    """
    HARD TASK — Full Audit, No Hints
    - 40 rows
    - All issue types combined + 2 ambiguous cases
    - No hints — agent must discover everything
    - Max 30 steps
    """
    rng = random.Random(seed)
    clean_table = generate_clean_table(40, rng, start_uid=3001)
    dirty_table = copy.deepcopy(clean_table)
    issues: List[IssueRecord] = []

    # Missing values — scattered across table
    for idx, col in [(2, "email"), (8, "phone"), (15, "loyalty_points"), (22, "email")]:
        dirty_table[idx], issue = inject_missing_value(dirty_table[idx], col, rng)
        issues.append(issue)

    # Phone format issues
    for idx in [4, 12, 25, 33]:
        dirty_table[idx], issue = inject_phone_format(dirty_table[idx], rng)
        issues.append(issue)

    # City inconsistencies
    for idx in [6, 17, 28, 36]:
        dirty_table[idx], issue = inject_city_case(dirty_table[idx], rng)
        issues.append(issue)

    # Date format issues
    for idx in [9, 19, 31]:
        dirty_table[idx], issue = inject_date_format(dirty_table[idx], rng)
        issues.append(issue)

    # Negative loyalty points
    for idx in [11, 23]:
        dirty_table[idx], issue = inject_negative_points(dirty_table[idx], rng)
        issues.append(issue)

    # Duplicate rows
    for source_uid, new_uid in [(3005, 3091), (3018, 3092), (3029, 3093)]:
        dup, issue = inject_duplicate_row(dirty_table, source_uid, new_uid, rng)
        dirty_table.append(dup)
        issues.append(issue)

    # AMBIGUOUS CASE 1: conflicting email on row 35
    dirty_table[35], issue = inject_ambiguous_email(dirty_table[35], rng)
    issues.append(issue)

    # AMBIGUOUS CASE 2: loyalty_points = 0 on row 38
    dirty_table[38], issue = inject_ambiguous_age_zero(dirty_table[38], rng)
    issues.append(issue)

    return TaskData(
        task_id="hard_full_audit",
        task_name="Hard — Full Audit, No Hints",
        task_description=(
            "You are a CRM data agent. This 40-row customer table has "
            "multiple data quality issues of various types. "
            "No hints are provided — you must discover all issues yourself. "
            "Use get_column_stats to explore columns before fixing. "
            "Some rows may have ambiguous data with no single correct answer — "
            "use flag_ambiguous when you encounter genuine uncertainty. "
            "Fix all issues you can find and submit when done."
        ),
        dirty_table=dirty_table,
        clean_table=clean_table,
        issues=issues,
        max_steps=30,
        hint_level="none",
        seed=seed,
    )


# ─────────────────────────────────────────────
# TASK REGISTRY
# Maps task_id to generator function
# ─────────────────────────────────────────────

TASK_GENERATORS = {
    "easy_basic_fix":      generate_easy_task,
    "medium_format_dedup": generate_medium_task,
    "hard_full_audit":     generate_hard_task,
}

def generate_task(task_id: str, seed: int) -> TaskData:
    """
    Main entry point. Environment calls this on every reset().
    Returns a fresh TaskData with dirty table and ground truth.
    """
    if task_id not in TASK_GENERATORS:
        raise ValueError(
            f"Unknown task_id: '{task_id}'. "
            f"Choose from: {list(TASK_GENERATORS.keys())}"
        )
    return TASK_GENERATORS[task_id](seed)

def get_column_stats(table, column):
    """
    Returns basic statistics for a column:
    - total values
    - missing count
    - unique values
    - sample values
    """
    values = [row.get(column) for row in table]

    total = len(values)
    missing = sum(1 for v in values if v is None)
    non_null = [v for v in values if v is not None]

    unique_values = list(set(non_null))

    return {
        "column": column,
        "total_rows": total,
        "missing_values": missing,
        "unique_count": len(unique_values),
        "sample_values": unique_values[:5],
    }