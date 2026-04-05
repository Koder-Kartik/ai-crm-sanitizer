# server/app.py
# CRM Sanitizer — FastAPI Web Server
#
# This file wraps the CRMEnvironment in a FastAPI server.
# It exposes all endpoints required by the OpenEnv spec:
#
#   GET  /health        → health check (judges ping this first)
#   POST /reset         → start new episode
#   POST /step          → take one action
#   GET  /state         → get episode metadata
#   GET  /docs          → auto-generated API documentation
#   GET  /web           → interactive web UI for human judges
#   WS   /ws            → WebSocket for persistent sessions
#
# The server creates one CRMEnvironment instance per WebSocket
# connection — this allows multiple agents to run simultaneously
# without interfering with each other.

import json
import os
import sys
import traceback
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import ValidationError

# Add parent directory to path so server can import models.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import CRMEnvironment
from models import CRMAction


# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="CRM Sanitizer",
    description=(
        "CRM Sanitizer: An OpenEnv benchmark where AI agents clean, "
        "audit, and reason over messy customer data — the task every "
        "revenue operations team does manually, every single day."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# One global environment for HTTP endpoints
# WebSocket endpoints create their own per-connection
_http_env = CRMEnvironment()


# ─────────────────────────────────────────────
# HEALTH CHECK
# First thing judges check
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Health check endpoint.
    Returns 200 if server is running correctly.
    Judges ping this before running any evaluation.
    """
    return JSONResponse({
        "status": "healthy",
        "environment": "crm-sanitizer",
        "version": "1.0.0",
        "tasks": [
            "easy_basic_fix",
            "medium_format_dedup",
            "hard_full_audit",
        ]
    })


# ─────────────────────────────────────────────
# HTTP ENDPOINTS
# Stateless — each request is independent
# Used for simple testing and validation
# ─────────────────────────────────────────────

@app.post("/reset")
async def reset(body: Dict[str, Any] = {}):
    """
    Reset the environment and start a new episode.

    Body (all optional):
        task_id:    "easy_basic_fix" | "medium_format_dedup" | "hard_full_audit"
        seed:       integer (default 42)
        episode_id: string

    Returns:
        CRMObservation as JSON
    """
    try:
        task_id    = body.get("task_id", "easy_basic_fix")
        seed       = body.get("seed", 42)
        episode_id = body.get("episode_id", None)

        obs = _http_env.reset(
            task_id=task_id,
            seed=seed,
            episode_id=episode_id,
        )
        return JSONResponse(obs.model_dump())

    except Exception as e:
        return JSONResponse(
            {"error": str(e), "trace": traceback.format_exc()},
            status_code=500,
        )


@app.post("/step")
async def step(body: Dict[str, Any] = {}):
    """
    Take one action in the environment.

    Body:
        operation:  One of the 8 valid operations
        column:     Target column name
        row_uid:    Target row uid (-1 for column-level operations)
        value:      New value or operation parameter
        reason:     Optional explanation (logged, not graded)

    Returns:
        CRMObservation as JSON with reward and updated table
    """
    try:
        action = CRMAction(
            operation = body.get("operation", "submit"),
            column    = body.get("column", ""),
            row_uid   = body.get("row_uid", -1),
            value     = body.get("value", ""),
            reason    = body.get("reason", ""),
        )
        obs = _http_env.step(action)
        return JSONResponse(obs.model_dump())

    except ValidationError as e:
        return JSONResponse(
            {"error": "Invalid action format", "detail": str(e)},
            status_code=422,
        )
    except Exception as e:
        return JSONResponse(
            {"error": str(e), "trace": traceback.format_exc()},
            status_code=500,
        )


@app.get("/state")
async def state():
    """
    Get current episode metadata.

    Returns:
        CRMState as JSON
    """
    try:
        s = _http_env.state()
        return JSONResponse(s.model_dump())
    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500,
        )


# ─────────────────────────────────────────────
# WEBSOCKET ENDPOINT
# Persistent session — one env per connection
# Used by the OpenEnv Python client
# ─────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for persistent agent sessions.

    Each connection gets its own CRMEnvironment instance.
    This means multiple agents can run simultaneously
    without interfering with each other.

    Message format (JSON):
        {"type": "reset", "task_id": "...", "seed": 42}
        {"type": "step",  "operation": "...", "column": "...", ...}
        {"type": "state"}
    """
    await websocket.accept()

    # Fresh environment for this connection
    env = CRMEnvironment()

    try:
        while True:
            # Wait for message from client
            raw = await websocket.receive_text()

            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "error": "Invalid JSON"
                }))
                continue

            msg_type = message.get("type", "")

            # ── Handle reset ──
            if msg_type == "reset":
                try:
                    obs = env.reset(
                        task_id    = message.get("task_id", "easy_basic_fix"),
                        seed       = message.get("seed", 42),
                        episode_id = message.get("episode_id", None),
                    )
                    await websocket.send_text(json.dumps({
                        "type":        "observation",
                        "observation": obs.model_dump(),
                        "reward":      obs.reward,
                        "done":        obs.done,
                    }))
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "error": f"reset failed: {str(e)}"
                    }))

            # ── Handle step ──
            elif msg_type == "step":
                try:
                    action = CRMAction(
                        operation = message.get("operation", "submit"),
                        column    = message.get("column", ""),
                        row_uid   = message.get("row_uid", -1),
                        value     = message.get("value", ""),
                        reason    = message.get("reason", ""),
                    )
                    obs = env.step(action)
                    await websocket.send_text(json.dumps({
                        "type":        "observation",
                        "observation": obs.model_dump(),
                        "reward":      obs.reward,
                        "done":        obs.done,
                    }))
                except ValidationError as e:
                    await websocket.send_text(json.dumps({
                        "error": f"Invalid action: {str(e)}"
                    }))
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "error": f"step failed: {str(e)}"
                    }))

            # ── Handle state ──
            elif msg_type == "state":
                try:
                    s = env.state()
                    await websocket.send_text(json.dumps({
                        "type":  "state",
                        "state": s.model_dump(),
                    }))
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "error": f"state failed: {str(e)}"
                    }))

            # ── Unknown message type ──
            else:
                await websocket.send_text(json.dumps({
                    "error": f"Unknown message type: '{msg_type}'. "
                             f"Use 'reset', 'step', or 'state'."
                }))

    except WebSocketDisconnect:
        # Client disconnected cleanly — nothing to do
        pass
    except Exception as e:
        # Unexpected error — log it
        print(f"[WebSocket Error] {e}", flush=True)
        traceback.print_exc()


# ─────────────────────────────────────────────
# WEB UI
# Interactive interface for human judges
# Judges click through this during Phase 3
# ─────────────────────────────────────────────

@app.get("/web", response_class=HTMLResponse)
async def web_ui():
    """
    Interactive web UI for human judges.
    Shows the current task, table, and allows manual actions.
    Automatically renders the Markdown table nicely.
    """
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CRM Sanitizer — OpenEnv</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont,
                         'Segoe UI', Roboto, sans-serif;
            background: #0f1117;
            color: #e0e0e0;
            padding: 24px;
            min-height: 100vh;
        }

        h1 {
            font-size: 1.6rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 4px;
        }

        .subtitle {
            font-size: 0.9rem;
            color: #888;
            margin-bottom: 24px;
        }

        .card {
            background: #1a1d27;
            border: 1px solid #2a2d3a;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 16px;
        }

        .card h2 {
            font-size: 1rem;
            font-weight: 600;
            color: #a0aec0;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .controls {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }

        select, input {
            background: #252836;
            border: 1px solid #3a3d4a;
            border-radius: 6px;
            color: #e0e0e0;
            padding: 8px 12px;
            font-size: 0.9rem;
        }

        button {
            background: #4f46e5;
            border: none;
            border-radius: 6px;
            color: white;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 600;
            padding: 8px 18px;
            transition: background 0.2s;
        }

        button:hover { background: #4338ca; }

        button.danger {
            background: #dc2626;
        }
        button.danger:hover { background: #b91c1c; }

        button.success {
            background: #16a34a;
        }
        button.success:hover { background: #15803d; }

        /* Progress bar */
        .progress-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }

        .progress-bar-bg {
            flex: 1;
            background: #252836;
            border-radius: 999px;
            height: 10px;
            overflow: hidden;
        }

        .progress-bar-fill {
            height: 100%;
            background: #4f46e5;
            border-radius: 999px;
            transition: width 0.3s ease;
        }

        .progress-label {
            font-size: 0.85rem;
            color: #a0aec0;
            min-width: 100px;
            text-align: right;
        }

        /* Stats row */
        .stats {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 12px;
        }

        .stat {
            background: #252836;
            border-radius: 8px;
            padding: 8px 14px;
            font-size: 0.85rem;
        }

        .stat span {
            color: #4f46e5;
            font-weight: 700;
        }

        /* Table */
        .table-wrapper {
            overflow-x: auto;
            max-height: 400px;
            overflow-y: auto;
        }

        table {
            border-collapse: collapse;
            width: 100%;
            font-size: 0.85rem;
        }

        th {
            background: #252836;
            color: #a0aec0;
            font-weight: 600;
            padding: 10px 14px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 1;
        }

        td {
            padding: 8px 14px;
            border-bottom: 1px solid #2a2d3a;
        }

        tr:hover td { background: #1e2130; }

        .missing {
            color: #ef4444;
            font-weight: 600;
            background: rgba(239,68,68,0.08);
            padding: 2px 8px;
            border-radius: 4px;
        }

        /* Hints */
        .hint-list {
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .hint-list li {
            background: #252836;
            border-left: 3px solid #f59e0b;
            border-radius: 0 6px 6px 0;
            padding: 6px 12px;
            font-size: 0.85rem;
            color: #fcd34d;
        }

        /* Result message */
        #result-msg {
            font-size: 0.9rem;
            padding: 10px 14px;
            border-radius: 6px;
            background: #252836;
            border-left: 3px solid #4f46e5;
            margin-top: 10px;
        }

        #result-msg.error  { border-color: #ef4444; color: #fca5a5; }
        #result-msg.success{ border-color: #22c55e; color: #86efac; }
        #result-msg.reward { border-color: #f59e0b; color: #fcd34d; }

        .score-big {
            font-size: 2rem;
            font-weight: 800;
            color: #4f46e5;
        }
    </style>
</head>
<body>

<h1>🧹 CRM Sanitizer</h1>
<p class="subtitle">
    OpenEnv benchmark — AI agents clean messy customer data
</p>

<!-- Task Setup -->
<div class="card">
    <h2>Task Setup</h2>
    <div class="controls">
        <select id="task-select">
            <option value="easy_basic_fix">Easy — Basic Audit & Fix</option>
            <option value="medium_format_dedup">Medium — Format & Deduplication</option>
            <option value="hard_full_audit">Hard — Full Audit, No Hints</option>
        </select>
        <input type="number" id="seed-input" value="42"
               placeholder="Seed" style="width:100px">
        <button onclick="doReset()">▶ Start Episode</button>
    </div>
    <div id="task-desc" style="font-size:0.85rem;color:#888;"></div>
</div>

<!-- Progress -->
<div class="card" id="progress-card" style="display:none">
    <h2>Progress</h2>
    <div class="stats">
        <div class="stat">Step <span id="stat-step">0</span></div>
        <div class="stat">Fixed <span id="stat-fixed">0</span></div>
        <div class="stat">Remaining <span id="stat-remaining">0</span></div>
        <div class="stat">Reward <span id="stat-reward">0.00</span></div>
    </div>
    <div class="progress-row">
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" id="progress-fill"
                 style="width:0%"></div>
        </div>
        <div class="progress-label" id="progress-label">0 / 0</div>
    </div>
</div>

<!-- Hints -->
<div class="card" id="hints-card" style="display:none">
    <h2>⚠ Issues Remaining</h2>
    <ul class="hint-list" id="hints-list"></ul>
</div>

<!-- CRM Table -->
<div class="card" id="table-card" style="display:none">
    <h2>Current CRM Table</h2>
    <div class="table-wrapper">
        <table id="crm-table">
            <thead id="crm-thead"></thead>
            <tbody id="crm-tbody"></tbody>
        </table>
    </div>
</div>

<!-- Actions -->
<div class="card" id="actions-card" style="display:none">
    <h2>Take Action</h2>
    <div class="controls">
        <select id="op-select">
            <option value="fill_missing">fill_missing</option>
            <option value="remove_duplicate">remove_duplicate</option>
            <option value="standardize_format">standardize_format</option>
            <option value="fix_value">fix_value</option>
            <option value="get_column_stats">get_column_stats</option>
            <option value="bulk_fix_column">bulk_fix_column</option>
            <option value="flag_ambiguous">flag_ambiguous</option>
        </select>
        <input type="number" id="uid-input"
               placeholder="row uid" style="width:110px">
        <input type="text"   id="col-input"
               placeholder="column name" style="width:140px">
        <input type="text"   id="val-input"
               placeholder="value" style="width:160px">
        <input type="text"   id="reason-input"
               placeholder="reason (optional)" style="width:200px">
        <button onclick="doStep()">→ Step</button>
        <button class="success" onclick="doSubmit()">✓ Submit</button>
    </div>
    <div id="result-msg"></div>
</div>

<script>
// ── State ──
let currentObs = null;
let totalReward = 0;

// ── Reset ──
async function doReset() {
    const taskId = document.getElementById('task-select').value;
    const seed   = parseInt(document.getElementById('seed-input').value) || 42;
    totalReward  = 0;

    const resp = await fetch('/reset', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({task_id: taskId, seed: seed}),
    });

    const obs = await resp.json();
    currentObs = obs;
    renderObs(obs, null);

    showCards();
    setResult('Episode started. Use the action panel below.', 'success');
}

// ── Step ──
async function doStep() {
    const op     = document.getElementById('op-select').value;
    const uid    = parseInt(document.getElementById('uid-input').value) || -1;
    const col    = document.getElementById('col-input').value.trim();
    const val    = document.getElementById('val-input').value.trim();
    const reason = document.getElementById('reason-input').value.trim();

    const resp = await fetch('/step', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            operation: op,
            row_uid:   uid,
            column:    col,
            value:     val,
            reason:    reason,
        }),
    });

    const obs = await resp.json();
    currentObs = obs;

    if (obs.reward !== null && obs.reward !== undefined) {
        totalReward += obs.reward;
    }

    renderObs(obs, obs.reward);

    const msgClass = obs.reward > 0 ? 'success'
                   : obs.reward < 0 ? 'error'
                   : 'reward';
    setResult(obs.last_action_result, msgClass);

    if (obs.done) {
        setResult(
            obs.last_action_result +
            ' | Episode complete!',
            'success'
        );
    }
}

// ── Submit ──
async function doSubmit() {
    const resp = await fetch('/step', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({operation: 'submit', column: '', row_uid: -1, value: ''}),
    });
    const obs = await resp.json();
    renderObs(obs, obs.reward);
    setResult(obs.last_action_result, obs.reward >= 0 ? 'success' : 'error');
}

// ── Render observation ──
function renderObs(obs, lastReward) {
    // Stats
    document.getElementById('stat-step').textContent      = obs.step_number  || 0;
    document.getElementById('stat-fixed').textContent     = obs.issues_fixed || 0;
    document.getElementById('stat-remaining').textContent =
        (obs.total_issues || 0) - (obs.issues_fixed || 0);
    document.getElementById('stat-reward').textContent    =
        totalReward.toFixed(2);

    // Progress bar
    const total  = obs.total_issues || 1;
    const fixed  = obs.issues_fixed || 0;
    const pct    = Math.round((fixed / total) * 100);
    document.getElementById('progress-fill').style.width  = pct + '%';
    document.getElementById('progress-label').textContent = fixed + ' / ' + total;

    // Task description
    document.getElementById('task-desc').textContent = obs.task_description || '';

    // Hints
    const hintsList = document.getElementById('hints-list');
    hintsList.innerHTML = '';
    const hints = obs.issues_remaining || [];
    if (hints.length > 0) {
        document.getElementById('hints-card').style.display = 'block';
        hints.forEach(h => {
            const li = document.createElement('li');
            li.textContent = h;
            hintsList.appendChild(li);
        });
    } else {
        document.getElementById('hints-card').style.display = 'none';
    }

    // Column stats (if returned)
    if (obs.column_stats) {
        const s = obs.column_stats;
        const statsText = s.error ? s.error :
            `${s.column}: ${s.null_count} nulls, ` +
            `${s.unique_count} unique, ` +
            `samples: [${(s.sample_values || []).join(', ')}]`;
        setResult('Column Stats — ' + statsText, 'reward');
    }

    // Parse and render markdown table
    renderMarkdownTable(obs.table_markdown || '');
}

// ── Parse markdown table and render as HTML ──
function renderMarkdownTable(md) {
    const lines = md.trim().split('\\n').filter(l => l.trim().startsWith('|'));
    if (lines.length < 2) return;

    const parseRow = line =>
        line.split('|').slice(1, -1).map(c => c.trim());

    const headers = parseRow(lines[0]);
    const rows    = lines.slice(2).map(parseRow);  // skip separator

    const thead = document.getElementById('crm-thead');
    const tbody = document.getElementById('crm-tbody');

    thead.innerHTML = '<tr>' +
        headers.map(h => `<th>${h}</th>`).join('') + '</tr>';

    tbody.innerHTML = rows.map(row =>
        '<tr>' + row.map(cell => {
            const isMissing = cell.includes('MISSING');
            return `<td>${isMissing
                ? `<span class="missing">${cell}</span>`
                : cell}</td>`;
        }).join('') + '</tr>'
    ).join('');
}

// ── Show cards after first reset ──
function showCards() {
    ['progress-card','table-card','actions-card']
        .forEach(id => {
            document.getElementById(id).style.display = 'block';
        });
}

// ── Set result message ──
function setResult(msg, cls) {
    const el = document.getElementById('result-msg');
    el.textContent  = msg;
    el.className    = cls || '';
}
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"[CRM Sanitizer] Starting server on {host}:{port}", flush=True)
    print(f"[CRM Sanitizer] Web UI:    http://{host}:{port}/web",  flush=True)
    print(f"[CRM Sanitizer] API Docs:  http://{host}:{port}/docs", flush=True)
    print(f"[CRM Sanitizer] Health:    http://{host}:{port}/health", flush=True)

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        workers=1,
        log_level="info",
    )

@app.get("/")
def root():
    return {
        "message": "CRM Sanitizer API is running 🚀",
        "docs": "/docs",
        "health": "/health",
        "web": "/web"
    }