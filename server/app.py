# server/app.py
# CRM Sanitizer — FastAPI Web Server
# Now includes autonomous AI Agent mode in the web UI

import json
import os
import sys
import traceback
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.responses import RedirectResponse
from pydantic import ValidationError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import CRMEnvironment
from models import CRMAction

app = FastAPI(
    title="CRM Sanitizer",
    description="CRM Sanitizer: An OpenEnv benchmark where AI agents clean messy customer data.",
    version="1.0.0",
)

_http_env = CRMEnvironment()


# ─────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return JSONResponse({
        "status":      "healthy",
        "environment": "crm-sanitizer",
        "version":     "1.0.0",
        "tasks": [
            "easy_basic_fix",
            "medium_format_dedup",
            "hard_full_audit",
        ],
    })


# ─────────────────────────────────────────────
# HTTP ENDPOINTS
# ─────────────────────────────────────────────

@app.post("/reset")
async def reset(body: Dict[str, Any] = {}):
    try:
        obs = _http_env.reset(
            task_id    = body.get("task_id", "easy_basic_fix"),
            seed       = body.get("seed", 42),
            episode_id = body.get("episode_id"),
        )
        return JSONResponse(obs.model_dump())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/step")
async def step(body: Dict[str, Any] = {}):
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
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/state")
async def state():
    try:
        return JSONResponse(_http_env.state().model_dump())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
@app.get("/score")
async def score():
    """
    Returns current episode score strictly in (0.01, 0.99).
    Called by openenv validate to check score range.
    """
    try:
        if _http_env._grader is None:
            # No episode started yet — return minimum valid score
            return JSONResponse({"score": 0.01, "min": 0.01, "max": 0.99})
        s = _http_env._grader.final_score()
        return JSONResponse({"score": s, "min": 0.01, "max": 0.99})
    except Exception as e:
        return JSONResponse({"score": 0.01, "error": str(e)}, status_code=200)


# ─────────────────────────────────────────────
# WEBSOCKET
# ─────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    env = CRMEnvironment()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                continue

            msg_type = message.get("type", "")

            if msg_type == "reset":
                try:
                    obs = env.reset(
                        task_id    = message.get("task_id", "easy_basic_fix"),
                        seed       = message.get("seed", 42),
                        episode_id = message.get("episode_id"),
                    )
                    await websocket.send_text(json.dumps({
                        "type": "observation",
                        "observation": obs.model_dump(),
                        "reward": obs.reward,
                        "done":   obs.done,
                    }))
                except Exception as e:
                    await websocket.send_text(json.dumps({"error": str(e)}))

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
                        "type": "observation",
                        "observation": obs.model_dump(),
                        "reward": obs.reward,
                        "done":   obs.done,
                    }))
                except Exception as e:
                    await websocket.send_text(json.dumps({"error": str(e)}))

            elif msg_type == "state":
                try:
                    s = env.state()
                    await websocket.send_text(json.dumps({
                        "type":  "state",
                        "state": s.model_dump(),
                    }))
                except Exception as e:
                    await websocket.send_text(json.dumps({"error": str(e)}))
            else:
                await websocket.send_text(json.dumps({
                    "error": f"Unknown type: {msg_type}"
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS Error] {e}", flush=True)


# ─────────────────────────────────────────────
# WEB UI — Full AI Agent Frontend
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return RedirectResponse(url="/web")

@app.get("/web", response_class=HTMLResponse)
async def web_ui():
    html = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CRM Sanitizer — OpenEnv</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     background:#0f1117;color:#e0e0e0;padding:20px;min-height:100vh}
h1{font-size:1.5rem;font-weight:700;color:#fff;margin-bottom:2px}
.sub{font-size:.85rem;color:#666;margin-bottom:20px}
.card{background:#1a1d27;border:1px solid #2a2d3a;border-radius:10px;
      padding:18px;margin-bottom:14px}
.card h2{font-size:.8rem;font-weight:600;color:#6b7280;margin-bottom:12px;
         text-transform:uppercase;letter-spacing:.06em}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
select,input{background:#252836;border:1px solid #3a3d4a;border-radius:6px;
             color:#e0e0e0;padding:7px 11px;font-size:.875rem}
input[type=number]{width:90px}
input[type=password]{width:260px}
input[type=text]{width:200px}
button{border:none;border-radius:6px;color:#fff;cursor:pointer;
       font-size:.875rem;font-weight:600;padding:8px 16px;transition:background .15s}
.btn-primary{background:#4f46e5}.btn-primary:hover{background:#4338ca}
.btn-ai{background:#7c3aed}.btn-ai:hover{background:#6d28d9}
.btn-stop{background:#dc2626}.btn-stop:hover{background:#b91c1c}
.btn-success{background:#16a34a}.btn-success:hover{background:#15803d}
.btn-manual{background:#374151}.btn-manual:hover{background:#4b5563}

/* stats bar */
.stats{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px}
.stat{background:#252836;border-radius:7px;padding:7px 13px;font-size:.82rem}
.stat span{color:#818cf8;font-weight:700}
.pbar-bg{background:#252836;border-radius:999px;height:8px;overflow:hidden;margin-bottom:4px}
.pbar-fill{height:100%;background:#4f46e5;border-radius:999px;
           transition:width .4s ease}
.plabel{font-size:.78rem;color:#6b7280;text-align:right}

/* hints */
.hint-list{list-style:none;display:flex;flex-direction:column;gap:5px}
.hint-list li{background:#252836;border-left:3px solid #f59e0b;
              border-radius:0 5px 5px 0;padding:5px 11px;
              font-size:.82rem;color:#fcd34d}

/* table */
.tbl-wrap{overflow-x:auto;max-height:380px;overflow-y:auto}
table{border-collapse:collapse;width:100%;font-size:.82rem}
th{background:#252836;color:#9ca3af;font-weight:600;padding:9px 13px;
   text-align:left;position:sticky;top:0;z-index:1}
td{padding:7px 13px;border-bottom:1px solid #2a2d3a;transition:background .2s}
tr:hover td{background:#1e2130}
.cell-missing{color:#ef4444;font-weight:700;background:rgba(239,68,68,.1);
              padding:2px 7px;border-radius:4px}
.cell-fixed{animation:flash-green 1.2s ease}
.row-deleted{animation:flash-red .8s ease;pointer-events:none}
@keyframes flash-green{
  0%{background:rgba(34,197,94,.35)}
  100%{background:transparent}
}
@keyframes flash-red{
  0%{background:rgba(239,68,68,.35)}
  80%{background:rgba(239,68,68,.1);opacity:.4}
  100%{opacity:0}
}

/* agent log */
#agent-log{list-style:none;display:flex;flex-direction:column;gap:5px;
           max-height:260px;overflow-y:auto}
.log-item{display:flex;gap:8px;align-items:flex-start;
          background:#252836;border-radius:6px;padding:7px 11px;
          font-size:.8rem;animation:fadein .3s ease}
@keyframes fadein{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.log-step{color:#818cf8;font-weight:700;min-width:44px}
.log-op{color:#34d399;font-weight:600;min-width:130px}
.log-reward{font-weight:700;min-width:54px}
.log-reward.pos{color:#4ade80}.log-reward.neg{color:#f87171}.log-reward.zero{color:#6b7280}
.log-reason{color:#9ca3af;font-style:italic}
.log-thinking{color:#a78bfa;font-style:italic;opacity:.85}

/* result message */
#result-msg{font-size:.85rem;padding:9px 13px;border-radius:6px;
            background:#252836;border-left:3px solid #4f46e5;margin-top:10px}
#result-msg.ok{border-color:#22c55e;color:#86efac}
#result-msg.bad{border-color:#ef4444;color:#fca5a5}
#result-msg.info{border-color:#f59e0b;color:#fcd34d}

/* token section */
.token-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.token-hint{font-size:.78rem;color:#6b7280;margin-top:5px}
.badge{font-size:.72rem;padding:2px 8px;border-radius:999px;
       font-weight:600;margin-left:6px}
.badge-local{background:rgba(34,197,94,.15);color:#4ade80}
.badge-hf{background:rgba(129,140,248,.15);color:#818cf8}

/* done banner */
#done-banner{display:none;text-align:center;padding:14px;
             border-radius:8px;background:rgba(79,70,229,.15);
             border:1px solid #4f46e5;margin-bottom:14px}
#done-banner h3{font-size:1.1rem;color:#818cf8;margin-bottom:4px}
#done-score{font-size:2rem;font-weight:800;color:#4f46e5}
</style>
</head>
<body>

<h1>🧹 CRM Sanitizer</h1>
<p class="sub">OpenEnv benchmark — AI agents clean messy customer data</p>

<!-- DONE BANNER -->
<div id="done-banner">
  <h3>Episode Complete</h3>
  <div id="done-score">0.00</div>
  <div id="done-msg" style="color:#9ca3af;font-size:.85rem;margin-top:4px"></div>
</div>

<!-- TASK SETUP -->
<div class="card">
  <h2>Task Setup</h2>
  <div class="row">
    <select id="task-select">
      <option value="easy_basic_fix">Easy — Basic Audit & Fix</option>
      <option value="medium_format_dedup">Medium — Format & Deduplication</option>
      <option value="hard_full_audit">Hard — Full Audit, No Hints</option>
    </select>
    <input type="number" id="seed-input" value="42" title="Random seed">
    <button class="btn-primary" onclick="doReset()">▶ Start Episode</button>
  </div>
  <div id="task-desc" style="font-size:.8rem;color:#6b7280;margin-top:8px"></div>
</div>

<!-- AI AGENT CONFIG -->
<div class="card">
  <h2>AI Agent Config
    <span id="provider-badge" class="badge badge-local" style="display:none"></span>
  </h2>
  <div class="token-row">
    <input type="password" id="token-input"
           placeholder="HF Token  (hf_...)">
    <input type="text" id="model-input"
           placeholder="Model name"
           value="Qwen/Qwen2.5-72B-Instruct">
    <input type="text" id="endpoint-input"
           placeholder="API endpoint"
           value="https://router.huggingface.co/v1"
           style="width:240px">
  </div>
  <div class="token-hint">
    HF Router: token=<b>hf_xxx</b> · endpoint=<b>https://router.huggingface.co/v1</b> · model=<b>Qwen/Qwen2.5-72B-Instruct</b>
  </div>
  <div class="row" style="margin-top:12px;gap:8px">
    <button class="btn-ai"  id="btn-run-ai" onclick="runAIAgent()">🤖 Run AI Agent</button>
    <button class="btn-stop" id="btn-stop"  onclick="stopAgent()" style="display:none">⏹ Stop</button>
    <span id="agent-status" style="font-size:.82rem;color:#6b7280"></span>
  </div>
</div>

<!-- PROGRESS -->
<div class="card" id="progress-card" style="display:none">
  <h2>Progress</h2>
  <div class="stats">
    <div class="stat">Step <span id="s-step">0</span></div>
    <div class="stat">Fixed <span id="s-fixed">0</span></div>
    <div class="stat">Remaining <span id="s-rem">0</span></div>
    <div class="stat">Total Reward <span id="s-reward">0.00</span></div>
    <div class="stat">Score <span id="s-score">0.00</span></div>
  </div>
  <div class="pbar-bg"><div class="pbar-fill" id="pbar" style="width:0%"></div></div>
  <div class="plabel" id="plabel">0 / 0</div>
</div>

<!-- AGENT LOG -->
<div class="card" id="log-card" style="display:none">
  <h2>Agent Action Log</h2>
  <ul id="agent-log"></ul>
</div>

<!-- HINTS -->
<div class="card" id="hints-card" style="display:none">
  <h2>⚠ Issues Remaining</h2>
  <ul class="hint-list" id="hints-list"></ul>
</div>

<!-- TABLE -->
<div class="card" id="table-card" style="display:none">
  <h2>Current CRM Table</h2>
  <div class="tbl-wrap">
    <table><thead id="thead"></thead><tbody id="tbody"></tbody></table>
  </div>
</div>

<!-- MANUAL ACTIONS -->
<div class="card" id="manual-card" style="display:none">
  <h2>Manual Action</h2>
  <div class="row">
    <select id="op-select">
      <option value="fill_missing">fill_missing</option>
      <option value="remove_duplicate">remove_duplicate</option>
      <option value="standardize_format">standardize_format</option>
      <option value="fix_value">fix_value</option>
      <option value="get_column_stats">get_column_stats</option>
      <option value="bulk_fix_column">bulk_fix_column</option>
      <option value="flag_ambiguous">flag_ambiguous</option>
    </select>
    <input type="number" id="uid-in"  placeholder="row uid">
    <input type="text"   id="col-in"  placeholder="column">
    <input type="text"   id="val-in"  placeholder="value">
    <input type="text"   id="rsn-in"  placeholder="reason">
    <button class="btn-manual" onclick="doManualStep()">→ Step</button>
    <button class="btn-success" onclick="doSubmit()">✓ Submit</button>
  </div>
  <div id="result-msg"></div>
</div>

<script>
// ── State ──────────────────────────────────────────────────
let totalReward  = 0;
let totalIssues  = 0;
let issuesFixed  = 0;
let stepCount    = 0;
let agentRunning = false;
let agentStop    = false;
let prevTable    = {};   // uid → row snapshot for change detection

// ── Reset ──────────────────────────────────────────────────
async function doReset() {
  stopAgent();
  totalReward = 0; issuesFixed = 0; stepCount = 0; prevTable = {};

  hideDone();

  const taskId = document.getElementById('task-select').value;
  const seed   = parseInt(document.getElementById('seed-input').value) || 42;

  const obs = await apiPost('/reset', {task_id: taskId, seed});
  if (obs.error) { alert('Reset failed: ' + obs.error); return; }

  totalIssues = obs.total_issues || 0;
  renderObs(obs);
  showCards();
  setResult('Episode started — use AI Agent or Manual Action.', 'info');
  clearLog();
  document.getElementById('task-desc').textContent = obs.task_description || '';
}

// ── AI AGENT ───────────────────────────────────────────────
async function runAIAgent() {
  const token    = document.getElementById('token-input').value.trim();
  const model    = document.getElementById('model-input').value.trim();
  const endpoint = document.getElementById('endpoint-input').value.trim();

  if (!token)    { alert('Enter your HF token or "ollama" first.'); return; }
  if (!model)    { alert('Enter a model name.'); return; }
  if (!endpoint) { alert('Enter an API endpoint.'); return; }

  // Show badge
  const badge = document.getElementById('provider-badge');
  if (token === 'ollama' || endpoint.includes('localhost')) {
    badge.textContent = 'Local Ollama';
    badge.className   = 'badge badge-local';
  } else {
    badge.textContent = 'HF Router';
    badge.className   = 'badge badge-hf';
  }
  badge.style.display = 'inline';

  agentStop    = false;
  agentRunning = true;

  document.getElementById('btn-run-ai').style.display = 'none';
  document.getElementById('btn-stop').style.display   = 'inline-block';
  setAgentStatus('🤖 Agent running…');

  // Make sure episode is started
  const taskId = document.getElementById('task-select').value;
  const seed   = parseInt(document.getElementById('seed-input').value) || 42;
  const obs0   = await apiPost('/reset', {task_id: taskId, seed});
  if (obs0.error) { agentDone(); return; }

  totalReward = 0; issuesFixed = 0; stepCount = 0; prevTable = {};
  totalIssues = obs0.total_issues || 0;
  renderObs(obs0);
  showCards();
  clearLog();
  hideDone();

  let currentObs = obs0;
  const history  = [];
  const maxSteps = {easy_basic_fix:15, medium_format_dedup:20, hard_full_audit:30};
  const limit    = maxSteps[taskId] || 20;

  for (let step = 1; step <= limit; step++) {
    if (agentStop || currentObs.done) break;

    setAgentStatus(`🤖 Step ${step} — thinking…`);

    // ── Build prompt ──
    const prompt = buildPrompt(currentObs, history, step);

    // ── Call LLM ──
    let action;
    try {
      action = await callLLM(endpoint, token, model, prompt);
    } catch(e) {
      addLog(step, 'error', 0, `LLM call failed: ${e.message}`, '');
      break;
    }

    if (agentStop) break;

    setAgentStatus(`🤖 Step ${step} — acting: ${action.operation}…`);

    // ── Step environment ──
    const result = await apiPost('/step', {
      operation: action.operation,
      column:    action.column    || '',
      row_uid:   action.row_uid   || -1,
      value:     action.value     || '',
      reason:    action.reason    || '',
    });

    if (result.error) {
      addLog(step, 'error', 0, result.error, '');
      break;
    }

    const reward = result.reward || 0;
    totalReward += reward;
    stepCount    = step;

    addLog(step, action.operation, reward, result.last_action_result, action.reason);

    history.push(
      `Step ${step}: ${action.operation} uid=${action.row_uid} → ${result.last_action_result} (${reward >= 0 ? '+' : ''}${reward.toFixed(2)})`
    );

    renderObs(result);
    currentObs = result;

    if (result.done) break;

    // Delay so human can see each step
    await sleep(1400);
  }

  // ── Show final score ──
  const maxR  = (totalIssues * 0.15) + 0.60;
  const score = Math.max(0, Math.min(1, totalReward / maxR));
  showDone(score, currentObs.issues_fixed || 0, totalIssues);

  agentDone();
}

function stopAgent() {
  agentStop    = true;
  agentRunning = false;
  agentDone();
}

function agentDone() {
  agentRunning = false;
  document.getElementById('btn-run-ai').style.display = 'inline-block';
  document.getElementById('btn-stop').style.display   = 'none';
  setAgentStatus('');
}

// ── LLM CALL ───────────────────────────────────────────────
async function callLLM(endpoint, token, model, prompt) {
  const systemPrompt = `You are a CRM data cleaning agent. Fix ONE issue per response.
RESPOND WITH ONLY RAW JSON. NO MARKDOWN. NO EXPLANATION.

JSON FORMAT:
{"operation":"OPERATION","column":"COLUMN","row_uid":UID,"value":"VALUE","reason":"REASON"}

Operations: fill_missing, remove_duplicate, standardize_format, fix_value, bulk_fix_column, submit

RULES:
- Use uid number from the uid column
- Phones: (XXX) XXX-XXXX format
- Dates: YYYY-MM-DD format
- Cities: Title Case
- loyalty_points must be >= 0
- MISSING in cell = null value, fill it
- Call submit when ALL issues are fixed`;

  const baseUrl = endpoint.endsWith('/v1') ? endpoint : endpoint.replace(/\/$/, '') + '/v1';

  const resp = await fetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type':  'application/json',
      'Authorization': `Bearer ${token}`,
    },
    body: JSON.stringify({
      model:       model,
      messages: [
        {role: 'system', content: systemPrompt},
        {role: 'user',   content: prompt},
      ],
      temperature: 0.1,
      max_tokens:  256,
    }),
  });

  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`${resp.status}: ${err.slice(0,120)}`);
  }

  const data = await resp.json();
  const text = (data.choices?.[0]?.message?.content || '').trim();

  console.log('[LLM raw]', text);

  return parseAction(text);
}

// ── PROMPT BUILDER ─────────────────────────────────────────
function buildPrompt(obs, history, step) {
  const rem     = (obs.total_issues || 0) - (obs.issues_fixed || 0);
  const hints   = (obs.issues_remaining || []);
  const table   = obs.table_markdown || '';

  let p = `Step ${step}. Progress: ${obs.issues_fixed||0} fixed, ${rem} remaining.\n`;

  if (obs.last_action_result && !obs.last_action_result.includes('Episode started')) {
    p += `Last result: ${obs.last_action_result}\n`;
  }

  if (hints.length > 0) {
    p += `\nISSUES TO FIX:\n${hints.map(h => '  - ' + h).join('\n')}\n`;
  } else {
    p += '\nNo hints. Find issues yourself by reading the table.\n';
  }

  if (history.length > 0) {
    p += `\nRecent actions:\n${history.slice(-3).map(h => '  ' + h).join('\n')}\n`;
  }

  p += `\nCURRENT TABLE:\n${table}\n`;

  if (rem === 0) {
    p += '\nALL ISSUES FIXED. Call submit now.';
  } else {
    p += `\nFix the next issue. Respond with ONE JSON action:`;
  }

  return p;
}

// ── ACTION PARSER ──────────────────────────────────────────
function parseAction(text) {
  const safe = {operation:'submit', column:'', row_uid:-1, value:'', reason:''};
  if (!text) return safe;

  // Strip markdown fences
  text = text.replace(/```json\s*/g,'').replace(/```/g,'').trim();

  // Extract JSON
  const m = text.match(/\{[^{}]*\}/) || text.match(/\{[\s\S]*\}/);
  if (!m) return safe;

  try {
    const d = JSON.parse(m[0]);
    const validOps = new Set([
      'fill_missing','remove_duplicate','standardize_format',
      'fix_value','get_column_stats','bulk_fix_column',
      'flag_ambiguous','submit'
    ]);
    let op = (d.operation || 'submit').toLowerCase().trim();
    if (!validOps.has(op)) op = 'submit';

    return {
      operation: op,
      column:    String(d.column  || '').trim(),
      row_uid:   parseInt(d.row_uid ?? d.uid ?? -1) || -1,
      value:     String(d.value   || '').trim(),
      reason:    String(d.reason  || '').trim(),
    };
  } catch(e) {
    return safe;
  }
}

// ── MANUAL STEP ────────────────────────────────────────────
async function doManualStep() {
  const result = await apiPost('/step', {
    operation: document.getElementById('op-select').value,
    row_uid:   parseInt(document.getElementById('uid-in').value)  || -1,
    column:    document.getElementById('col-in').value.trim(),
    value:     document.getElementById('val-in').value.trim(),
    reason:    document.getElementById('rsn-in').value.trim(),
  });
  if (result.reward !== undefined) totalReward += result.reward;
  renderObs(result);
  const cls = result.reward > 0 ? 'ok' : result.reward < 0 ? 'bad' : 'info';
  setResult(result.last_action_result || '', cls);
}

async function doSubmit() {
  const result = await apiPost('/step',
    {operation:'submit',column:'',row_uid:-1,value:'',reason:''});
  if (result.reward !== undefined) totalReward += result.reward;
  renderObs(result);
  const cls = result.reward >= 0 ? 'ok' : 'bad';
  setResult(result.last_action_result || '', cls);
  if (result.done) {
    const maxR  = (totalIssues * 0.15) + 0.60;
    const score = Math.max(0, Math.min(1, totalReward / maxR));
    showDone(score, result.issues_fixed || 0, totalIssues);
  }
}

// ── RENDER OBSERVATION ─────────────────────────────────────
function renderObs(obs) {
  issuesFixed = obs.issues_fixed || 0;
  stepCount   = obs.step_number  || stepCount;
  totalIssues = obs.total_issues || totalIssues;

  const maxR  = (totalIssues * 0.15) + 0.60 || 1;
  const score = Math.max(0, Math.min(1, totalReward / maxR));

  // Stats
  document.getElementById('s-step').textContent   = stepCount;
  document.getElementById('s-fixed').textContent  = issuesFixed;
  document.getElementById('s-rem').textContent    = Math.max(0, totalIssues - issuesFixed);
  document.getElementById('s-reward').textContent = totalReward.toFixed(2);
  document.getElementById('s-score').textContent  = score.toFixed(2);

  // Progress bar
  const pct = totalIssues > 0 ? Math.round((issuesFixed / totalIssues) * 100) : 0;
  document.getElementById('pbar').style.width  = pct + '%';
  document.getElementById('plabel').textContent = `${issuesFixed} / ${totalIssues}`;

  // Hints
  const hints = obs.issues_remaining || [];
  const hc    = document.getElementById('hints-card');
  const hl    = document.getElementById('hints-list');
  if (hints.length > 0) {
    hc.style.display = 'block';
    hl.innerHTML = hints.map(h => `<li>${h}</li>`).join('');
  } else {
    hc.style.display = 'none';
  }

  // Table
  renderTable(obs.table_markdown || '');
}

// ── TABLE RENDERER ─────────────────────────────────────────
function renderTable(md) {
  const lines = md.trim().split('\n').filter(l => l.trim().startsWith('|'));
  if (lines.length < 2) return;

  const parse = line => line.split('|').slice(1,-1).map(c => c.trim());
  const heads = parse(lines[0]);
  const rows  = lines.slice(2).map(parse);

  // Build uid → row map for change detection
  const uidCol = heads.indexOf('uid');
  const newSnap = {};
  rows.forEach(r => { if (uidCol >= 0) newSnap[r[uidCol]] = r.slice(); });

  // Header
  document.getElementById('thead').innerHTML =
    '<tr>' + heads.map(h => `<th>${h}</th>`).join('') + '</tr>';

  // Rows
  const tbody = document.getElementById('tbody');
  tbody.innerHTML = '';

  rows.forEach(row => {
    const uid = uidCol >= 0 ? row[uidCol] : null;
    const tr  = document.createElement('tr');

    const wasDeleted = uid && prevTable[uid] && !newSnap[uid];
    const isNew      = uid && !prevTable[uid] && Object.keys(prevTable).length > 0;

    if (wasDeleted) tr.classList.add('row-deleted');

    row.forEach((cell, i) => {
      const td      = document.createElement('td');
      const isMiss  = cell.includes('MISSING');
      const changed = uid && prevTable[uid] && prevTable[uid][i] !== cell;

      if (isMiss) {
        td.innerHTML = `<span class="cell-missing">${cell}</span>`;
      } else {
        td.textContent = cell;
        if (changed) td.classList.add('cell-fixed');
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  prevTable = newSnap;
}

// ── AGENT LOG ──────────────────────────────────────────────
function addLog(step, operation, reward, result, reason) {
  const ul  = document.getElementById('agent-log');
  const li  = document.createElement('li');
  li.className = 'log-item';

  const rClass = reward > 0 ? 'pos' : reward < 0 ? 'neg' : 'zero';
  const rSign  = reward > 0 ? '+' : '';

  li.innerHTML = `
    <span class="log-step">Step ${step}</span>
    <span class="log-op">${operation}</span>
    <span class="log-reward ${rClass}">${rSign}${reward.toFixed(2)}</span>
    <span class="log-reason">${result || ''}${reason ? ' — <em>' + reason + '</em>' : ''}</span>
  `;
  ul.appendChild(li);
  ul.scrollTop = ul.scrollHeight;
}

function clearLog() {
  document.getElementById('agent-log').innerHTML = '';
}

// ── DONE BANNER ────────────────────────────────────────────
function showDone(score, fixed, total) {
  const banner  = document.getElementById('done-banner');
  const scoreEl = document.getElementById('done-score');
  const msgEl   = document.getElementById('done-msg');

  scoreEl.textContent = score.toFixed(2);
  scoreEl.style.color = score >= 0.8 ? '#4ade80' : score >= 0.4 ? '#fbbf24' : '#f87171';
  msgEl.textContent   = `${fixed} of ${total} issues resolved`;
  banner.style.display = 'block';
}

function hideDone() {
  document.getElementById('done-banner').style.display = 'none';
}

// ── HELPERS ────────────────────────────────────────────────
async function apiPost(url, body) {
  try {
    const r = await fetch(url, {
      method:  'POST',
      headers: {'Content-Type':'application/json'},
      body:    JSON.stringify(body),
    });
    return await r.json();
  } catch(e) {
    return {error: e.message};
  }
}

function showCards() {
  ['progress-card','log-card','table-card','manual-card']
    .forEach(id => document.getElementById(id).style.display = 'block');
}

function setResult(msg, cls) {
  const el = document.getElementById('result-msg');
  el.textContent = msg;
  el.className   = cls || '';
}

function setAgentStatus(msg) {
  document.getElementById('agent-status').textContent = msg;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    """
    Main entry point. Required by openenv validate.
    Called by: server = "server.app:main" in pyproject.toml
    """
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"[CRM Sanitizer] Starting on {host}:{port}", flush=True)
    print(f"[CRM Sanitizer] Web UI:   http://{host}:{port}/web",    flush=True)
    print(f"[CRM Sanitizer] Health:   http://{host}:{port}/health", flush=True)

    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()