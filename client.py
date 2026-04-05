# client.py
# CRM Sanitizer — Python Client
#
# This is what users import to connect to the environment.
# Works whether the server is running:
#   - Locally:          base_url="http://localhost:7860"
#   - In Docker:        base_url="http://localhost:7860"
#   - On HF Spaces:     base_url="https://your-username-crm-sanitizer.hf.space"
#
# Usage (simple):
#   from client import CRMSanitizerEnv, CRMAction
#   env = CRMSanitizerEnv(base_url="http://localhost:7860")
#   obs = env.reset(task_id="easy_basic_fix", seed=42)
#   obs = env.step(CRMAction(operation="submit", column="", row_uid=-1, value=""))
#
# Usage (context manager — recommended):
#   with CRMSanitizerEnv(base_url="http://localhost:7860") as env:
#       obs = env.reset(task_id="easy_basic_fix", seed=42)
#       obs = env.step(CRMAction(operation="fill_missing",
#                                column="email",
#                                row_uid=1001,
#                                value="john@example.com"))

import json
import time
from typing import Any, Dict, List, Optional

import httpx

# Import models from parent directory
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import CRMAction, CRMObservation, CRMState


# ─────────────────────────────────────────────
# STEP RESULT
# Wraps observation + reward + done
# Returned by every reset() and step() call
# ─────────────────────────────────────────────

class StepResult:
    """
    Wraps everything returned after one environment step.

    Attributes:
        observation:  CRMObservation — the full environment state
        reward:       float or None — reward from last action
        done:         bool — True if episode is finished
        info:         dict — extra metadata
    """

    def __init__(
        self,
        observation: CRMObservation,
        reward: Optional[float],
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ):
        self.observation = observation
        self.reward      = reward
        self.done        = done
        self.info        = info or {}

    def __repr__(self) -> str:
        return (
            f"StepResult("
            f"reward={self.reward}, "
            f"done={self.done}, "
            f"issues_fixed={self.observation.issues_fixed}/"
            f"{self.observation.total_issues})"
        )


# ─────────────────────────────────────────────
# MAIN CLIENT CLASS
# ─────────────────────────────────────────────

class CRMSanitizerEnv:
    """
    Python client for the CRM Sanitizer OpenEnv environment.

    Connects to a running CRM Sanitizer server via HTTP.
    Provides clean Python methods — hides all HTTP details.

    Args:
        base_url:   URL of the running server.
                    Default: "http://localhost:7860"
        timeout:    Seconds to wait for server response.
                    Default: 30
        retries:    Number of connection retries on failure.
                    Default: 3
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        timeout: int = 30,
        retries: int = 3,
    ):
        # Remove trailing slash if present
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self.retries  = retries

        # HTTP client — reused across all requests
        self._client = httpx.Client(
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )

        # Track current episode state
        self._last_obs: Optional[CRMObservation] = None
        self._total_reward: float = 0.0
        self._step_count: int = 0

    # ─────────────────────────────────────────
    # CONTEXT MANAGER SUPPORT
    # Allows: with CRMSanitizerEnv(...) as env:
    # ─────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """Close the HTTP client cleanly."""
        try:
            self._client.close()
        except Exception:
            pass

    # ─────────────────────────────────────────
    # HEALTH CHECK
    # ─────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """
        Check if the server is running.
        Returns server status dict.
        Raises ConnectionError if server is unreachable.
        """
        return self._get("/health")

    def wait_until_ready(self, max_wait: int = 60) -> bool:
        """
        Poll health endpoint until server responds.
        Useful after starting Docker — server needs a few seconds.

        Args:
            max_wait: Maximum seconds to wait. Default 60.

        Returns:
            True if server is ready, False if timed out.
        """
        start = time.time()
        while time.time() - start < max_wait:
            try:
                result = self.health()
                if result.get("status") == "healthy":
                    return True
            except Exception:
                pass
            time.sleep(2)
        return False

    # ─────────────────────────────────────────
    # CORE ENVIRONMENT METHODS
    # ─────────────────────────────────────────

    def reset(
        self,
        task_id: str = "easy_basic_fix",
        seed: int = 42,
        episode_id: Optional[str] = None,
    ) -> StepResult:
        """
        Start a new episode.

        Args:
            task_id:    Which task to run.
                        "easy_basic_fix"      — Easy (10 rows, full hints)
                        "medium_format_dedup" — Medium (25 rows, partial hints)
                        "hard_full_audit"     — Hard (40 rows, no hints)
            seed:       Random seed. Same seed = same dirty table.
                        Use seed=42 for reproducible baseline scores.
            episode_id: Optional custom episode identifier.

        Returns:
            StepResult with initial observation of dirty CRM table.
        """
        # Reset tracking
        self._total_reward = 0.0
        self._step_count   = 0

        payload = {
            "task_id":    task_id,
            "seed":       seed,
            "episode_id": episode_id,
        }

        data = self._post("/reset", payload)
        obs  = self._parse_observation(data)

        self._last_obs = obs

        return StepResult(
            observation=obs,
            reward=None,
            done=False,
        )

    def step(self, action: CRMAction) -> StepResult:
        """
        Take one action in the environment.

        Args:
            action: CRMAction with operation details.

        Returns:
            StepResult with updated observation, reward, and done flag.

        Example:
            result = env.step(CRMAction(
                operation = "fill_missing",
                column    = "email",
                row_uid   = 1003,
                value     = "jane@company.com",
                reason    = "email was null, filled with plausible value",
            ))
        """
        self._step_count += 1

        payload = {
            "operation": action.operation,
            "column":    action.column,
            "row_uid":   action.row_uid,
            "value":     action.value,
            "reason":    action.reason,
        }

        data   = self._post("/step", payload)
        obs    = self._parse_observation(data)
        reward = obs.reward or 0.0

        self._total_reward += reward
        self._last_obs = obs

        return StepResult(
            observation=obs,
            reward=reward,
            done=obs.done,
            info={
                "total_reward": self._total_reward,
                "step_count":   self._step_count,
            },
        )

    def state(self) -> CRMState:
        """
        Get current episode metadata.

        Returns:
            CRMState with step count, issues fixed, seed, etc.
        """
        data = self._get("/state")
        return CRMState(
            episode_id            = data.get("episode_id"),
            step_count            = data.get("step_count", 0),
            task_id               = data.get("task_id", ""),
            seed                  = data.get("seed", 42),
            total_issues          = data.get("total_issues", 0),
            issues_fixed          = data.get("issues_fixed", 0),
            issues_wrongly_touched= data.get("issues_wrongly_touched", 0),
            max_steps             = data.get("max_steps", 15),
            is_complete           = data.get("is_complete", False),
        )

    # ─────────────────────────────────────────
    # CONVENIENCE PROPERTIES
    # ─────────────────────────────────────────

    @property
    def total_reward(self) -> float:
        """Total reward accumulated in current episode."""
        return self._total_reward

    @property
    def last_observation(self) -> Optional[CRMObservation]:
        """Most recent observation."""
        return self._last_obs

    # ─────────────────────────────────────────
    # HTTP HELPERS
    # All server communication goes through here
    # ─────────────────────────────────────────

    def _post(
        self,
        endpoint: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Send POST request to server with retry logic.
        Raises RuntimeError with clear message on failure.
        """
        url = f"{self.base_url}{endpoint}"

        for attempt in range(1, self.retries + 1):
            try:
                response = self._client.post(
                    url,
                    content=json.dumps(payload),
                )
                response.raise_for_status()
                return response.json()

            except httpx.ConnectError:
                if attempt == self.retries:
                    raise ConnectionError(
                        f"Cannot connect to CRM Sanitizer server at {self.base_url}. "
                        f"Is the server running? "
                        f"Start it with: python server/app.py"
                    )
                time.sleep(1)

            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"Server returned error {e.response.status_code} "
                    f"for POST {endpoint}: {e.response.text}"
                )

            except Exception as e:
                if attempt == self.retries:
                    raise RuntimeError(
                        f"Request to {endpoint} failed after "
                        f"{self.retries} attempts: {str(e)}"
                    )
                time.sleep(1)

        return {}

    def _get(
        self,
        endpoint: str,
    ) -> Dict[str, Any]:
        """
        Send GET request to server with retry logic.
        """
        url = f"{self.base_url}{endpoint}"

        for attempt in range(1, self.retries + 1):
            try:
                response = self._client.get(url)
                response.raise_for_status()
                return response.json()

            except httpx.ConnectError:
                if attempt == self.retries:
                    raise ConnectionError(
                        f"Cannot connect to CRM Sanitizer server at {self.base_url}. "
                        f"Is the server running?"
                    )
                time.sleep(1)

            except Exception as e:
                if attempt == self.retries:
                    raise RuntimeError(
                        f"GET {endpoint} failed: {str(e)}"
                    )
                time.sleep(1)

        return {}

    def _parse_observation(
        self,
        data: Dict[str, Any],
    ) -> CRMObservation:
        """
        Parse raw server response into a typed CRMObservation.
        Falls back to safe defaults if any field is missing.
        Never raises — always returns a valid observation.
        """
        try:
            return CRMObservation(
                done                = data.get("done", False),
                reward              = data.get("reward"),
                table_markdown      = data.get("table_markdown", ""),
                issues_remaining    = data.get("issues_remaining", []),
                issues_fixed        = data.get("issues_fixed", 0),
                total_issues        = data.get("total_issues", 0),
                last_action_result  = data.get("last_action_result", ""),
                task_description    = data.get("task_description", ""),
                task_id             = data.get("task_id", ""),
                step_number         = data.get("step_number", 0),
                column_stats        = data.get("column_stats"),
            )
        except Exception as e:
            # Return a safe fallback observation instead of crashing
            print(f"[client] Warning: could not parse observation: {e}")
            return CRMObservation(
                done               = True,
                reward             = 0.0,
                table_markdown     = "",
                last_action_result = f"parse error: {str(e)}",
                task_description   = "",
                task_id            = "",
                step_number        = 0,
            )

    def __repr__(self) -> str:
        return (
            f"CRMSanitizerEnv("
            f"base_url={self.base_url!r}, "
            f"steps={self._step_count}, "
            f"total_reward={self._total_reward:.3f})"
        )