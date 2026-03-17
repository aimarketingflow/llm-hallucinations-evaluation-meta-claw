"""
Checkpoint / Resume system for DragonClaw test suites.

Saves progress after each test so a crashed run can resume from
the last completed test instead of restarting from scratch.

Usage in a test suite:
    from checkpoint import CheckpointManager
    ckpt = CheckpointManager("multimodel_cascade")
    if ckpt.should_skip("O46"):
        results.append(ckpt.get_result("O46"))
    else:
        result = run_test_o46()
        ckpt.save("O46", result)
    ...
    ckpt.finish()  # clears checkpoint file on clean completion

Checkpoint files: records/checkpoints/<suite_name>.ckpt.json
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("checkpoint")

RECORDS_DIR = Path(__file__).resolve().parent.parent / "records"
CKPT_DIR = RECORDS_DIR / "checkpoints"


class CheckpointManager:
    """Manages checkpoint state for a test suite."""

    def __init__(self, suite_name: str, enabled: bool = True):
        self.suite_name = suite_name
        self.enabled = enabled
        self.ckpt_path = CKPT_DIR / f"{suite_name}.ckpt.json"
        self.state: dict[str, Any] = {}
        self.started_at = time.time()
        self.resumed = False

        if not enabled:
            return

        CKPT_DIR.mkdir(parents=True, exist_ok=True)

        # Load existing checkpoint if present
        if self.ckpt_path.exists():
            try:
                with open(self.ckpt_path) as f:
                    self.state = json.load(f)
                completed = list(self.state.get("completed", {}).keys())
                if completed:
                    self.resumed = True
                    logger.info(
                        "[CHECKPOINT] Resuming %s — %d tests already done: %s",
                        suite_name, len(completed), ", ".join(completed),
                    )
                else:
                    logger.info("[CHECKPOINT] Found empty checkpoint for %s — starting fresh", suite_name)
            except (json.JSONDecodeError, KeyError):
                logger.warning("[CHECKPOINT] Corrupt checkpoint for %s — starting fresh", suite_name)
                self.state = {}

        if "completed" not in self.state:
            self.state["completed"] = {}
        if "metadata" not in self.state:
            self.state["metadata"] = {
                "suite": suite_name,
                "created_at": time.time(),
            }

    def should_skip(self, test_id: str) -> bool:
        """Return True if this test was already completed in a previous run."""
        if not self.enabled:
            return False
        return test_id in self.state.get("completed", {})

    def get_result(self, test_id: str) -> dict | None:
        """Get the saved result for a previously completed test."""
        return self.state.get("completed", {}).get(test_id)

    def save(self, test_id: str, result: dict) -> None:
        """Save a completed test result and flush to disk."""
        if not self.enabled:
            return
        self.state["completed"][test_id] = result
        self.state["metadata"]["last_saved"] = time.time()
        self.state["metadata"]["last_test"] = test_id
        self._flush()
        logger.info("[CHECKPOINT] Saved %s (%d/%s tests done)",
                    test_id, len(self.state["completed"]),
                    self.state["metadata"].get("total_tests", "?"))

    def set_total(self, n: int) -> None:
        """Record total number of tests for progress tracking."""
        self.state["metadata"]["total_tests"] = n

    def get_completed_results(self) -> list[dict]:
        """Return all completed test results in order."""
        return list(self.state.get("completed", {}).values())

    def finish(self, keep: bool = False) -> None:
        """Mark suite as fully done. Removes checkpoint unless keep=True."""
        if not self.enabled:
            return
        self.state["metadata"]["finished_at"] = time.time()
        self.state["metadata"]["elapsed"] = time.time() - self.started_at
        self.state["metadata"]["status"] = "complete"
        self._flush()
        if not keep and self.ckpt_path.exists():
            # Move to .done instead of deleting
            done_path = self.ckpt_path.with_suffix(".done.json")
            self.ckpt_path.rename(done_path)
            logger.info("[CHECKPOINT] Suite complete — checkpoint moved to %s",
                        done_path.name)

    def _flush(self) -> None:
        """Write state to disk atomically."""
        tmp = self.ckpt_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=2, default=str)
        tmp.replace(self.ckpt_path)
