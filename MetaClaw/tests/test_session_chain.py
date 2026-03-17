#!/usr/bin/env python3
"""O71-O75: Session Chain — Auto-Spawn Conversation Chaining Tests.

Tests the auto-spawn chain architecture:
  O71 — Disk persistence: save/load ConversationMemory round-trip
  O72 — Token budget monitor: spawn signal at threshold
  O73 — Session summarizer: extract mode (no LLM needed)
  O74 — Handoff protocol: memory continuity across sessions
  O75 — End-to-end chain: multi-session with recall validation

Run:
    python tests/test_session_chain.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time

# ── path setup ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

# Direct imports to avoid dragonclaw/__init__.py (requires uvicorn etc.)
import importlib.util

def _direct_import(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

_cm_mod = _direct_import(
    "dragonclaw.conversation_memory",
    os.path.join(PROJECT_DIR, "dragonclaw", "conversation_memory.py"),
)
ConversationMemory = _cm_mod.ConversationMemory
MemoryChunk = _cm_mod.MemoryChunk

_sc_mod = _direct_import(
    "dragonclaw.session_chain",
    os.path.join(PROJECT_DIR, "dragonclaw", "session_chain.py"),
)
HandoffPayload = _sc_mod.HandoffPayload
SessionChain = _sc_mod.SessionChain
SessionSummarizer = _sc_mod.SessionSummarizer
TokenBudgetMonitor = _sc_mod.TokenBudgetMonitor
estimate_tokens = _sc_mod.estimate_tokens

# ── logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── results ─────────────────────────────────────────────────────────
RESULTS_FILE = os.path.join(PROJECT_DIR, "records", "session_chain_results.json")
RESULTS = []

TOTAL_TESTS = 5
completed = 0


def _save_results(elapsed: float = 0.0):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    verdicts = [r["verdict"] for r in RESULTS]
    payload = {
        "suite": "session_chain",
        "suite_name": "O71-O75: Session Chain Auto-Spawn",
        "mode": "local",
        "elapsed_sec": elapsed,
        "total_turns": sum(r.get("total_turns", 0) for r in RESULTS),
        "verdicts": {
            "PASS": verdicts.count("PASS"),
            "WARN": verdicts.count("WARN"),
            "FAIL": verdicts.count("FAIL"),
        },
        "tests": RESULTS,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _log_progress(test_id: str, status: str):
    global completed
    completed += 1
    pct = round(completed / TOTAL_TESTS * 100)
    logger.info("[%d%%] %s: %s", pct, test_id, status)


# ====================================================================
# O71 — Disk persistence: save/load round-trip
# ====================================================================

def test_o71_disk_persistence():
    """Save ConversationMemory to disk, load it back, verify retrieval works."""
    logger.info("=" * 60)
    logger.info("O71: Disk Persistence Round-Trip")
    logger.info("=" * 60)

    r = {"test_id": "O71", "name": "Disk Persistence Round-Trip"}
    t0 = time.time()

    # Build a memory with known facts
    cm = ConversationMemory(use_embeddings=False, confidence_threshold=0.3)
    facts = [
        ("user", "The capital of France is Paris.", 1),
        ("assistant", "Yes, Paris is the capital of France.", 2),
        ("user", "The speed of light is 299,792,458 meters per second.", 3),
        ("assistant", "That's correct, approximately 300,000 km/s.", 4),
        ("user", "Mount Everest is the tallest mountain on Earth.", 5),
        ("assistant", "Mount Everest stands at 8,849 meters above sea level.", 6),
    ]

    for role, content, turn in facts:
        cm.ingest_turn(role, content, turn)

    # Verify retrieval works before save
    pre_save = cm.retrieve("capital of France", top_k=1, defense_gate=False)
    assert len(pre_save) > 0, "Pre-save retrieval failed"
    pre_save_text = pre_save[0]["chunk_text"]

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = f.name

    try:
        save_result = cm.save_to_disk(tmp_path)
        logger.info("  Saved: %d chunks, %d bytes", save_result["chunks_saved"], save_result["size_bytes"])

        # Verify file exists and is valid JSON
        assert os.path.exists(tmp_path), "Save file not found"
        with open(tmp_path) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert len(data["chunks"]) == save_result["chunks_saved"]

        # Load from disk
        cm2 = ConversationMemory.load_from_disk(tmp_path)
        logger.info("  Loaded: %d chunks", len(cm2._chunks))

        # Verify same number of chunks
        assert len(cm2._chunks) == len(cm._chunks), (
            f"Chunk count mismatch: {len(cm2._chunks)} vs {len(cm._chunks)}"
        )

        # Verify retrieval works after load
        post_load = cm2.retrieve("capital of France", top_k=1, defense_gate=False)
        assert len(post_load) > 0, "Post-load retrieval failed"
        post_load_text = post_load[0]["chunk_text"]

        # Verify same result
        match = pre_save_text == post_load_text
        logger.info("  Pre-save result:  %s", pre_save_text[:80])
        logger.info("  Post-load result: %s", post_load_text[:80])
        logger.info("  Match: %s", match)

        # Verify stats survived
        stats = cm2.get_stats()
        assert stats["total_turns"] == 6, f"Stats mismatch: {stats}"
        assert stats["max_turn"] == 6

        r["chunks_saved"] = save_result["chunks_saved"]
        r["chunks_loaded"] = len(cm2._chunks)
        r["file_size_bytes"] = save_result["size_bytes"]
        r["retrieval_match"] = match
        r["stats_preserved"] = stats["total_turns"] == 6
        r["verdict"] = "PASS" if match and r["stats_preserved"] else "FAIL"

    finally:
        os.unlink(tmp_path)

    r["elapsed_s"] = round(time.time() - t0, 2)
    RESULTS.append(r)
    _log_progress("O71", r["verdict"])
    return r


# ====================================================================
# O72 — Token budget monitor: spawn signal at threshold
# ====================================================================

def test_o72_token_budget_monitor():
    """Verify TokenBudgetMonitor fires spawn signal at correct threshold."""
    logger.info("=" * 60)
    logger.info("O72: Token Budget Monitor")
    logger.info("=" * 60)

    r = {"test_id": "O72", "name": "Token Budget Monitor"}
    t0 = time.time()

    monitor = TokenBudgetMonitor(
        max_tokens=1000,
        reserve_tokens=100,
        spawn_threshold=0.80,
    )

    # Usable budget = 1000 - 100 = 900
    # Spawn at 80% = 720 tokens
    assert monitor.usable_budget == 900
    assert not monitor.should_spawn

    # Add turns until we cross threshold
    spawn_turn = None
    statuses = []
    for i in range(100):
        content = f"Turn {i}: " + "word " * 20  # ~25 tokens per turn
        status = monitor.add_turn(content)
        statuses.append(status)
        if status["should_spawn"] and spawn_turn is None:
            spawn_turn = status["turn"]
            logger.info("  Spawn signal at turn %d (%.1f%% used, %d tokens)",
                        spawn_turn, status["usage_pct"], status["current_tokens"])
            break

    assert spawn_turn is not None, "Monitor never fired spawn signal"

    # Verify it fired at approximately the right point
    # 720 tokens / ~25 tokens per turn ≈ 29 turns
    logger.info("  Spawn turn: %d (expected ~29)", spawn_turn)
    reasonable = 20 <= spawn_turn <= 40  # generous range for estimation variance

    # Verify remaining tokens
    remaining = monitor.tokens_remaining
    logger.info("  Tokens remaining: %d", remaining)
    assert remaining >= 0

    # Test reset
    monitor.reset()
    assert monitor.current_tokens == 0
    assert not monitor.should_spawn

    r["spawn_turn"] = spawn_turn
    r["usable_budget"] = 900
    r["spawn_at_pct"] = statuses[-1]["usage_pct"] if statuses else 0
    r["reasonable_timing"] = reasonable
    r["reset_works"] = monitor.current_tokens == 0
    r["verdict"] = "PASS" if reasonable and r["reset_works"] else "FAIL"

    r["elapsed_s"] = round(time.time() - t0, 2)
    RESULTS.append(r)
    _log_progress("O72", r["verdict"])
    return r


# ====================================================================
# O73 — Session summarizer: extract mode
# ====================================================================

def test_o73_session_summarizer():
    """Test extract-mode summarization (no LLM needed)."""
    logger.info("=" * 60)
    logger.info("O73: Session Summarizer (Extract Mode)")
    logger.info("=" * 60)

    r = {"test_id": "O73", "name": "Session Summarizer (Extract Mode)"}
    t0 = time.time()

    summarizer = SessionSummarizer(mode="extract")

    # Build a realistic conversation
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris. It is the largest city in France."},
        {"role": "user", "content": "What about Germany?"},
        {"role": "assistant", "content": "The capital of Germany is Berlin. It has been the capital since reunification in 1990."},
        {"role": "user", "content": "How fast does light travel?"},
        {"role": "assistant", "content": "Light travels at 299,792,458 meters per second in a vacuum."},
        {"role": "user", "content": "What is the tallest mountain?"},
        {"role": "assistant", "content": "Mount Everest is the tallest mountain at 8,849 meters above sea level."},
        {"role": "user", "content": "Can you remind me about France's capital later?"},
        {"role": "assistant", "content": "Of course! I'll remember that Paris is the capital of France."},
    ]

    result = summarizer.summarize(messages)
    summary = result["summary"]

    logger.info("  Summary mode: %s", result["mode"])
    logger.info("  Input turns: %d", result["input_turns"])
    logger.info("  Summary tokens: %d", result["tokens_est"])
    logger.info("  Summary preview: %s...", summary[:200])

    # Verify summary contains key information
    summary_lower = summary.lower()
    checks = {
        "mentions_paris": "paris" in summary_lower,
        "mentions_berlin": "berlin" in summary_lower,
        "mentions_light": "light" in summary_lower or "299" in summary_lower,
        "mentions_everest": "everest" in summary_lower,
        "has_content": len(summary) > 50,
        "not_too_long": result["tokens_est"] < 1000,
    }

    passed = sum(checks.values())
    total = len(checks)
    logger.info("  Checks: %d/%d passed", passed, total)
    for k, v in checks.items():
        logger.info("    %s: %s", k, "PASS" if v else "FAIL")

    r["summary_length"] = len(summary)
    r["summary_tokens"] = result["tokens_est"]
    r["checks"] = checks
    r["checks_passed"] = passed
    r["checks_total"] = total
    r["verdict"] = "PASS" if passed >= 4 else "WARN" if passed >= 2 else "FAIL"

    r["elapsed_s"] = round(time.time() - t0, 2)
    RESULTS.append(r)
    _log_progress("O73", r["verdict"])
    return r


# ====================================================================
# O74 — Handoff protocol: memory continuity across sessions
# ====================================================================

def test_o74_handoff_protocol():
    """Test that handoff payload correctly transfers memory between sessions."""
    logger.info("=" * 60)
    logger.info("O74: Handoff Protocol — Memory Continuity")
    logger.info("=" * 60)

    r = {"test_id": "O74", "name": "Handoff Protocol — Memory Continuity"}
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        chain = SessionChain(
            max_tokens=2000,
            reserve_tokens=200,
            spawn_threshold=0.80,
            memory_dir=tmpdir,
            summarizer_mode="extract",
            use_embeddings=False,
            confidence_threshold=0.3,
        )

        # Session 1: inject facts
        session1 = chain.start_session()
        logger.info("  Session 1 started: #%d", session1["session_number"])

        facts = [
            ("The capital of France is Paris.", "Yes, Paris is the capital."),
            ("The speed of light is 299,792,458 m/s.", "Correct, approximately 300,000 km/s."),
            ("Mount Everest is 8,849 meters tall.", "That's the tallest mountain on Earth."),
            ("Water boils at 100 degrees Celsius.", "At standard atmospheric pressure, yes."),
            ("The Earth orbits the Sun.", "It takes approximately 365.25 days."),
        ]

        for user_msg, asst_msg in facts:
            status = chain.add_exchange(user_msg, asst_msg)

        logger.info("  Session 1: %d tokens used (%.1f%%)",
                     status["current_tokens"], status["usage_pct"])

        # Force spawn (even if not at threshold)
        handoff = chain.spawn_new_session(recent_query="capital of France")
        logger.info("  Handoff: summary=%d tokens, memory=%s, facts=%d",
                     estimate_tokens(handoff.summary), handoff.memory_path, handoff.facts_count)

        # Verify handoff payload
        assert handoff.session_number == 1
        assert handoff.total_turns == 5
        assert handoff.facts_count > 0
        assert os.path.exists(handoff.memory_path)

        # Session 2: load from handoff
        session2 = chain.start_session(handoff=handoff)
        logger.info("  Session 2 started: #%d, loaded %d chunks",
                     session2["session_number"], session2["memory_chunks"])

        assert session2["session_number"] == 2
        assert session2["memory_chunks"] > 0
        assert session2["system_message"] is not None

        # Verify retrieval works in session 2 (facts from session 1)
        queries = {
            "capital of France": "paris",
            "speed of light": "299",
            "tallest mountain": "everest",
        }

        correct = 0
        total_q = len(queries)
        for query, expected_keyword in queries.items():
            results = chain._memory.retrieve(query, top_k=1, defense_gate=False)
            if results and expected_keyword.lower() in results[0]["chunk_text"].lower():
                correct += 1
                logger.info("    '%s' -> FOUND: %s", query, results[0]["chunk_text"][:80])
            else:
                found = results[0]["chunk_text"][:80] if results else "NONE"
                logger.info("    '%s' -> MISS: %s", query, found)

        recall = correct / total_q if total_q > 0 else 0
        logger.info("  Cross-session recall: %d/%d (%.0f%%)", correct, total_q, recall * 100)

        # Verify system message format
        sys_msg = handoff.to_system_message()
        assert "SESSION HANDOFF" in sys_msg
        assert "PREVIOUS SESSION SUMMARY" in sys_msg
        logger.info("  System message length: %d chars", len(sys_msg))

        r["session1_turns"] = 5
        r["session1_tokens"] = status["current_tokens"]
        r["handoff_facts"] = handoff.facts_count
        r["session2_chunks_loaded"] = session2["memory_chunks"]
        r["cross_session_recall"] = round(recall, 2)
        r["queries_correct"] = correct
        r["queries_total"] = total_q
        r["system_message_present"] = session2["system_message"] is not None
        r["verdict"] = "PASS" if recall >= 0.6 else "WARN" if recall > 0 else "FAIL"

    r["elapsed_s"] = round(time.time() - t0, 2)
    RESULTS.append(r)
    _log_progress("O74", r["verdict"])
    return r


# ====================================================================
# O75 — End-to-end chain: multi-session recall validation
# ====================================================================

def test_o75_end_to_end_chain():
    """Full chain test: 3 sessions with auto-spawn and cross-session recall."""
    logger.info("=" * 60)
    logger.info("O75: End-to-End Chain — Multi-Session Recall")
    logger.info("=" * 60)

    r = {"test_id": "O75", "name": "End-to-End Chain — Multi-Session Recall"}
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Use a tiny context window to force auto-spawns
        chain = SessionChain(
            max_tokens=800,
            reserve_tokens=100,
            spawn_threshold=0.80,
            memory_dir=tmpdir,
            summarizer_mode="extract",
            use_embeddings=False,
            confidence_threshold=0.3,
        )

        # Session 1
        chain.start_session()
        logger.info("  === Session 1 ===")

        session1_facts = [
            ("The capital of Japan is Tokyo.", "Tokyo is indeed the capital of Japan."),
            ("Python was created by Guido van Rossum.", "Yes, Guido created Python in 1991."),
            ("The Pacific Ocean is the largest ocean.", "It covers about 63 million square miles."),
        ]

        handoff = None
        for user_msg, asst_msg in session1_facts:
            status = chain.add_exchange(user_msg, asst_msg)
            logger.info("    Turn %d: %.1f%% tokens, spawn=%s",
                         status["turn"], status["usage_pct"], status["should_spawn"])
            if status["should_spawn"] and handoff is None:
                handoff = chain.spawn_new_session()
                break

        # If we didn't hit spawn naturally, force it
        if handoff is None:
            handoff = chain.spawn_new_session()

        # Session 2
        chain.start_session(handoff=handoff)
        logger.info("  === Session 2 (loaded %d chunks) ===", len(chain._memory._chunks))

        session2_facts = [
            ("The boiling point of water is 100 degrees Celsius.", "At standard pressure, correct."),
            ("Albert Einstein developed the theory of relativity.", "He published special relativity in 1905."),
        ]

        for user_msg, asst_msg in session2_facts:
            status = chain.add_exchange(user_msg, asst_msg)
            logger.info("    Turn %d: %.1f%% tokens, spawn=%s",
                         status["turn"], status["usage_pct"], status["should_spawn"])
            if status["should_spawn"]:
                handoff = chain.spawn_new_session()
                break
        else:
            handoff = chain.spawn_new_session()

        # Session 3
        chain.start_session(handoff=handoff)
        logger.info("  === Session 3 (loaded %d chunks) ===", len(chain._memory._chunks))

        # Now test recall of facts from ALL previous sessions
        queries = {
            "capital of Japan": "tokyo",
            "who created Python": "guido",
            "largest ocean": "pacific",
            "boiling point of water": "100",
            "theory of relativity": "einstein",
        }

        correct = 0
        total_q = len(queries)
        for query, expected in queries.items():
            results = chain._memory.retrieve(query, top_k=1, defense_gate=False)
            if results and expected.lower() in results[0]["chunk_text"].lower():
                correct += 1
                logger.info("    '%s' -> FOUND: %s", query, results[0]["chunk_text"][:60])
            else:
                found = results[0]["chunk_text"][:60] if results else "NONE"
                logger.info("    '%s' -> MISS: %s", query, found)

        recall = correct / total_q if total_q > 0 else 0
        logger.info("  Cross-session recall (3 sessions): %d/%d (%.0f%%)", correct, total_q, recall * 100)

        # Check chain history
        history = chain.get_chain_history()
        logger.info("  Chain history: %d sessions recorded", len(history))

        # Check overall status
        status = chain.get_status()
        logger.info("  Total turns across all sessions: %d", status["total_turns"])
        logger.info("  Memory chunks in session 3: %d", status["memory"].get("total_chunks", 0))

        r["sessions_completed"] = 3
        r["total_turns"] = status["total_turns"]
        r["session3_memory_chunks"] = status["memory"].get("total_chunks", 0)
        r["chain_history_length"] = len(history)
        r["cross_session_recall"] = round(recall, 2)
        r["queries_correct"] = correct
        r["queries_total"] = total_q

        # Verdict: PASS if >=60% recall across 3 sessions
        # WARN if some recall but under 60%
        # FAIL if no recall at all
        if recall >= 0.6:
            r["verdict"] = "PASS"
        elif recall > 0:
            r["verdict"] = "WARN"
        else:
            r["verdict"] = "FAIL"

    r["elapsed_s"] = round(time.time() - t0, 2)
    RESULTS.append(r)
    _log_progress("O75", r["verdict"])
    return r


# ====================================================================
# Main
# ====================================================================

def main():
    logger.info("=" * 60)
    logger.info("O71-O75: Session Chain — Auto-Spawn Conversation Chaining")
    logger.info("=" * 60)

    t_start = time.time()

    test_o71_disk_persistence()
    test_o72_token_budget_monitor()
    test_o73_session_summarizer()
    test_o74_handoff_protocol()
    test_o75_end_to_end_chain()

    elapsed = round(time.time() - t_start, 1)

    _save_results(elapsed)

    # Summary
    verdicts = [r["verdict"] for r in RESULTS]
    p = verdicts.count("PASS")
    w = verdicts.count("WARN")
    f = verdicts.count("FAIL")

    logger.info("=" * 60)
    logger.info("RESULTS: %dP %dW %dF (%.1fs)", p, w, f, elapsed)
    logger.info("=" * 60)
    for r in RESULTS:
        logger.info("  %s: %s — %s", r["test_id"], r["verdict"], r["name"])
    logger.info("Results saved to %s", RESULTS_FILE)


if __name__ == "__main__":
    main()
