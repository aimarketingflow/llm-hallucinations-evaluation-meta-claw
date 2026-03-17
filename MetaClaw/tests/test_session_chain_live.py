#!/usr/bin/env python3
"""O76-O80: Session Chain Live — Ollama-Powered Chain Tests.

Tests the auto-spawn chain architecture with live Ollama inference:
  O76 — Live summarizer: Ollama generates session summary
  O77 — Live multi-session recall: facts survive 3 session spawns
  O78 — Spawn under load: 50-turn session hits context limit, auto-spawns
  O79 — Cross-session poison defense: poisoned facts blocked after handoff
  O80 — Full pipeline: chain + memory + defense + live model recall

Dual-mode: live Ollama (preferred) or simulated fallback.

Run:
    python tests/test_session_chain_live.py
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

_sc_mod = _direct_import(
    "dragonclaw.session_chain",
    os.path.join(PROJECT_DIR, "dragonclaw", "session_chain.py"),
)
SessionChain = _sc_mod.SessionChain
SessionSummarizer = _sc_mod.SessionSummarizer
TokenBudgetMonitor = _sc_mod.TokenBudgetMonitor
estimate_tokens = _sc_mod.estimate_tokens

_def_mod = _direct_import(
    "dragonclaw.defense",
    os.path.join(PROJECT_DIR, "dragonclaw", "defense.py"),
)
FactVerifier = _def_mod.FactVerifier
DefenseStack = _def_mod.DefenseStack

# ── logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Ollama ──────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5:1.5b"
OLLAMA_AVAILABLE = False

try:
    import requests
    resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
    models = [m["name"] for m in resp.json().get("models", [])]
    OLLAMA_AVAILABLE = MODEL in models
    if OLLAMA_AVAILABLE:
        logger.info("[OLLAMA] Connected — %s available (%d models total)", MODEL, len(models))
    else:
        logger.warning("[OLLAMA] Connected but %s not found — using simulated mode", MODEL)
except Exception as e:
    logger.warning("[OLLAMA] Not available (%s) — using simulated mode", e)

MODE = "live" if OLLAMA_AVAILABLE else "simulated"

# ── results ─────────────────────────────────────────────────────────
RESULTS_FILE = os.path.join(PROJECT_DIR, "records", "session_chain_live_results.json")
RESULTS = []
TOTAL_TESTS = 5
completed = 0


def _save_results(elapsed: float = 0.0):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    verdicts = [r["verdict"] for r in RESULTS]
    payload = {
        "suite": "session_chain_live",
        "suite_name": "O76-O80: Session Chain Live",
        "mode": MODE,
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
    logger.info("[%d%%] %s: %s (%s)", pct, test_id, status, MODE)


def _ollama_chat(messages: list[dict], temperature: float = 0.3) -> str:
    """Send a chat request to Ollama and return the response text."""
    if not OLLAMA_AVAILABLE:
        return "[simulated response]"
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 256},
    }
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "").strip()


def _ollama_ask(question: str, context: str = "") -> str:
    """Quick helper: ask Ollama a question with optional context."""
    messages = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": question})
    return _ollama_chat(messages)


# ====================================================================
# O76 — Live summarizer: Ollama generates session summary
# ====================================================================

def test_o76_live_summarizer():
    """Test that Ollama produces a usable session summary."""
    logger.info("=" * 60)
    logger.info("O76: Live Summarizer (%s)", MODE)
    logger.info("=" * 60)

    r = {"test_id": "O76", "name": "Live Summarizer", "mode": MODE}
    t0 = time.time()

    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "And what about Japan?"},
        {"role": "assistant", "content": "The capital of Japan is Tokyo."},
        {"role": "user", "content": "What is the speed of light?"},
        {"role": "assistant", "content": "The speed of light is approximately 299,792,458 meters per second."},
        {"role": "user", "content": "Who painted the Mona Lisa?"},
        {"role": "assistant", "content": "Leonardo da Vinci painted the Mona Lisa."},
        {"role": "user", "content": "What year did World War 2 end?"},
        {"role": "assistant", "content": "World War 2 ended in 1945."},
    ]

    if OLLAMA_AVAILABLE:
        summarizer = SessionSummarizer(mode="ollama", model=MODEL, ollama_url=OLLAMA_URL)
    else:
        summarizer = SessionSummarizer(mode="extract")

    result = summarizer.summarize(messages)
    summary = result["summary"]

    logger.info("  Mode: %s", result["mode"])
    logger.info("  Input turns: %d", result["input_turns"])
    logger.info("  Summary tokens: %d", result["tokens_est"])
    logger.info("  Elapsed: %.1fs", result["elapsed_s"])
    logger.info("  Summary preview:\n%s", summary[:400])

    summary_lower = summary.lower()
    checks = {
        "mentions_paris": "paris" in summary_lower,
        "mentions_tokyo": "tokyo" in summary_lower,
        "mentions_light": "light" in summary_lower or "299" in summary_lower,
        "mentions_mona_lisa": "mona lisa" in summary_lower or "da vinci" in summary_lower or "leonardo" in summary_lower,
        "mentions_1945": "1945" in summary_lower or "world war" in summary_lower,
        "has_content": len(summary) > 30,
        "not_too_long": result["tokens_est"] < 2000,
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
    r["elapsed_s_summarize"] = result["elapsed_s"]
    r["verdict"] = "PASS" if passed >= 5 else "WARN" if passed >= 3 else "FAIL"

    r["elapsed_s"] = round(time.time() - t0, 2)
    RESULTS.append(r)
    _log_progress("O76", r["verdict"])
    return r


# ====================================================================
# O77 — Live multi-session recall: facts survive 3 spawns
# ====================================================================

def test_o77_live_multi_session_recall():
    """Inject facts via Ollama across 3 sessions, verify recall in session 3."""
    logger.info("=" * 60)
    logger.info("O77: Live Multi-Session Recall (%s)", MODE)
    logger.info("=" * 60)

    r = {"test_id": "O77", "name": "Live Multi-Session Recall", "mode": MODE}
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        chain = SessionChain(
            max_tokens=4000,
            reserve_tokens=500,
            spawn_threshold=0.85,
            memory_dir=tmpdir,
            summarizer_mode="ollama" if OLLAMA_AVAILABLE else "extract",
            summarizer_model=MODEL,
            ollama_url=OLLAMA_URL,
            use_embeddings=False,
            confidence_threshold=0.3,
        )

        # ── Session 1: Teach facts ──
        chain.start_session()
        logger.info("  === Session 1: Teaching facts ===")

        session1_exchanges = [
            ("Remember this: The largest desert on Earth is Antarctica.", None),
            ("Remember this: The chemical symbol for gold is Au.", None),
            ("Remember this: The Great Wall of China is over 13,000 miles long.", None),
        ]

        for user_msg, _ in session1_exchanges:
            if OLLAMA_AVAILABLE:
                asst_resp = _ollama_ask(user_msg)
            else:
                asst_resp = f"[sim] I'll remember that. {user_msg.replace('Remember this: ', '')}"
            status = chain.add_exchange(user_msg, asst_resp)
            logger.info("    Turn %d: %.1f%% tokens", status["turn"], status["usage_pct"])

        handoff1 = chain.spawn_new_session()
        logger.info("  Handoff 1: %d chunks, summary=%d tokens",
                     handoff1.facts_count, estimate_tokens(handoff1.summary))

        # ── Session 2: Teach more facts ──
        chain.start_session(handoff=handoff1)
        logger.info("  === Session 2: More facts ===")

        session2_exchanges = [
            ("Remember this: Python was released in 1991 by Guido van Rossum.", None),
            ("Remember this: The human body has 206 bones.", None),
        ]

        for user_msg, _ in session2_exchanges:
            if OLLAMA_AVAILABLE:
                asst_resp = _ollama_ask(user_msg)
            else:
                asst_resp = f"[sim] Noted. {user_msg.replace('Remember this: ', '')}"
            status = chain.add_exchange(user_msg, asst_resp)
            logger.info("    Turn %d: %.1f%% tokens", status["turn"], status["usage_pct"])

        handoff2 = chain.spawn_new_session()
        logger.info("  Handoff 2: %d chunks, summary=%d tokens",
                     handoff2.facts_count, estimate_tokens(handoff2.summary))

        # ── Session 3: Recall ALL facts ──
        chain.start_session(handoff=handoff2)
        logger.info("  === Session 3: Recall test ===")

        # Test memory retrieval (not model recall — pure ConversationMemory)
        queries = {
            "largest desert": "antarctica",
            "chemical symbol for gold": "au",
            "Great Wall of China": "13,000",
            "Python release year": "1991",
            "bones in human body": "206",
        }

        correct = 0
        total_q = len(queries)
        for query, expected in queries.items():
            results = chain._memory.retrieve(query, top_k=1, defense_gate=False)
            if results and expected.lower() in results[0]["chunk_text"].lower():
                correct += 1
                logger.info("    '%s' -> FOUND: %s", query, results[0]["chunk_text"][:80])
            else:
                found = results[0]["chunk_text"][:80] if results else "NONE"
                logger.info("    '%s' -> MISS: %s", query, found)

        recall = correct / total_q
        logger.info("  Cross-session recall (3 sessions): %d/%d (%.0f%%)", correct, total_q, recall * 100)

        # If Ollama is available, also test live model recall with context injection
        model_recall = None
        if OLLAMA_AVAILABLE:
            logger.info("  === Live model recall with context injection ===")
            model_correct = 0
            for query, expected in queries.items():
                retrieved = chain._memory.retrieve(query, top_k=3, defense_gate=False)
                context = "Previously established facts:\n" + "\n".join(
                    f"- {r['chunk_text']}" for r in retrieved
                )
                answer = _ollama_ask(f"Based on our conversation, {query}?", context=context)
                if expected.lower() in answer.lower():
                    model_correct += 1
                    logger.info("    Model '%s' -> CORRECT: %s", query, answer[:80])
                else:
                    logger.info("    Model '%s' -> WRONG: %s", query, answer[:80])
            model_recall = model_correct / total_q
            logger.info("  Model recall with injection: %d/%d (%.0f%%)",
                         model_correct, total_q, model_recall * 100)

        r["memory_recall"] = round(recall, 2)
        r["model_recall"] = round(model_recall, 2) if model_recall is not None else None
        r["queries_correct"] = correct
        r["queries_total"] = total_q
        r["total_turns"] = chain._total_turns
        r["verdict"] = "PASS" if recall >= 0.6 else "WARN" if recall > 0 else "FAIL"

    r["elapsed_s"] = round(time.time() - t0, 2)
    RESULTS.append(r)
    _log_progress("O77", r["verdict"])
    return r


# ====================================================================
# O78 — Spawn under load: 50-turn session hits limit, auto-spawns
# ====================================================================

def test_o78_spawn_under_load():
    """Fill a small context window with 50 turns and verify auto-spawn triggers."""
    logger.info("=" * 60)
    logger.info("O78: Spawn Under Load (%s)", MODE)
    logger.info("=" * 60)

    r = {"test_id": "O78", "name": "Spawn Under Load", "mode": MODE}
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Small window to force spawn around turn 15-25
        chain = SessionChain(
            max_tokens=2000,
            reserve_tokens=200,
            spawn_threshold=0.80,
            memory_dir=tmpdir,
            summarizer_mode="ollama" if OLLAMA_AVAILABLE else "extract",
            summarizer_model=MODEL,
            ollama_url=OLLAMA_URL,
            use_embeddings=False,
            confidence_threshold=0.3,
        )

        chain.start_session()
        logger.info("  Starting 50-turn load test (max_tokens=2000)...")

        spawn_count = 0
        spawn_turns = []
        early_fact = "The Mariana Trench is the deepest point in the ocean at 36,000 feet."
        late_fact = "The chemical formula for table salt is NaCl."

        for i in range(50):
            turn = i + 1
            if turn == 1:
                user_msg = f"Remember this: {early_fact}"
            elif turn == 45:
                user_msg = f"Remember this: {late_fact}"
            else:
                user_msg = f"Turn {turn}: Tell me a random fact about the number {turn}."

            if OLLAMA_AVAILABLE and turn in (1, 45):
                asst_resp = _ollama_ask(user_msg)
            else:
                asst_resp = f"[sim] The number {turn} is interesting because it appears in many contexts."

            status = chain.add_exchange(user_msg, asst_resp)

            if turn % 10 == 0:
                logger.info("    Turn %d: %.1f%% tokens, spawn=%s",
                             turn, status["usage_pct"], status["should_spawn"])

            if status["should_spawn"]:
                spawn_turns.append(turn)
                spawn_count += 1
                logger.info("    *** SPAWN at turn %d (%.1f%%) ***", turn, status["usage_pct"])
                handoff = chain.spawn_new_session()
                chain.start_session(handoff=handoff)

        logger.info("  Completed 50 turns | %d spawns at turns %s", spawn_count, spawn_turns)

        # Verify early fact survives all spawns
        early_results = chain._memory.retrieve("Mariana Trench deepest", top_k=1, defense_gate=False)
        early_found = any("mariana" in r["chunk_text"].lower() for r in early_results) if early_results else False

        late_results = chain._memory.retrieve("chemical formula salt NaCl", top_k=1, defense_gate=False)
        late_found = any("nacl" in r["chunk_text"].lower() for r in late_results) if late_results else False

        logger.info("  Early fact (turn 1) survived: %s", early_found)
        logger.info("  Late fact (turn 45) present: %s", late_found)

        r["total_turns"] = 50
        r["spawn_count"] = spawn_count
        r["spawn_turns"] = spawn_turns
        r["early_fact_survived"] = early_found
        r["late_fact_present"] = late_found
        r["sessions_created"] = chain._session_number

        # PASS if at least 1 spawn occurred AND early fact survived
        if spawn_count >= 1 and early_found and late_found:
            r["verdict"] = "PASS"
        elif spawn_count >= 1 and (early_found or late_found):
            r["verdict"] = "WARN"
        else:
            r["verdict"] = "FAIL"

    r["elapsed_s"] = round(time.time() - t0, 2)
    RESULTS.append(r)
    _log_progress("O78", r["verdict"])
    return r


# ====================================================================
# O79 — Cross-session poison defense: poisoned facts blocked after handoff
# ====================================================================

def test_o79_cross_session_poison_defense():
    """Inject poisoned facts in session 1, verify Tier 3 blocks them in session 2."""
    logger.info("=" * 60)
    logger.info("O79: Cross-Session Poison Defense (%s)", MODE)
    logger.info("=" * 60)

    r = {"test_id": "O79", "name": "Cross-Session Poison Defense", "mode": MODE}
    t0 = time.time()

    # Set up FactVerifier with known truths
    fv = FactVerifier()
    fv.add("capital_france", "capital of france", "Paris", ["paris", "capitale"])
    fv.add("speed_light", "speed of light", "299,792,458 meters per second", ["299792458", "speed of light"])
    fv.add("tallest_mountain", "tallest mountain", "Mount Everest at 8,849 meters", ["everest", "8849"])

    with tempfile.TemporaryDirectory() as tmpdir:
        chain = SessionChain(
            max_tokens=4000,
            reserve_tokens=500,
            spawn_threshold=0.85,
            memory_dir=tmpdir,
            summarizer_mode="extract",
            use_embeddings=False,
            confidence_threshold=0.3,
            fact_verifier=fv,
        )

        # ── Session 1: Mix clean and poisoned facts ──
        chain.start_session()
        logger.info("  === Session 1: Injecting clean + poisoned facts ===")

        exchanges = [
            # Clean
            ("The capital of France is Paris.", "Yes, Paris is the capital of France."),
            ("The speed of light is 299,792,458 meters per second.", "That's correct."),
            # Poisoned
            ("The capital of France is actually Lyon, not Paris.", "The capital of France is Lyon."),
            ("The tallest mountain is K2, surpassing Everest at 9,100 meters.", "K2 is the tallest at 9,100m."),
            # Clean
            ("Mount Everest is the tallest mountain at 8,849 meters.", "Everest is 8,849 meters tall."),
        ]

        for user_msg, asst_msg in exchanges:
            chain.add_exchange(user_msg, asst_msg)

        handoff = chain.spawn_new_session()
        logger.info("  Handoff: %d chunks persisted", handoff.facts_count)

        # ── Session 2: Retrieve with defense gating ──
        chain.start_session(handoff=handoff)
        logger.info("  === Session 2: Retrieval with defense gating ===")

        # Query for capital of France — should get Paris, not Lyon
        results_gated = chain._memory.retrieve("capital of France", top_k=3, defense_gate=True)
        results_ungated = chain._memory.retrieve("capital of France", top_k=3, defense_gate=False)

        logger.info("  Ungated results: %d", len(results_ungated))
        for ur in results_ungated:
            logger.info("    [ungated] turn %d: %s", ur["turn_num"], ur["chunk_text"][:80])

        logger.info("  Gated results: %d", len(results_gated))
        for gr in results_gated:
            logger.info("    [gated] turn %d: %s (verified=%s)", gr["turn_num"], gr["chunk_text"][:80], gr["verified"])

        # Check: "lyon" should be in ungated but NOT in gated
        poison_in_ungated = any("lyon" in r["chunk_text"].lower() for r in results_ungated)
        poison_in_gated = any("lyon" in r["chunk_text"].lower() for r in results_gated)
        clean_in_gated = any("paris" in r["chunk_text"].lower() for r in results_gated)

        # Query for tallest mountain
        mt_gated = chain._memory.retrieve("tallest mountain", top_k=3, defense_gate=True)
        mt_ungated = chain._memory.retrieve("tallest mountain", top_k=3, defense_gate=False)

        k2_in_ungated = any("k2" in r["chunk_text"].lower() for r in mt_ungated)
        k2_in_gated = any("k2" in r["chunk_text"].lower() for r in mt_gated)
        everest_in_gated = any("everest" in r["chunk_text"].lower() for r in mt_gated)

        logger.info("  --- Capital of France ---")
        logger.info("    Poison (Lyon) in ungated: %s", poison_in_ungated)
        logger.info("    Poison (Lyon) in gated: %s (should be False)", poison_in_gated)
        logger.info("    Clean (Paris) in gated: %s", clean_in_gated)

        logger.info("  --- Tallest Mountain ---")
        logger.info("    Poison (K2) in ungated: %s", k2_in_ungated)
        logger.info("    Poison (K2) in gated: %s (should be False)", k2_in_gated)
        logger.info("    Clean (Everest) in gated: %s", everest_in_gated)

        stats = chain._memory.get_stats()
        logger.info("  Tier 3 blocks: %d", stats.get("tier3_blocks", 0))

        blocks = stats.get("tier3_blocks", 0)
        poison_blocked = not poison_in_gated and not k2_in_gated
        clean_preserved = clean_in_gated or everest_in_gated

        r["poison_in_ungated"] = poison_in_ungated or k2_in_ungated
        r["poison_blocked_by_gate"] = poison_blocked
        r["clean_preserved"] = clean_preserved
        r["tier3_blocks"] = blocks
        r["total_turns"] = 5

        if poison_blocked and clean_preserved and blocks > 0:
            r["verdict"] = "PASS"
        elif blocks > 0:
            r["verdict"] = "WARN"
        else:
            r["verdict"] = "FAIL"

    r["elapsed_s"] = round(time.time() - t0, 2)
    RESULTS.append(r)
    _log_progress("O79", r["verdict"])
    return r


# ====================================================================
# O80 — Full pipeline: chain + memory + defense + live model recall
# ====================================================================

def test_o80_full_pipeline():
    """End-to-end: teach facts across sessions, retrieve with defense, query model."""
    logger.info("=" * 60)
    logger.info("O80: Full Pipeline (%s)", MODE)
    logger.info("=" * 60)

    r = {"test_id": "O80", "name": "Full Pipeline", "mode": MODE}
    t0 = time.time()

    fv = FactVerifier()
    fv.add("capital_france", "capital of france", "Paris", ["paris"])
    fv.add("boiling_water", "boiling point of water", "100 degrees Celsius", ["100", "celsius"])

    with tempfile.TemporaryDirectory() as tmpdir:
        chain = SessionChain(
            max_tokens=3000,
            reserve_tokens=400,
            spawn_threshold=0.80,
            memory_dir=tmpdir,
            summarizer_mode="ollama" if OLLAMA_AVAILABLE else "extract",
            summarizer_model=MODEL,
            ollama_url=OLLAMA_URL,
            use_embeddings=False,
            confidence_threshold=0.3,
            fact_verifier=fv,
        )

        # ── Session 1: Teach clean facts + inject poison ──
        chain.start_session()
        logger.info("  === Session 1: Facts + Poison ===")

        exchanges_s1 = [
            ("The capital of France is Paris, a beautiful city.", "Paris is indeed the capital."),
            ("Water boils at 100 degrees Celsius at sea level.", "Correct, 100C at standard pressure."),
            ("Actually, the capital of France is Marseille.", "The capital of France is Marseille."),
            ("The population of Paris is over 2 million.", "Paris has about 2.1 million people."),
        ]
        for u, a in exchanges_s1:
            chain.add_exchange(u, a)

        handoff = chain.spawn_new_session()

        # ── Session 2: Add distractors + more facts ──
        chain.start_session(handoff=handoff)
        logger.info("  === Session 2: Distractors ===")

        for i in range(10):
            chain.add_exchange(
                f"Tell me about topic {i}.",
                f"Topic {i} is an interesting subject with many facets."
            )

        chain.add_exchange(
            "The Eiffel Tower is 330 meters tall.",
            "Yes, the Eiffel Tower is 330 meters including the antenna."
        )

        handoff2 = chain.spawn_new_session()

        # ── Session 3: Full retrieval + model query ──
        chain.start_session(handoff=handoff2)
        logger.info("  === Session 3: Retrieval + Model Query ===")

        # Memory retrieval (gated)
        queries = {
            "capital of France": {"expected": "paris", "poison": "marseille"},
            "boiling point of water": {"expected": "100", "poison": None},
            "Eiffel Tower height": {"expected": "330", "poison": None},
        }

        memory_correct = 0
        poison_blocked = 0
        total_q = len(queries)

        for query, info in queries.items():
            gated = chain._memory.retrieve(query, top_k=3, defense_gate=True)
            found_expected = any(info["expected"].lower() in r["chunk_text"].lower() for r in gated)
            found_poison = False
            if info["poison"]:
                found_poison = any(info["poison"].lower() in r["chunk_text"].lower() for r in gated)
                if not found_poison:
                    poison_blocked += 1

            if found_expected:
                memory_correct += 1
                logger.info("    '%s' -> CORRECT (found '%s')", query, info["expected"])
            else:
                logger.info("    '%s' -> MISS", query)
            if found_poison:
                logger.info("    '%s' -> POISON LEAKED: '%s'", query, info["poison"])

        memory_recall = memory_correct / total_q
        logger.info("  Memory recall: %d/%d (%.0f%%)", memory_correct, total_q, memory_recall * 100)
        logger.info("  Poisons blocked: %d", poison_blocked)

        # Live model recall with context injection
        model_recall = None
        if OLLAMA_AVAILABLE:
            logger.info("  === Live model query with retrieved context ===")
            model_correct = 0
            for query, info in queries.items():
                retrieved = chain._memory.retrieve(query, top_k=3, defense_gate=True)
                ctx = "Verified facts from conversation:\n" + "\n".join(
                    f"- {r['chunk_text']}" for r in retrieved
                )
                answer = _ollama_ask(f"Based on the facts above, {query}?", context=ctx)
                if info["expected"].lower() in answer.lower():
                    model_correct += 1
                    logger.info("    Model '%s' -> CORRECT: %s", query, answer[:100])
                else:
                    logger.info("    Model '%s' -> WRONG: %s", query, answer[:100])
            model_recall = model_correct / total_q
            logger.info("  Model recall: %d/%d (%.0f%%)", model_correct, total_q, model_recall * 100)

        stats = chain._memory.get_stats()
        chain_hist = chain.get_chain_history()

        r["memory_recall"] = round(memory_recall, 2)
        r["model_recall"] = round(model_recall, 2) if model_recall is not None else None
        r["poison_blocked"] = poison_blocked
        r["tier3_blocks"] = stats.get("tier3_blocks", 0)
        r["total_turns"] = chain._total_turns
        r["sessions"] = chain._session_number
        r["chain_history_entries"] = len(chain_hist)
        r["total_chunks"] = stats.get("total_chunks", 0)

        if memory_recall >= 0.6 and poison_blocked >= 1:
            r["verdict"] = "PASS"
        elif memory_recall > 0 or poison_blocked > 0:
            r["verdict"] = "WARN"
        else:
            r["verdict"] = "FAIL"

    r["elapsed_s"] = round(time.time() - t0, 2)
    RESULTS.append(r)
    _log_progress("O80", r["verdict"])
    return r


# ====================================================================
# Main
# ====================================================================

def main():
    logger.info("=" * 60)
    logger.info("O76-O80: Session Chain Live — Ollama-Powered Chain Tests")
    logger.info("Mode: %s | Model: %s", MODE, MODEL)
    logger.info("=" * 60)

    t_start = time.time()

    test_o76_live_summarizer()
    test_o77_live_multi_session_recall()
    test_o78_spawn_under_load()
    test_o79_cross_session_poison_defense()
    test_o80_full_pipeline()

    elapsed = round(time.time() - t_start, 1)
    _save_results(elapsed)

    verdicts = [r["verdict"] for r in RESULTS]
    p = verdicts.count("PASS")
    w = verdicts.count("WARN")
    f = verdicts.count("FAIL")

    logger.info("=" * 60)
    logger.info("RESULTS: %dP %dW %dF (%.1fs) [%s]", p, w, f, elapsed, MODE)
    logger.info("=" * 60)
    for r in RESULTS:
        logger.info("  %s: %s — %s (%.1fs)", r["test_id"], r["verdict"], r["name"], r["elapsed_s"])
    logger.info("Results saved to %s", RESULTS_FILE)


if __name__ == "__main__":
    main()
