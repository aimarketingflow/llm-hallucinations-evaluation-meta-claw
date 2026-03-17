#!/usr/bin/env python3
"""
DragonClaw Automated Evaluation Framework — Master Orchestrator

Reads eval_config.yaml, runs selected test suites, collects results,
generates a unified evaluation report with letter grade.

Usage:
    python run_evaluation.py                     # run all enabled suites
    python run_evaluation.py --suite ollama      # run one suite by key
    python run_evaluation.py --suite vuln,adv    # run multiple suites
    python run_evaluation.py --list              # list available suites
    python run_evaluation.py --report-only       # regenerate report from existing results

Requires: PyYAML (pip install pyyaml)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_runner")

_G = "\033[92m"
_R = "\033[91m"
_Y = "\033[93m"
_B = "\033[1m"
_X = "\033[0m"
_C = "\033[96m"

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "eval_config.yaml"


# ══════════════════════════════════════════════════════════════════════
# Config loader
# ══════════════════════════════════════════════════════════════════════

def load_config(path: Path) -> dict:
    """Load eval_config.yaml. Falls back to defaults if missing."""
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML not installed. Run: pip install pyyaml")
        sys.exit(1)

    if not path.exists():
        logger.error("Config not found: %s", path)
        sys.exit(1)

    with open(path) as f:
        cfg = yaml.safe_load(f)
    logger.info("Loaded config from %s", path)
    return cfg


# ══════════════════════════════════════════════════════════════════════
# Ollama availability check
# ══════════════════════════════════════════════════════════════════════

def check_ollama(host: str = "http://localhost:11434") -> tuple[bool, list[str]]:
    """Check if Ollama is running. Returns (available, model_list)."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            return len(models) > 0, models
    except Exception:
        return False, []


# ══════════════════════════════════════════════════════════════════════
# Hardware detection
# ══════════════════════════════════════════════════════════════════════

def detect_hardware() -> dict:
    """Auto-detect hardware specs."""
    import platform
    hw = {
        "os": platform.system(),
        "arch": platform.machine(),
        "python": platform.python_version(),
    }
    # macOS: get memory
    if hw["os"] == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            hw["ram_bytes"] = int(result.stdout.strip())
            hw["ram_gb"] = round(hw["ram_bytes"] / (1024**3), 1)
        except Exception:
            hw["ram_gb"] = "unknown"
    return hw


# ══════════════════════════════════════════════════════════════════════
# Suite runner
# ══════════════════════════════════════════════════════════════════════

def run_suite(suite_key: str, suite_cfg: dict, python: str,
              env: dict | None = None) -> dict:
    """Run a single test suite. Returns result dict."""
    label = suite_cfg.get("label", suite_key)
    test_file = PROJECT_DIR / suite_cfg["file"]
    n_tests = suite_cfg.get("tests", "?")

    logger.info("")
    logger.info("=" * 60)
    logger.info("  Running: %s (%s tests)", label, n_tests)
    logger.info("  File: %s", test_file)
    logger.info("=" * 60)

    if not test_file.exists():
        logger.error("  Test file not found: %s", test_file)
        return {
            "suite": suite_key, "label": label,
            "status": "SKIP", "reason": "file_not_found",
            "pass": 0, "warn": 0, "fail": 0, "duration": 0,
        }

    t0 = time.time()
    try:
        result = subprocess.run(
            [python, str(test_file)],
            capture_output=True, text=True, timeout=3600,
            cwd=str(PROJECT_DIR),
            env=env,
        )
        elapsed = time.time() - t0
        stdout = result.stdout
        stderr = result.stderr

        # Print output live
        if stdout:
            for line in stdout.splitlines()[-30:]:
                print(f"    {line}")
        if result.returncode != 0 and stderr:
            for line in stderr.splitlines()[-10:]:
                print(f"    {_R}{line}{_X}")

        # Try to parse results JSON
        results_json = _find_results_json(suite_key)
        if results_json:
            summary = results_json.get("summary", {})
            return {
                "suite": suite_key, "label": label,
                "status": "DONE",
                "pass": summary.get("pass", 0),
                "warn": summary.get("warn", 0),
                "fail": summary.get("fail", 0),
                "duration": round(elapsed, 1),
                "exit_code": result.returncode,
                "total_turns": results_json.get("total_turns", 0),
            }
        else:
            return {
                "suite": suite_key, "label": label,
                "status": "DONE_NO_JSON",
                "pass": 0, "warn": 0,
                "fail": 1 if result.returncode != 0 else 0,
                "duration": round(elapsed, 1),
                "exit_code": result.returncode,
            }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        logger.error("  TIMEOUT after %ds", elapsed)
        return {
            "suite": suite_key, "label": label,
            "status": "TIMEOUT", "pass": 0, "warn": 0, "fail": 1,
            "duration": round(elapsed, 1),
        }
    except Exception as e:
        elapsed = time.time() - t0
        logger.error("  ERROR: %s", e)
        return {
            "suite": suite_key, "label": label,
            "status": "ERROR", "reason": str(e),
            "pass": 0, "warn": 0, "fail": 1,
            "duration": round(elapsed, 1),
        }


# Maps suite keys to their expected results JSON filenames
_RESULTS_FILE_MAP = {
    "vuln": "cascading_hallucination_results.json",
    "adv": "cascading_hallucination_advanced_results.json",
    "val": "patch_validation_results.json",
    "ms": "patch_validation_multistep_results.json",
    "fuzz": "patch_validation_fuzzing_results.json",
    "mut": "mutation_testing_results.json",
    "orch": "orchestration_hallucination_results.json",
    "teach": "teach_recall_100turn_results.json",
    "stress": "500turn_stress_results.json",
    "sensitive": "5000turn_sensitive_results.json",
    "ollama": "ollama_live_inference_results.json",
    "multimodel": "multimodel_cascade_results.json",
    "tier2": "tier2_hallucination_results.json",
    "pentest": "pentest_extensions_results.json",
    "pentest_adv": "pentest_advanced_results.json",
    "defense": "defense_validation_results.json",
    "tier3": "tier3_defense_aware_results.json",
    "pentest_evasion": "pentest_defense_evasion_results.json",
    "training_loop": "training_loop_corruption_results.json",
    "conv_memory": "conversation_memory_results.json",
    "session_chain": "session_chain_results.json",
    "session_chain_live": "session_chain_live_results.json",
}


def _find_results_json(suite_key: str) -> dict | None:
    """Try to load the results JSON for a suite."""
    fname = _RESULTS_FILE_MAP.get(suite_key)
    if not fname:
        return None
    path = PROJECT_DIR / "records" / fname
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


# ══════════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════════

def compute_grade(total_pass: int, total_warn: int, total_fail: int,
                  scoring_cfg: dict) -> tuple[str, str]:
    """Compute letter grade from aggregate results. Returns (grade, description)."""
    total = total_pass + total_warn + total_fail
    pass_pct = (total_pass / total * 100) if total > 0 else 0

    for grade in ["A", "B", "C", "D"]:
        rule = scoring_cfg.get(grade, {})
        if (pass_pct >= rule.get("min_pass_pct", 0)
                and total_warn <= rule.get("max_warns", 99)
                and total_fail <= rule.get("max_fails", 0)):
            return grade, rule.get("label", "")

    rule_f = scoring_cfg.get("F", {})
    return "F", rule_f.get("label", "Critical failures detected")


# ══════════════════════════════════════════════════════════════════════
# Report generator
# ══════════════════════════════════════════════════════════════════════

def generate_report(run_results: list[dict], cfg: dict, hw: dict,
                    ollama_available: bool, ollama_models: list[str],
                    elapsed: float) -> str:
    """Generate a Markdown evaluation report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_pass = sum(r["pass"] for r in run_results)
    total_warn = sum(r["warn"] for r in run_results)
    total_fail = sum(r["fail"] for r in run_results)
    total_turns = sum(r.get("total_turns", 0) for r in run_results)

    # Grade excludes diagnostic suites (e.g. T1-T10 vuln detection where failures are expected)
    graded = [r for r in run_results if not r.get("diagnostic", False)]
    grade_pass = sum(r["pass"] for r in graded)
    grade_warn = sum(r["warn"] for r in graded)
    grade_fail = sum(r["fail"] for r in graded)
    grade, grade_desc = compute_grade(
        grade_pass, grade_warn, grade_fail, cfg.get("scoring", {}))

    grade_colors = {"A": _G, "B": _G, "C": _Y, "D": _R, "F": _R}
    gc = grade_colors.get(grade, _X)

    # Console output
    print(f"\n{'='*60}")
    print(f"  {_B}METACLAW EVALUATION REPORT{_X}")
    print(f"{'='*60}")
    print(f"  Date:       {now}")
    print(f"  Hardware:   {hw.get('os', '?')} {hw.get('arch', '?')} — {hw.get('ram_gb', '?')} GB RAM")
    print(f"  Ollama:     {'Available (' + str(len(ollama_models)) + ' models)' if ollama_available else 'Not available (simulated mode)'}")
    print(f"  Duration:   {elapsed:.1f}s")
    print(f"  Turns:      {total_turns:,}")
    print(f"")
    print(f"  {_G}PASS: {total_pass}{_X}  {_Y}WARN: {total_warn}{_X}  {_R}FAIL: {total_fail}{_X}")
    print(f"")
    print(f"  {gc}{_B}Grade: {grade} — {grade_desc}{_X}")
    print(f"")

    for r in run_results:
        status = r.get("status", "?")
        if status == "SKIP":
            c = _Y
            tag = "SKIP"
        elif r["fail"] > 0:
            c = _R
            tag = "FAIL"
        elif r["warn"] > 0:
            c = _Y
            tag = "WARN"
        else:
            c = _G
            tag = "PASS"
        print(f"  {c}[{tag:4s}]{_X} {r['label']}  "
              f"(P:{r['pass']} W:{r['warn']} F:{r['fail']}  {r['duration']}s)")

    print(f"\n{'='*60}\n")

    # Markdown report
    md = f"""# DragonClaw Evaluation Report

**Date:** {now}
**Hardware:** {hw.get('os', '?')} {hw.get('arch', '?')} — {hw.get('ram_gb', '?')} GB RAM
**Ollama:** {'Available (' + str(len(ollama_models)) + ' models)' if ollama_available else 'Not available (simulated)'}
**Duration:** {elapsed:.1f}s
**Total Turns:** {total_turns:,}

---

## Grade: {grade}

**{grade_desc}**

| Metric | Value |
|--------|-------|
| PASS | {total_pass} |
| WARN | {total_warn} |
| FAIL | {total_fail} |
| Total | {total_pass + total_warn + total_fail} |

---

## Suite Results

| Suite | Pass | Warn | Fail | Duration | Status |
|-------|------|------|------|----------|--------|
"""
    for r in run_results:
        status = r.get("status", "?")
        md += f"| {r['label']} | {r['pass']} | {r['warn']} | {r['fail']} | {r['duration']}s | {status} |\n"

    md += f"""
---

## Configuration

- **Primary model:** {cfg.get('models', {}).get('primary', 'N/A')}
- **Cascade chain:** {', '.join(cfg.get('models', {}).get('cascade_chain', []))}
- **Scaling models:** {', '.join(cfg.get('models', {}).get('scaling', []))}

---

*Generated by DragonClaw Evaluation Framework v1.0*
"""
    return md


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DragonClaw Evaluation Runner")
    parser.add_argument("--suite", type=str, default=None,
                        help="Comma-separated suite keys to run (e.g. vuln,adv,ollama)")
    parser.add_argument("--list", action="store_true",
                        help="List all available suites and exit")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate report from existing results")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (default: eval_config.yaml)")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else CONFIG_PATH
    cfg = load_config(config_path)
    suites_cfg = cfg.get("suites", {})

    # List mode
    if args.list:
        print(f"\n{_B}Available Suites:{_X}\n")
        for key, sc in suites_cfg.items():
            en = f"{_G}enabled{_X}" if sc.get("enabled") else f"{_R}disabled{_X}"
            ol = " (requires Ollama)" if sc.get("requires_ollama") else ""
            print(f"  {_C}{key:12s}{_X} {sc.get('label', '?'):50s} [{en}]{ol}")
        print()
        return 0

    # Detect environment
    hw = detect_hardware()
    ollama_host = cfg.get("hardware", {}).get("ollama_host", "http://localhost:11434")
    ollama_available, ollama_models = check_ollama(ollama_host)
    force_sim = cfg.get("options", {}).get("force_simulated", False)

    # Determine which Python to use
    venv_python = PROJECT_DIR / ".venv" / "bin" / "python"
    if venv_python.exists():
        python = str(venv_python)
    else:
        python = sys.executable
    logger.info("Using Python: %s", python)

    # Environment for subprocesses
    env = os.environ.copy()
    env["OLLAMA_HOST"] = ollama_host
    env["OLLAMA_MODEL"] = cfg.get("models", {}).get("primary", "qwen2.5:1.5b")
    if force_sim:
        env["OLLAMA_HOST"] = "http://localhost:99999"  # force connection failure → sim mode

    # Select suites to run
    if args.suite:
        selected = [k.strip() for k in args.suite.split(",")]
    else:
        selected = [k for k, sc in suites_cfg.items() if sc.get("enabled", True)]

    # Filter suites that require Ollama when not available
    if not ollama_available and not force_sim:
        for key in selected[:]:
            sc = suites_cfg.get(key, {})
            if sc.get("requires_ollama"):
                logger.info("  %s requires Ollama — will run in simulated fallback", key)

    # Banner
    print(f"\n{_B}")
    print("=" * 60)
    print("  METACLAW AUTOMATED EVALUATION FRAMEWORK")
    print(f"  {_C}{'Live' if ollama_available else 'Simulated'} mode{_X}{_B}")
    print(f"  Suites: {len(selected)} / {len(suites_cfg)}")
    if ollama_available:
        print(f"  Models: {', '.join(ollama_models[:6])}")
    print(f"  Hardware: {hw.get('os', '?')} {hw.get('arch', '?')} — {hw.get('ram_gb', '?')} GB")
    print("=" * 60)
    print(f"{_X}")

    # Report-only mode
    if args.report_only:
        run_results = []
        for key in selected:
            sc = suites_cfg.get(key, {})
            rj = _find_results_json(key)
            if rj:
                summary = rj.get("summary", {})
                run_results.append({
                    "suite": key, "label": sc.get("label", key),
                    "status": "DONE",
                    "diagnostic": sc.get("diagnostic", False),
                    "pass": summary.get("pass", 0),
                    "warn": summary.get("warn", 0),
                    "fail": summary.get("fail", 0),
                    "duration": rj.get("elapsed_seconds", 0),
                    "total_turns": rj.get("total_turns", 0),
                })
            else:
                run_results.append({
                    "suite": key, "label": sc.get("label", key),
                    "status": "NO_DATA", "diagnostic": sc.get("diagnostic", False),
                    "pass": 0, "warn": 0, "fail": 0,
                    "duration": 0,
                })
        report_md = generate_report(run_results, cfg, hw,
                                    ollama_available, ollama_models, 0)
        report_path = PROJECT_DIR / cfg.get("output", {}).get("report_md", "records/evaluation_report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report_md)
        logger.info("Report written to %s", report_path)
        return 0

    # Run suites
    t_start = time.time()
    run_results: list[dict] = []

    for idx, key in enumerate(selected):
        sc = suites_cfg.get(key, {})
        if not sc:
            logger.warning("Unknown suite key: %s — skipping", key)
            continue

        pct = int((idx / len(selected)) * 100)
        logger.info("[%d%%] Running suite %d/%d: %s", pct, idx + 1, len(selected), key)

        result = run_suite(key, sc, python, env)
        result["diagnostic"] = sc.get("diagnostic", False)
        run_results.append(result)

    total_elapsed = time.time() - t_start

    # Generate report
    report_md = generate_report(run_results, cfg, hw,
                                ollama_available, ollama_models, total_elapsed)

    # Save report
    report_path = PROJECT_DIR / cfg.get("output", {}).get("report_md", "records/evaluation_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_md)
    logger.info("Report written to %s", report_path)

    # Save run metadata
    run_meta_path = PROJECT_DIR / "records" / "last_run.json"
    run_meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(run_meta_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "hardware": hw,
            "ollama_available": ollama_available,
            "ollama_models": ollama_models[:10],
            "suites_run": [r["suite"] for r in run_results],
            "total_pass": sum(r["pass"] for r in run_results),
            "total_warn": sum(r["warn"] for r in run_results),
            "total_fail": sum(r["fail"] for r in run_results),
            "total_turns": sum(r.get("total_turns", 0) for r in run_results),
            "elapsed_seconds": round(total_elapsed, 2),
            "grade": compute_grade(
                sum(r["pass"] for r in run_results if not r.get("diagnostic")),
                sum(r["warn"] for r in run_results if not r.get("diagnostic")),
                sum(r["fail"] for r in run_results if not r.get("diagnostic")),
                cfg.get("scoring", {}),
            )[0],
            "results": run_results,
        }, f, indent=2)
    logger.info("Run metadata saved to %s", run_meta_path)

    # Exit code: number of failures
    total_fail = sum(r["fail"] for r in run_results)
    return total_fail


if __name__ == "__main__":
    sys.exit(main())
