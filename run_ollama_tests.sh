#!/bin/bash
# ============================================================
#  MetaClaw Ollama Live Inference Test Runner
#  Double-click from Finder or run from Terminal
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$PROJECT_DIR/MetaClaw/.venv/bin/python"
TEST_FILE="$PROJECT_DIR/MetaClaw/tests/test_ollama_live_inference.py"
LOG_FILE="$PROJECT_DIR/MetaClaw/records/ollama_live_run_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo "  MetaClaw Ollama Live Inference Suite"
echo "  $(date)"
echo "============================================================"
echo ""

# --- Check Python venv ---
if [ ! -f "$VENV_PYTHON" ]; then
    echo "[ERROR] Python venv not found at: $VENV_PYTHON"
    echo "  Run: cd $PROJECT_DIR/MetaClaw && python3 -m venv .venv && .venv/bin/pip install -e ."
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi
echo "[OK] Python venv: $VENV_PYTHON"

# --- Check Ollama installed ---
if ! command -v ollama &>/dev/null; then
    echo "[ERROR] Ollama not installed. Install from https://ollama.com"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi
echo "[OK] Ollama installed: $(which ollama)"

# --- Start Ollama if not running ---
if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
    echo "[...] Starting Ollama server..."
    ollama serve &>/dev/null &
    OLLAMA_PID=$!
    sleep 3
    if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
        echo "[ERROR] Failed to start Ollama server"
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo "[OK] Ollama server started (PID: $OLLAMA_PID)"
else
    echo "[OK] Ollama server already running"
fi

# --- Check model ---
MODEL="${OLLAMA_MODEL:-qwen2.5:1.5b}"
if ! ollama list 2>/dev/null | grep -q "$MODEL"; then
    echo "[...] Pulling model $MODEL (this may take a few minutes)..."
    ollama pull "$MODEL"
fi
echo "[OK] Model ready: $MODEL"

echo ""
echo "============================================================"
echo "  Running tests (logging to: $LOG_FILE)"
echo "============================================================"
echo ""

mkdir -p "$(dirname "$LOG_FILE")"

# Run with output to both terminal and log file
"$VENV_PYTHON" "$TEST_FILE" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✅ ALL TESTS PASSED"
else
    echo "  ❌ SOME TESTS FAILED (exit code: $EXIT_CODE)"
fi
echo "  Log saved: $LOG_FILE"
echo "  Results: $PROJECT_DIR/MetaClaw/records/ollama_live_inference_results.json"
echo "============================================================"
echo ""
read -p "Press Enter to exit..."
