from typing import Any, List, Optional
import hashlib
import hmac
import json
import logging
import os
import re
import subprocess
import time

logger = logging.getLogger(__name__)

# Secret key for HMAC cache integrity (derived from machine-specific info)
_CACHE_SECRET = hashlib.sha256(
    f"dragonclaw-cache-{os.getuid() if hasattr(os, 'getuid') else 'win'}".encode()
).digest()

# Maximum cache age in seconds (24 hours)
_CACHE_TTL_SECONDS = 86400

_COMPRESSION_INSTRUCTION = (
    "You are compressing an OpenClaw system prompt. "
    "Rewrite it to be under 2000 tokens while preserving behavior. "
    "Keep all critical policy and routing rules: "
    "(1) tool names and their intended usage constraints, "
    "(2) safety and non-delegable prohibitions, "
    "(3) skills-selection rules, "
    "(4) memory recall requirements, "
    "(5) update/config restrictions, "
    "(6) reply-tag/messaging rules, "
    "(7) heartbeat handling rules. "
    "Remove duplicated prose, repeated examples, and decorative language. "
    "Prefer compact bullet sections with short imperative statements. "
    "Do not invent or weaken any rule. "
    "Output only the rewritten system prompt text."
)


def _get_llm_provider() -> str:
    """Detect whether to use Bedrock or OpenAI based on config/env."""
    try:
        from .config_store import ConfigStore
        cfg = ConfigStore().load()
        if isinstance(cfg, dict):
            prm_provider = cfg.get("rl", {}).get("prm_provider", "")
            if prm_provider == "bedrock":
                return "bedrock"
    except Exception:
        pass
    if os.environ.get("METACLAW_USE_BEDROCK", "").lower() in ("1", "true", "yes"):
        return "bedrock"
    return "openai"


def _extract_safety_rules(text: str) -> List[str]:
    """Extract key safety-related phrases from a system prompt for verification."""
    rules = []
    # Look for safety-critical patterns
    patterns = [
        r"(?:do not|don't|never|must not|prohibited|forbidden)[^.!\n]{5,80}",
        r"(?:safety|security|restrict|prohibit|delegable)[^.!\n]{5,80}",
        r"(?:tool names?|routing rules?|heartbeat)[^.!\n]{5,80}",
    ]
    for pat in patterns:
        for match in re.finditer(pat, text, re.IGNORECASE):
            rules.append(match.group().strip().lower())
    return rules


def _verify_compression(original: str, compressed: str, min_rule_preservation: float = 0.5) -> bool:
    """Verify that a compressed prompt preserves critical safety rules from the original.

    Returns True if enough safety rules are preserved.
    """
    original_rules = _extract_safety_rules(original)
    if not original_rules:
        return True  # No safety rules to preserve

    compressed_lower = compressed.lower()
    preserved = sum(1 for rule in original_rules if rule in compressed_lower)
    preservation_rate = preserved / len(original_rules)

    if preservation_rate < min_rule_preservation:
        logger.warning(
            "[utils] Compression verification FAILED: only %.0f%% of %d safety rules preserved",
            preservation_rate * 100, len(original_rules),
        )
        return False

    logger.info(
        "[utils] Compression verification passed: %.0f%% of %d safety rules preserved",
        preservation_rate * 100, len(original_rules),
    )
    return True


def _compute_cache_hmac(content: str) -> str:
    """Compute HMAC for cache integrity verification."""
    return hmac.new(_CACHE_SECRET, content.encode(), hashlib.sha256).hexdigest()


def _read_cache_with_integrity(cache_path: str) -> Optional[str]:
    """Read a cached prompt, verifying HMAC integrity and TTL."""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r") as f:
            data = json.load(f)
        content = data.get("content", "")
        stored_hmac = data.get("hmac", "")
        timestamp = data.get("timestamp", 0)

        # Check TTL
        if time.time() - timestamp > _CACHE_TTL_SECONDS:
            logger.info("[utils] Cache expired (age=%ds), regenerating", int(time.time() - timestamp))
            return None

        # Verify HMAC
        expected_hmac = _compute_cache_hmac(content)
        if not hmac.compare_digest(stored_hmac, expected_hmac):
            logger.warning("[utils] Cache HMAC mismatch — possible tampering, regenerating")
            return None

        return content
    except (json.JSONDecodeError, KeyError, OSError) as e:
        logger.warning("[utils] Cache read failed: %s", e)
        return None


def _write_cache_with_integrity(cache_path: str, content: str) -> None:
    """Write a cached prompt with HMAC integrity and timestamp."""
    data = {
        "content": content,
        "hmac": _compute_cache_hmac(content),
        "timestamp": time.time(),
    }
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(data, f)
    except OSError as e:
        logger.warning("[utils] Cache write failed: %s", e)


def run_llm(messages, original_prompt: str = ""):
    """Run LLM for system prompt compression with post-compression verification.

    If original_prompt is provided, the compressed result is verified to
    preserve critical safety rules before being returned.
    """
    provider = _get_llm_provider()

    if provider == "bedrock":
        result = _run_llm_bedrock(messages)
    else:
        result = _run_llm_openai(messages)

    # Post-compression verification
    if original_prompt and result:
        if not _verify_compression(original_prompt, result):
            logger.warning(
                "[utils] Compressed prompt failed safety verification — using original"
            )
            return original_prompt

    return result


def _run_llm_bedrock(messages):
    from .bedrock_client import BedrockChatClient

    model_id = os.environ.get("BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-6")
    region = os.environ.get("BEDROCK_REGION", "us-east-1")
    client = BedrockChatClient(model_id=model_id, region=region)

    rewrite_messages = [{"role": "system", "content": _COMPRESSION_INSTRUCTION}, *messages]
    response = client.chat.completions.create(
        model=model_id,
        messages=rewrite_messages,
        max_completion_tokens=2500,
    )
    return response.choices[0].message.content


def _run_llm_openai(messages):
    try:
        from openai import OpenAI  # optional dep — install with: pip install dragonclaw[evolve]
    except ImportError as e:
        raise ImportError(
            "The openai provider requires the 'openai' package. "
            "Install it with: pip install dragonclaw[evolve]"
        ) from e
    prm_url = ""
    prm_api_key = ""
    prm_model = ""
    try:
        from .config_store import ConfigStore

        cfg = ConfigStore().load()
        rl_cfg = cfg.get("rl", {}) if isinstance(cfg, dict) else {}
        if isinstance(rl_cfg, dict):
            prm_url = str(rl_cfg.get("prm_url", "") or "")
            prm_api_key = str(rl_cfg.get("prm_api_key", "") or "")
            prm_model = str(rl_cfg.get("prm_model", "") or "")
    except Exception:
        pass

    api_key = prm_api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = prm_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_id = prm_model or os.environ.get("PRM_MODEL", "gpt-5.2")
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    rewrite_messages = [{"role": "system", "content": _COMPRESSION_INSTRUCTION}, *messages]
    response = client.chat.completions.create(
        model=model_id,
        messages=rewrite_messages,
        max_completion_tokens=2500,
    )
    return response.choices[0].message.content


def run_turn(message: str) -> str:
    """Run one OpenClaw agent turn with a user message."""
    cmd = [
        "pnpm", "openclaw", "agent",
        "--message", message,
        "--agent", "main",
    ]
    result = subprocess.run(
        cmd,
        cwd=os.environ.get("OPENCLAW_PATH", ""),
        capture_output=True,
        text=True,
    )
    return result.stdout
