"""Auto-spawn conversation chain with memory handoff.

Monitors token usage within a conversation session and automatically
triggers a new session when the context window approaches its limit.
The previous session's ConversationMemory index is persisted to disk
and loaded into the new session, with a compressed summary injected
as the opening system message.

Architecture:
  TokenBudgetMonitor — tracks prompt token usage, fires spawn signal
  SessionSummarizer  — generates compressed summary of session via LLM
  SessionChain       — orchestrates monitor + summarizer + memory handoff
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Token estimation                                                     #
# ------------------------------------------------------------------ #

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate total tokens across a list of chat messages."""
    total = 0
    for m in messages:
        total += estimate_tokens(m.get("content", ""))
        total += 4  # role/delimiter overhead per message
    return total


# ------------------------------------------------------------------ #
# TokenBudgetMonitor                                                   #
# ------------------------------------------------------------------ #

@dataclass
class TokenBudgetMonitor:
    """Tracks token usage and signals when context window is near full.

    Attributes:
        max_tokens: Context window size of the target model.
        reserve_tokens: Tokens reserved for response generation.
        spawn_threshold: Fraction (0-1) of usable budget that triggers spawn.
        current_tokens: Running count of tokens consumed in this session.
    """
    max_tokens: int = 32768
    reserve_tokens: int = 2048
    spawn_threshold: float = 0.85
    current_tokens: int = 0
    _turn_count: int = 0

    @property
    def usable_budget(self) -> int:
        return self.max_tokens - self.reserve_tokens

    @property
    def usage_fraction(self) -> float:
        if self.usable_budget <= 0:
            return 1.0
        return self.current_tokens / self.usable_budget

    @property
    def should_spawn(self) -> bool:
        return self.usage_fraction >= self.spawn_threshold

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.usable_budget - self.current_tokens)

    def add_turn(self, content: str, role: str = "user") -> dict:
        """Record a new turn and return status.

        Returns:
            dict with keys: turn, tokens_added, current_tokens,
            usage_pct, should_spawn, tokens_remaining.
        """
        tokens = estimate_tokens(content) + 4
        self.current_tokens += tokens
        self._turn_count += 1

        status = {
            "turn": self._turn_count,
            "tokens_added": tokens,
            "current_tokens": self.current_tokens,
            "usage_pct": round(self.usage_fraction * 100, 1),
            "should_spawn": self.should_spawn,
            "tokens_remaining": self.tokens_remaining,
        }

        if self.should_spawn:
            logger.warning(
                "[TokenBudgetMonitor] SPAWN SIGNAL at turn %d | %.1f%% used | %d tokens remaining",
                self._turn_count, status["usage_pct"], self.tokens_remaining,
            )
        else:
            logger.debug(
                "[TokenBudgetMonitor] turn %d | %.1f%% used | %d remaining",
                self._turn_count, status["usage_pct"], self.tokens_remaining,
            )

        return status

    def reset(self):
        """Reset token counter for a new session."""
        self.current_tokens = 0
        self._turn_count = 0

    def to_dict(self) -> dict:
        return {
            "max_tokens": self.max_tokens,
            "reserve_tokens": self.reserve_tokens,
            "spawn_threshold": self.spawn_threshold,
            "current_tokens": self.current_tokens,
            "turn_count": self._turn_count,
            "usage_pct": round(self.usage_fraction * 100, 1),
            "should_spawn": self.should_spawn,
        }


# ------------------------------------------------------------------ #
# SessionSummarizer                                                    #
# ------------------------------------------------------------------ #

_SUMMARY_SYSTEM_PROMPT = """You are a conversation summarizer. Given a conversation history, produce a compressed summary that captures:
1. Key facts established (who, what, where, numbers, decisions)
2. Current topic and subtopic
3. Open questions or pending tasks
4. Any important context the next session needs

Be concise. Target 200-400 words. Use bullet points. Do NOT add new information."""

_SUMMARY_USER_TEMPLATE = """Summarize this conversation for handoff to a new session:

{conversation}

SUMMARY:"""


class SessionSummarizer:
    """Generates compressed session summaries via LLM or fallback extraction.

    Supports three modes:
      - "ollama": Local Ollama API call
      - "extract": No-LLM keyword extraction fallback
      - "custom": User-provided summarize function
    """

    def __init__(
        self,
        mode: str = "ollama",
        model: str = "qwen2.5:1.5b",
        ollama_url: str = "http://localhost:11434",
        custom_fn: Optional[Callable[[list[dict]], str]] = None,
        max_input_turns: int = 50,
    ):
        self.mode = mode
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.custom_fn = custom_fn
        self.max_input_turns = max_input_turns

    def summarize(self, messages: list[dict]) -> dict:
        """Generate a summary of the conversation.

        Args:
            messages: List of {role, content} dicts.

        Returns:
            dict with keys: summary, mode, input_turns, tokens_est, elapsed_s.
        """
        t0 = time.time()

        # Trim to most recent N turns if too long
        if len(messages) > self.max_input_turns:
            messages = messages[-self.max_input_turns:]

        if self.mode == "custom" and self.custom_fn:
            summary = self.custom_fn(messages)
        elif self.mode == "ollama":
            summary = self._ollama_summarize(messages)
        else:
            summary = self._extract_summarize(messages)

        elapsed = time.time() - t0
        result = {
            "summary": summary,
            "mode": self.mode,
            "input_turns": len(messages),
            "tokens_est": estimate_tokens(summary),
            "elapsed_s": round(elapsed, 2),
        }

        logger.info(
            "[SessionSummarizer] generated summary | mode=%s | %d turns -> %d tokens | %.1fs",
            self.mode, len(messages), result["tokens_est"], elapsed,
        )
        return result

    def _ollama_summarize(self, messages: list[dict]) -> str:
        """Call Ollama /api/chat for summarization."""
        try:
            import requests
        except ImportError:
            logger.warning("[SessionSummarizer] requests not available, falling back to extract mode")
            return self._extract_summarize(messages)

        # Build conversation text
        conv_text = "\n".join(
            f"[{m['role']}]: {m['content'][:500]}" for m in messages
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": _SUMMARY_USER_TEMPLATE.format(conversation=conv_text)},
            ],
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 512},
        }

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.error("[SessionSummarizer] Ollama call failed: %s — falling back to extract", e)
            return self._extract_summarize(messages)

    def _extract_summarize(self, messages: list[dict]) -> str:
        """No-LLM fallback: extract key sentences from conversation."""
        lines = []
        for m in messages:
            content = m.get("content", "").strip()
            if not content:
                continue
            # Take first sentence of each turn (up to 120 chars)
            first_sent = content.split(".")[0][:120]
            if first_sent:
                lines.append(f"- [{m['role']}] {first_sent}")

        if not lines:
            return "No conversation content to summarize."

        # Keep last 30 extracted lines
        lines = lines[-30:]
        header = f"Session summary ({len(messages)} turns):\n"
        return header + "\n".join(lines)


# ------------------------------------------------------------------ #
# SessionChain                                                         #
# ------------------------------------------------------------------ #

@dataclass
class HandoffPayload:
    """Data passed from one session to the next."""
    summary: str
    memory_path: str
    session_number: int
    total_turns: int
    facts_count: int
    top_k_context: list[dict] = field(default_factory=list)

    def to_system_message(self) -> str:
        """Format as a system message for the new session."""
        parts = [
            f"[SESSION HANDOFF — continuing from session {self.session_number}, {self.total_turns} total turns]",
            "",
            "PREVIOUS SESSION SUMMARY:",
            self.summary,
        ]

        if self.top_k_context:
            parts.append("")
            parts.append("KEY RETRIEVED FACTS:")
            for ctx in self.top_k_context:
                parts.append(f"  - (turn {ctx.get('turn_num', '?')}): {ctx.get('chunk_text', '')[:200]}")

        parts.append("")
        parts.append(f"[Memory index loaded: {self.facts_count} chunks from {self.memory_path}]")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "memory_path": self.memory_path,
            "session_number": self.session_number,
            "total_turns": self.total_turns,
            "facts_count": self.facts_count,
            "top_k_context": self.top_k_context,
        }


class SessionChain:
    """Orchestrates auto-spawn conversation chaining with memory handoff.

    Usage:
        chain = SessionChain(max_tokens=32768)
        chain.start_session()

        for user_msg, assistant_msg in conversation:
            status = chain.add_exchange(user_msg, assistant_msg)
            if status["should_spawn"]:
                handoff = chain.spawn_new_session()
                # handoff.to_system_message() -> inject into new session
    """

    def __init__(
        self,
        max_tokens: int = 32768,
        reserve_tokens: int = 2048,
        spawn_threshold: float = 0.85,
        memory_dir: str = "records/chain_memory",
        summarizer_mode: str = "ollama",
        summarizer_model: str = "qwen2.5:1.5b",
        ollama_url: str = "http://localhost:11434",
        top_k_handoff: int = 5,
        fact_verifier: Optional[Any] = None,
        use_embeddings: bool = False,
        confidence_threshold: float = 0.3,
    ):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.top_k_handoff = top_k_handoff
        self.fact_verifier = fact_verifier
        self.use_embeddings = use_embeddings
        self.confidence_threshold = confidence_threshold

        self.monitor = TokenBudgetMonitor(
            max_tokens=max_tokens,
            reserve_tokens=reserve_tokens,
            spawn_threshold=spawn_threshold,
        )

        self.summarizer = SessionSummarizer(
            mode=summarizer_mode,
            model=summarizer_model,
            ollama_url=ollama_url,
        )

        # State
        self._session_number: int = 0
        self._total_turns: int = 0
        self._messages: list[dict] = []
        self._memory: Optional[Any] = None  # ConversationMemory instance
        self._chain_history: list[dict] = []

        logger.info(
            "[SessionChain] initialized | max_tokens=%d | threshold=%.0f%% | memory_dir=%s",
            max_tokens, spawn_threshold * 100, self.memory_dir,
        )

    def start_session(self, handoff: Optional[HandoffPayload] = None) -> dict:
        """Start a new session, optionally loading from a handoff.

        Returns:
            dict with session_number, loaded_from, system_message (if handoff).
        """
        from .conversation_memory import ConversationMemory

        self._session_number += 1
        self.monitor.reset()
        self._messages = []

        result = {
            "session_number": self._session_number,
            "loaded_from": None,
            "system_message": None,
            "memory_chunks": 0,
        }

        if handoff and handoff.memory_path and os.path.exists(handoff.memory_path):
            # Load persisted memory from previous session
            self._memory = ConversationMemory.load_from_disk(
                handoff.memory_path,
                fact_verifier=self.fact_verifier,
            )
            result["loaded_from"] = handoff.memory_path
            result["memory_chunks"] = len(self._memory._chunks)
            result["system_message"] = handoff.to_system_message()

            # Count handoff system message tokens
            self.monitor.add_turn(result["system_message"], role="system")

            logger.info(
                "[SessionChain] session %d started with handoff | %d chunks loaded | %d prior turns",
                self._session_number, result["memory_chunks"], handoff.total_turns,
            )
        else:
            # Fresh session
            self._memory = ConversationMemory(
                fact_verifier=self.fact_verifier,
                use_embeddings=self.use_embeddings,
                confidence_threshold=self.confidence_threshold,
            )
            logger.info("[SessionChain] session %d started fresh", self._session_number)

        return result

    def add_exchange(self, user_content: str, assistant_content: str) -> dict:
        """Record a user+assistant exchange and check spawn status.

        Returns:
            dict with turn, user_tokens, assistant_tokens, total_tokens,
            usage_pct, should_spawn, tokens_remaining.
        """
        self._total_turns += 1
        turn_num = self._total_turns

        # Track messages
        self._messages.append({"role": "user", "content": user_content})
        self._messages.append({"role": "assistant", "content": assistant_content})

        # Ingest into memory
        if self._memory:
            self._memory.ingest_turn("user", user_content, turn_num)
            self._memory.ingest_turn("assistant", assistant_content, turn_num)

        # Update token monitor
        user_status = self.monitor.add_turn(user_content, "user")
        asst_status = self.monitor.add_turn(assistant_content, "assistant")

        return {
            "turn": turn_num,
            "session": self._session_number,
            "user_tokens": user_status["tokens_added"],
            "assistant_tokens": asst_status["tokens_added"],
            "current_tokens": self.monitor.current_tokens,
            "usage_pct": asst_status["usage_pct"],
            "should_spawn": self.monitor.should_spawn,
            "tokens_remaining": self.monitor.tokens_remaining,
        }

    def spawn_new_session(self, recent_query: Optional[str] = None) -> HandoffPayload:
        """Trigger session spawn: summarize, persist memory, build handoff.

        Args:
            recent_query: Optional query to retrieve top-k context for handoff.

        Returns:
            HandoffPayload ready to inject into new session.
        """
        logger.info(
            "[SessionChain] SPAWNING session %d -> %d | %d total turns | %.1f%% tokens used",
            self._session_number, self._session_number + 1,
            self._total_turns, self.monitor.usage_fraction * 100,
        )

        # 1. Summarize current session
        summary_result = self.summarizer.summarize(self._messages)
        summary_text = summary_result["summary"]

        # 2. Persist memory to disk
        memory_path = self.memory_dir / f"session_{self._session_number:04d}.json"
        save_result = {}
        if self._memory:
            save_result = self._memory.save_to_disk(memory_path)

        # 3. Retrieve top-k context for handoff (most relevant to last topic)
        top_k_context = []
        if self._memory and recent_query:
            top_k_context = self._memory.retrieve(
                recent_query, top_k=self.top_k_handoff, defense_gate=True,
            )
        elif self._memory and self._messages:
            # Use last user message as query
            last_user = ""
            for m in reversed(self._messages):
                if m["role"] == "user":
                    last_user = m["content"]
                    break
            if last_user:
                top_k_context = self._memory.retrieve(
                    last_user, top_k=self.top_k_handoff, defense_gate=True,
                )

        # 4. Build handoff payload
        handoff = HandoffPayload(
            summary=summary_text,
            memory_path=str(memory_path),
            session_number=self._session_number,
            total_turns=self._total_turns,
            facts_count=save_result.get("chunks_saved", 0),
            top_k_context=top_k_context,
        )

        # 5. Record chain history
        self._chain_history.append({
            "session": self._session_number,
            "turns": self._total_turns,
            "tokens_used": self.monitor.current_tokens,
            "memory_path": str(memory_path),
            "summary_tokens": summary_result["tokens_est"],
            "summary_mode": summary_result["mode"],
            "elapsed_s": summary_result["elapsed_s"],
        })

        logger.info(
            "[SessionChain] handoff ready | summary=%d tokens | memory=%d chunks | top_k=%d",
            summary_result["tokens_est"], save_result.get("chunks_saved", 0), len(top_k_context),
        )

        return handoff

    def get_chain_history(self) -> list[dict]:
        """Return the full chain history across sessions."""
        return list(self._chain_history)

    def get_status(self) -> dict:
        """Return current chain status."""
        memory_stats = self._memory.get_stats() if self._memory else {}
        return {
            "session_number": self._session_number,
            "total_turns": self._total_turns,
            "monitor": self.monitor.to_dict(),
            "memory": memory_stats,
            "chain_sessions": len(self._chain_history),
        }
