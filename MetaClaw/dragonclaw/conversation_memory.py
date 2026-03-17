"""Conversation-scoped memory with tiered retrieval.

Indexes every turn in a conversation as embeddable chunks, enabling
semantic search across the full thread with recency weighting and
defense-gated verification via FactVerifier.

Architecture:
  Tier 1 — Immediate context window (raw attention, 0ms)
  Tier 2 — Conversation memory index (embedding search, ~50-200ms)
  Tier 3 — Defense-gated retrieval (FactVerifier on results)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Chunk dataclass                                                      #
# ------------------------------------------------------------------ #

@dataclass
class MemoryChunk:
    """A single indexed chunk from a conversation turn."""
    text: str
    turn_num: int
    role: str
    chunk_idx: int
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "chunk_text": self.text,
            "turn_num": self.turn_num,
            "role": self.role,
            "chunk_idx": self.chunk_idx,
        }


# ------------------------------------------------------------------ #
# Sentence splitter                                                    #
# ------------------------------------------------------------------ #

_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "of", "in", "to", "for", "with", "on", "at", "by", "from",
    "and", "or", "not", "it", "this", "that", "what", "which",
    "i", "you", "we", "they", "he", "she", "my", "your",
    "do", "does", "did", "has", "have", "had", "will", "would",
    "can", "could", "should", "just", "about", "me", "its",
    "answer", "tell", "remember", "know", "please",
})

def _significant_words(text: str) -> set[str]:
    """Extract significant (non-stopword) lowercase words."""
    words = set(re.findall(r'[a-z0-9]+', text.lower()))
    sig = words - _STOPWORDS
    return sig if sig else words  # fall back to all words if all are stopwords

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Falls back to full text if no splits."""
    sentences = _SENT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _chunk_text(text: str, max_sentences: int = 3) -> list[str]:
    """Split a turn's text into chunks of up to *max_sentences* sentences."""
    sentences = _split_sentences(text)
    if not sentences:
        return [text.strip()] if text.strip() else []
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        if chunk:
            chunks.append(chunk)
    return chunks


# ------------------------------------------------------------------ #
# ConversationMemory                                                   #
# ------------------------------------------------------------------ #

class ConversationMemory:
    """Conversation-scoped memory with tiered retrieval.

    Indexes every turn in the conversation as embeddable chunks,
    enabling semantic search across the full thread with recency
    weighting and defense-gated verification.
    """

    def __init__(
        self,
        embedding_model_path: str = "Qwen/Qwen3-Embedding-0.6B",
        fact_verifier: Optional[Any] = None,
        recency_base: float = 0.1,
        recency_bonus: float = 0.9,
        confidence_threshold: float = 0.6,
        chunk_max_sentences: int = 3,
        use_embeddings: bool = True,
    ):
        """Initialize conversation memory.

        Args:
            embedding_model_path: SentenceTransformer model for embeddings.
            fact_verifier: Optional FactVerifier for Tier 3 gating.
            recency_base: Minimum weight for oldest turns.
            recency_bonus: Additional weight scaled by recency.
            confidence_threshold: Cosine sim threshold for "found".
            chunk_max_sentences: Max sentences per chunk.
            use_embeddings: If False, fall back to keyword matching (simulated mode).
        """
        self.embedding_model_path = embedding_model_path
        self.fact_verifier = fact_verifier
        self.recency_base = recency_base
        self.recency_bonus = recency_bonus
        self.confidence_threshold = confidence_threshold
        self.chunk_max_sentences = chunk_max_sentences
        self.use_embeddings = use_embeddings

        # Storage
        self._chunks: list[MemoryChunk] = []
        self._max_turn: int = 0

        # Embedding model (lazy-loaded)
        self._embedding_model = None

        # Stats
        self._stats = {
            "total_turns": 0,
            "total_chunks": 0,
            "retrievals": 0,
            "tier1_hits": 0,
            "tier2_hits": 0,
            "tier3_blocks": 0,
        }

        logger.info(
            "[ConversationMemory] initialized | embeddings=%s | recency=%.1f+%.1f | threshold=%.2f",
            use_embeddings, recency_base, recency_bonus, confidence_threshold,
        )

    # ------------------------------------------------------------------ #
    # Embedding model                                                      #
    # ------------------------------------------------------------------ #

    def _get_embedding_model(self):
        """Lazy-load the SentenceTransformer model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                logger.warning("[ConversationMemory] sentence-transformers not available, using keyword fallback")
                self.use_embeddings = False
                return None
            logger.info("[ConversationMemory] loading embedding model: %s", self.embedding_model_path)
            self._embedding_model = SentenceTransformer(self.embedding_model_path)
        return self._embedding_model

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns (N, D) array."""
        model = self._get_embedding_model()
        if model is None:
            return np.zeros((len(texts), 1))
        return model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    # ------------------------------------------------------------------ #
    # Ingest                                                               #
    # ------------------------------------------------------------------ #

    def ingest_turn(self, role: str, content: str, turn_num: int) -> int:
        """Add a conversation turn to the memory index.

        Returns the number of chunks created from this turn.
        """
        chunks_text = _chunk_text(content, self.chunk_max_sentences)
        if not chunks_text:
            return 0

        # Create chunk objects
        new_chunks = []
        for i, text in enumerate(chunks_text):
            chunk = MemoryChunk(
                text=text,
                turn_num=turn_num,
                role=role,
                chunk_idx=i,
            )
            new_chunks.append(chunk)

        # Embed if using embeddings
        if self.use_embeddings:
            embeddings = self._embed([c.text for c in new_chunks])
            for chunk, emb in zip(new_chunks, embeddings):
                chunk.embedding = emb

        self._chunks.extend(new_chunks)
        self._max_turn = max(self._max_turn, turn_num)
        self._stats["total_turns"] += 1
        self._stats["total_chunks"] += len(new_chunks)

        logger.debug(
            "[ConversationMemory] ingested turn %d (%s): %d chunk(s)",
            turn_num, role, len(new_chunks),
        )
        return len(new_chunks)

    # ------------------------------------------------------------------ #
    # Recency weighting                                                    #
    # ------------------------------------------------------------------ #

    def _recency_weight(self, turn_num: int) -> float:
        """Compute recency weight for a given turn number.

        weight = base + bonus * (turn_num / max_turn)
        Turn 1 of 100 → ~0.1, Turn 100 of 100 → ~1.0
        """
        if self._max_turn <= 0:
            return 1.0
        ratio = turn_num / self._max_turn
        return self.recency_base + self.recency_bonus * ratio

    # ------------------------------------------------------------------ #
    # Tier 1: Immediate context search (keyword)                           #
    # ------------------------------------------------------------------ #

    def _tier1_search(self, query: str, recent_turns: int = 10, top_k: int = 3) -> list[dict]:
        """Search recent turns using significant-word overlap."""
        query_sig = _significant_words(query)
        if not query_sig:
            return []

        # Get chunks from the most recent N turns
        cutoff = max(0, self._max_turn - recent_turns)
        recent_chunks = [c for c in self._chunks if c.turn_num > cutoff]

        scored = []
        for chunk in recent_chunks:
            chunk_sig = _significant_words(chunk.text)
            overlap = len(query_sig & chunk_sig)
            if overlap > 0:
                score = overlap / len(query_sig)
                scored.append((score, chunk))

        scored.sort(key=lambda x: -x[0])
        results = []
        for score, chunk in scored[:top_k]:
            r = chunk.to_dict()
            r["score"] = round(score, 4)
            r["tier"] = 1
            r["verified"] = None  # not yet verified
            results.append(r)
        return results

    # ------------------------------------------------------------------ #
    # Tier 2: Full-thread embedding search with recency weighting          #
    # ------------------------------------------------------------------ #

    def _tier2_search(self, query: str, top_k: int = 3) -> list[dict]:
        """Search entire conversation index using embeddings + recency weight."""
        if not self._chunks:
            return []

        if self.use_embeddings:
            query_emb = self._embed([query])[0]
            scores = []
            for chunk in self._chunks:
                if chunk.embedding is not None:
                    cosine = float(np.dot(query_emb, chunk.embedding))
                else:
                    cosine = 0.0
                weight = self._recency_weight(chunk.turn_num)
                scores.append(cosine * weight)
        else:
            # Keyword fallback (simulated mode)
            query_sig = _significant_words(query)
            scores = []
            for chunk in self._chunks:
                chunk_sig = _significant_words(chunk.text)
                overlap = len(query_sig & chunk_sig) / max(len(query_sig), 1)
                weight = self._recency_weight(chunk.turn_num)
                scores.append(overlap * weight)

        # Sort by score descending
        indexed = sorted(enumerate(scores), key=lambda x: -x[1])
        results = []
        for idx, score in indexed[:top_k]:
            chunk = self._chunks[idx]
            r = chunk.to_dict()
            r["score"] = round(score, 4)
            r["tier"] = 2
            r["verified"] = None
            results.append(r)
        return results

    # ------------------------------------------------------------------ #
    # Tier 3: Defense-gated verification                                   #
    # ------------------------------------------------------------------ #

    def _tier3_verify(self, results: list[dict]) -> list[dict]:
        """Run FactVerifier on retrieved chunks. Mark verified/blocked."""
        if self.fact_verifier is None:
            for r in results:
                r["verified"] = True  # no verifier = pass-through
            return results

        verified = []
        for r in results:
            conversation = [{"role": r.get("role", "assistant"), "content": r["chunk_text"]}]
            is_safe, violations = self.fact_verifier.verify_conversation(conversation)
            r["verified"] = is_safe
            if is_safe:
                verified.append(r)
            else:
                self._stats["tier3_blocks"] += 1
                logger.info(
                    "[ConversationMemory] Tier 3 BLOCKED chunk from turn %d: %s",
                    r["turn_num"], r["chunk_text"][:60],
                )
        return verified

    # ------------------------------------------------------------------ #
    # Confidence scoring                                                   #
    # ------------------------------------------------------------------ #

    def get_confidence(self, query: str, context_turns: Optional[list[dict]] = None) -> float:
        """Score whether the immediate context likely contains the answer.

        Returns 0.0-1.0 confidence score.
        Used to decide whether to escalate from Tier 1 to Tier 2.
        """
        # Strategy: check Tier 1 results — if best score is above threshold, confident
        tier1 = self._tier1_search(query, recent_turns=10, top_k=1)
        if tier1:
            return tier1[0]["score"]
        return 0.0

    # ------------------------------------------------------------------ #
    # Public retrieve                                                      #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        tier: str = "auto",
        defense_gate: bool = True,
    ) -> list[dict]:
        """Tiered retrieval.

        Args:
            query: The search query.
            top_k: Number of results to return.
            tier: "auto" (escalate as needed), "immediate" (Tier 1 only),
                  "full" (Tier 2 only).
            defense_gate: If True, run Tier 3 verification on results.

        Returns:
            List of {chunk_text, turn_num, role, score, tier, verified}.
        """
        self._stats["retrievals"] += 1
        t0 = time.time()

        results = []

        if tier in ("auto", "immediate"):
            results = self._tier1_search(query, top_k=top_k)
            if results and results[0]["score"] >= self.confidence_threshold:
                self._stats["tier1_hits"] += 1
                logger.debug("[ConversationMemory] Tier 1 hit (score=%.3f)", results[0]["score"])
                if defense_gate:
                    results = self._tier3_verify(results)
                elapsed = time.time() - t0
                for r in results:
                    r["elapsed_ms"] = round(elapsed * 1000, 1)
                return results

        # Escalate to Tier 2
        if tier in ("auto", "full"):
            results = self._tier2_search(query, top_k=top_k)
            if results:
                self._stats["tier2_hits"] += 1
                logger.debug("[ConversationMemory] Tier 2 hit (score=%.3f)", results[0]["score"])

        # Tier 3 defense gate
        if defense_gate and results:
            results = self._tier3_verify(results)

        elapsed = time.time() - t0
        for r in results:
            r["elapsed_ms"] = round(elapsed * 1000, 1)

        return results

    # ------------------------------------------------------------------ #
    # Stats                                                                #
    # ------------------------------------------------------------------ #

    def get_stats(self) -> dict:
        """Return memory statistics."""
        return {
            **self._stats,
            "max_turn": self._max_turn,
            "avg_chunks_per_turn": (
                round(self._stats["total_chunks"] / self._stats["total_turns"], 2)
                if self._stats["total_turns"] > 0 else 0
            ),
        }

    def reset(self):
        """Clear all stored chunks and reset stats."""
        self._chunks.clear()
        self._max_turn = 0
        self._stats = {k: 0 for k in self._stats}
        logger.info("[ConversationMemory] reset")

    # ------------------------------------------------------------------ #
    # Disk persistence                                                     #
    # ------------------------------------------------------------------ #

    def save_to_disk(self, path: str | Path) -> dict:
        """Persist memory index to disk as JSON.

        Saves all chunks (text + metadata), stats, and config.
        Embeddings are saved as lists for JSON serialization.

        Returns:
            dict with keys: path, chunks_saved, turns, size_bytes.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        chunks_data = []
        for c in self._chunks:
            cd = c.to_dict()
            if c.embedding is not None:
                cd["embedding"] = c.embedding.tolist()
            chunks_data.append(cd)

        payload = {
            "version": 1,
            "config": {
                "embedding_model_path": self.embedding_model_path,
                "recency_base": self.recency_base,
                "recency_bonus": self.recency_bonus,
                "confidence_threshold": self.confidence_threshold,
                "chunk_max_sentences": self.chunk_max_sentences,
                "use_embeddings": self.use_embeddings,
            },
            "stats": dict(self._stats),
            "max_turn": self._max_turn,
            "chunks": chunks_data,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        size = path.stat().st_size
        logger.info(
            "[ConversationMemory] saved to %s | %d chunks | %d turns | %.1f KB",
            path, len(chunks_data), self._stats["total_turns"], size / 1024,
        )
        return {
            "path": str(path),
            "chunks_saved": len(chunks_data),
            "turns": self._stats["total_turns"],
            "size_bytes": size,
        }

    @classmethod
    def load_from_disk(
        cls,
        path: str | Path,
        fact_verifier: Optional[Any] = None,
    ) -> "ConversationMemory":
        """Load a persisted memory index from disk.

        Reconstructs ConversationMemory with all chunks, stats,
        and optionally re-attaches embeddings.

        Args:
            path: Path to the saved JSON file.
            fact_verifier: Optional FactVerifier for Tier 3 gating.

        Returns:
            A new ConversationMemory instance with restored state.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        cfg = payload.get("config", {})
        cm = cls(
            embedding_model_path=cfg.get("embedding_model_path", "Qwen/Qwen3-Embedding-0.6B"),
            fact_verifier=fact_verifier,
            recency_base=cfg.get("recency_base", 0.1),
            recency_bonus=cfg.get("recency_bonus", 0.9),
            confidence_threshold=cfg.get("confidence_threshold", 0.6),
            chunk_max_sentences=cfg.get("chunk_max_sentences", 3),
            use_embeddings=cfg.get("use_embeddings", True),
        )

        # Restore chunks
        for cd in payload.get("chunks", []):
            emb = None
            if "embedding" in cd:
                emb = np.array(cd["embedding"], dtype=np.float32)
            chunk = MemoryChunk(
                text=cd["chunk_text"],
                turn_num=cd["turn_num"],
                role=cd["role"],
                chunk_idx=cd["chunk_idx"],
                embedding=emb,
            )
            cm._chunks.append(chunk)

        cm._max_turn = payload.get("max_turn", 0)
        cm._stats = payload.get("stats", {k: 0 for k in cm._stats})

        logger.info(
            "[ConversationMemory] loaded from %s | %d chunks | %d turns",
            path, len(cm._chunks), cm._stats.get("total_turns", 0),
        )
        return cm
