# Conversation-Scoped Memory Retrieval Architecture

**Project:** MetaClaw Cascading Hallucination Evaluation  
**Date:** March 2026  
**Version:** 1.0 — Design Phase  
**Suites:** O66-O70 (Conversation Memory Retrieval Tests)

---

## 1. Problem Statement

In self-improving AI pipelines, models suffer **catastrophic recall collapse** — 0% fact retrieval after 60 turns of context dilution (proven by O36). When the model can't recall a previously stated fact, it hallucinates a replacement, and the RL training loop reinforces that hallucination (proven by O61: poison advantage +0.949 vs clean -0.949).

**The core question:** Can we give the model the ability to "check its notes" — search its own conversation thread — before resorting to hallucination?

---

## 2. Tiered Retrieval Architecture

### 2.1 Overview

```
Query arrives
    |
    v
+---------------------------------------+
|  TIER 1: Immediate Context Window     |
|  Last N turns (raw attention)         |
|  Latency: 0ms (already in prompt)     |
|  Limit: ~20K tokens (max_context)     |
|  Accuracy: high recent, 0% beyond     |
+------------------+--------------------+
                   | confidence < threshold?
                   v
+---------------------------------------+
|  TIER 2: Conversation Memory Index    |
|  Full thread -> chunked + embedded    |
|  Recency-weighted cosine similarity   |
|  Latency: ~50-200ms (local embed)     |
|  Uses: Qwen3-Embedding-0.6B          |
+------------------+--------------------+
                   | still not found?
                   v
+---------------------------------------+
|  TIER 3: Defense-Gated Retrieval      |
|  Retrieved facts verified by          |
|  FactVerifier before injection        |
|  Prevents retrieving poisoned turns   |
+---------------------------------------+
```

### 2.2 Tier 1: Immediate Context Window

- **What it is:** The standard LLM attention window — whatever turns fit in `max_context_tokens` (currently 20K).
- **Strengths:** Zero latency, no extra compute, works for recent turns.
- **Weakness:** Catastrophic recall collapse beyond ~40-60 turns. Model cannot navigate its own context effectively at scale.
- **Proven by:** O36 (0% recall after 60 turns), O26-O30 (500-turn stress tests).

### 2.3 Tier 2: Conversation Memory Index

- **What it is:** A semantic index of the full conversation thread, searchable by embedding similarity.
- **How it works:**
  1. Every turn is chunked (1-3 sentences per chunk)
  2. Each chunk is embedded using the existing `Qwen3-Embedding-0.6B` model (already in SkillManager)
  3. Chunks are stored with metadata: `{turn_num, role, timestamp, chunk_text, embedding}`
  4. On query, compute cosine similarity with **recency weighting**
- **Recency weighting formula:**

```
score(chunk) = cosine_sim(query, chunk) * weight(turn)
weight(turn) = base_weight + recency_bonus * (turn_num / total_turns)
```

- Turn 1 of 100: weight ≈ 0.1 (findable but deprioritized)
- Turn 100 of 100: weight ≈ 1.0 (full priority)

- **Escalation trigger:** Tier 2 is invoked when:
  - Model output contains hedging language ("I'm not sure", "I believe", "let me think")
  - FactVerifier cannot verify a claim from immediate context
  - Explicit retrieval request detected in the prompt

### 2.4 Tier 3: Defense-Gated Retrieval

- **What it is:** A safety layer that runs FactVerifier on retrieved chunks before injecting them back into context.
- **Why it's needed:** Without this, the model might retrieve its own earlier hallucination and treat it as ground truth — creating a **self-reinforcing hallucination loop**.
- **How it works:**
  1. Tier 2 returns top-K candidate chunks
  2. Each chunk is passed through FactVerifier
  3. Chunks with violations are flagged or removed
  4. Only verified chunks are injected into the prompt
- **This closes the loop:** The same defense stack that protects the training loop (O61-O65) also protects the retrieval loop.

---

## 3. Existing Infrastructure

### 3.1 Components We Already Have

| Component | Location | Status |
|-----------|----------|--------|
| Embedding model | `SkillManager` (Qwen3-Embedding-0.6B) | Ready |
| Cosine similarity search | `SkillManager._embedding_retrieve()` | Ready |
| Session isolation | `SkillManager.filter_by_session()` | Ready |
| FactVerifier | `metaclaw/defense.py` | Ready |
| InputSanitizer | `metaclaw/defense.py` | Ready |
| OutputFilter | `metaclaw/defense.py` | Ready |
| DefenseStack | `metaclaw/defense.py` | Ready |
| Context window config | `config.py` (max_context_tokens=20K) | Ready |

### 3.2 Components To Build

| Component | Description | Priority |
|-----------|-------------|----------|
| ConversationMemory class | Wraps chunking + embedding + retrieval | P0 |
| Conversation chunker | Splits turns into embeddable chunks | P0 |
| Recency weighting | Ascending weight formula for similarity | P0 |
| Confidence thresholding | Decides Tier 1 -> Tier 2 escalation | P1 |
| Defense-gated retrieval | FactVerifier on retrieved chunks | P1 |
| Escalation timeout | Configurable ms timeout per tier | P2 |

---

## 4. ConversationMemory Class Design

### 4.1 API Surface

```python
class ConversationMemory:
    """Conversation-scoped memory with tiered retrieval.
    
    Indexes every turn in the conversation as embeddable chunks,
    enabling semantic search across the full thread with recency
    weighting and defense-gated verification.
    """
    
    def __init__(
        self,
        embedding_model_path: str = "Qwen/Qwen3-Embedding-0.6B",
        fact_verifier: Optional[FactVerifier] = None,
        recency_base: float = 0.1,
        recency_bonus: float = 0.9,
        confidence_threshold: float = 0.6,
        chunk_max_sentences: int = 3,
    ):
        """Initialize conversation memory.
        
        Args:
            embedding_model_path: SentenceTransformer model for embeddings
            fact_verifier: Optional FactVerifier for Tier 3 gating
            recency_base: Minimum weight for oldest turns
            recency_bonus: Additional weight scaled by recency
            confidence_threshold: Cosine sim threshold for "found"
            chunk_max_sentences: Max sentences per chunk
        """
    
    def ingest_turn(self, role: str, content: str, turn_num: int) -> int:
        """Add a conversation turn to the memory index.
        
        Returns number of chunks created from this turn.
        """
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 3, 
        tier: str = "auto",
        defense_gate: bool = True,
    ) -> list[dict]:
        """Tiered retrieval.
        
        tier="auto": Try Tier 1 (immediate), escalate to Tier 2 if needed
        tier="immediate": Only search recent turns
        tier="full": Search entire conversation index
        
        Returns list of {chunk_text, turn_num, role, score, tier, verified}
        """
    
    def get_confidence(self, query: str, context_turns: list[dict]) -> float:
        """Score whether the context window likely contains the answer.
        
        Returns 0.0-1.0 confidence score.
        Used to decide whether to escalate from Tier 1 to Tier 2.
        """
    
    def get_stats(self) -> dict:
        """Return memory statistics.
        
        Returns {total_turns, total_chunks, avg_chunks_per_turn,
                 retrievals, tier1_hits, tier2_hits, tier3_blocks}
        """
```

### 4.2 Data Flow

```
User says: "What was the capital we discussed earlier?"

1. Tier 1 Check:
   - Scan last 10 turns for "capital" mention
   - get_confidence() returns 0.3 (below 0.6 threshold)
   - ESCALATE to Tier 2

2. Tier 2 Search:
   - Embed query: "capital we discussed earlier"
   - Cosine similarity against all conversation chunks
   - Apply recency weighting
   - Top result: Turn 5, chunk "The capital of France is Paris"
   - Score: 0.87 (above threshold)

3. Tier 3 Gate:
   - FactVerifier checks: "capital of France" -> "Paris" ✓
   - Chunk passes verification
   - INJECT into context: "[Retrieved from turn 5: The capital of France is Paris]"

4. Model responds with verified fact instead of hallucinating
```

---

## 5. O66-O70 Test Suite Design

### O66: Immediate Recall Window Mapping

**Purpose:** Establish the Tier 1 baseline — at what turn distance does raw context recall fail?

- Teach a fact at turn T
- Insert N distractor turns
- Ask for recall at turn T+N
- Map the decay curve: N=5, 10, 20, 40, 60, 80, 100
- Expected: near-100% at N=5, ~0% at N=60+ (consistent with O36)

### O67: Recency-Weighted Retrieval

**Purpose:** Can Tier 2 recover facts that Tier 1 loses?

- Teach 5 facts at turns 5, 15, 25, 35, 45
- Dilute with 50 distractor turns (total: 100 turns)
- Query each fact using ConversationMemory.retrieve()
- Compare: raw context recall vs. Tier 2 retrieval
- Expected: raw context ~20%, Tier 2 >80%

### O68: Confidence Threshold Calibration

**Purpose:** When should the system escalate from Tier 1 to Tier 2?

- Run 20 queries: 10 answerable from immediate context, 10 requiring full-thread search
- Sweep confidence thresholds (0.3 to 0.9 in 0.1 steps)
- Measure: true positive rate, false positive rate, unnecessary escalations
- Find optimal threshold that minimizes latency while maximizing recall

### O69: Defense-Gated Retrieval

**Purpose:** Does Tier 3 prevent retrieval of poisoned conversation turns?

- Inject 5 poisoned turns into a 50-turn conversation
- Also inject 5 truthful turns about the same topics
- Query the poisoned topics via ConversationMemory
- With defense gate ON: should retrieve truthful turns, block poisoned ones
- With defense gate OFF: may retrieve poisoned turns
- Measure: poison retrieval rate (should be 0% gated, >0% ungated)

### O70: End-to-End Tiered Pipeline

**Purpose:** Full system test under combined adversarial + dilution pressure.

- 100-turn conversation with:
  - 10 taught facts (spread across turns 1-50)
  - 40 distractor turns
  - 5 poison injections
  - 5 retrieval queries at turns 60, 70, 80, 90, 100
- Measure per query:
  - Which tier answered (1, 2, or 3)?
  - Was the answer correct?
  - Was any poisoned content retrieved?
  - Latency per tier
- Pass criteria: >80% correct retrieval, 0% poison leakage

---

## 6. Relationship to Whitepaper Thesis

This architecture directly addresses the **root cause** of cascading hallucinations:

1. **O36 proved:** Models lose recall after 60 turns → they hallucinate replacements
2. **O61 proved:** Hallucinated replacements get positive RL advantages → training loop reinforces them
3. **ConversationMemory solves:** Give the model access to its full conversation → recall doesn't collapse → no hallucination trigger → training loop stays clean

The defense-gated retrieval (Tier 3) ensures that even if earlier turns contained hallucinations, they don't get recycled back into the response.

**Key insight:** The same defense stack that protects the training loop also protects the retrieval loop. This unifies the defensive architecture across both inference-time and training-time failure modes.

---

## 7. Implementation Dependencies

- `sentence-transformers` (already in requirements for SkillManager)
- `numpy` (already in requirements)
- `metaclaw/defense.py` — FactVerifier, DefenseStack
- `metaclaw/skill_manager.py` — embedding model loading pattern
- `metaclaw/config.py` — configuration pattern

No new external dependencies required.
