# DragonClaw — DALL-E Diagram Prompts

**Purpose:** Generate 4 data flow diagrams for the DragonClaw website breakdown page (modeled after ERLA format)

---

## Image 1: Three-Tier Defense-Gated Retrieval Pipeline

**Use in page:** Section 2 — Architecture, after the tiered retrieval explanation

**DALL-E Prompt:**
```
A vertical data flow diagram on a dark background (#0a0a1a) showing how a user query flows through a three-tier memory retrieval pipeline with a defense gate.

Top: A glowing blue chat bubble labeled "User Query" with an arrow pointing down.

Tier 1 (green, leftmost path): A rounded box labeled "Keyword Match" with a speed icon (~1ms). Small label: "Cheap, instant". Arrow flows down to a decision diamond: "Match found?" — Yes arrow exits right to "Inject into prompt". No arrow continues down.

Tier 2 (blue, middle path): A rounded box labeled "Embedding Search" with a vector icon (~50ms). Small label: "Semantic, accurate". Arrow flows down to another decision diamond. Yes arrow goes right. No arrow continues down to "No memory found".

Tier 3 (purple, rightmost path — the defense gate): A larger, glowing box with a shield icon labeled "FactVerifier Gate". Inside: "Check vs Truth Store", "Score Confidence", "Block Poison". Two output arrows: green arrow labeled "VERIFIED" goes to "Inject into prompt", red arrow labeled "BLOCKED" goes to a red X with "Poison rejected".

Bottom: A bright box labeled "LLM Generation" receiving only verified facts.

Style: Clean infographic, sans-serif labels, neon glow on tier boxes against dark background. Purple/blue/green color scheme. 960x960 pixels (square). Professional, minimal, no 3D effects.
```

**Placement:** After "2.1 Tiered Retrieval Architecture" heading

---

## Image 2: Auto-Spawn Session Chain Architecture

**Use in page:** Section 2 — Architecture, after the session chaining explanation

**DALL-E Prompt:**
```
A horizontal data flow diagram on a dark background (#0a0a1a) showing three connected conversation sessions forming an infinite chain.

Left: Session 1 — a tall rounded rectangle containing stacked chat bubbles (user/assistant alternating). A progress bar at the bottom labeled "Token Usage: 80%" glows yellow/orange. An arrow exits right labeled "SPAWN TRIGGER".

Middle pipeline (between sessions): Three stacked process boxes connected by arrows flowing downward then right:
1. Orange box: "Summarize" with a document icon
2. Blue box: "Save Memory to Disk" with a hard drive icon  
3. Green box: "Create Handoff Payload" with a package icon
An arrow exits right labeled "summary + memory index"

Center: Session 2 — another tall rounded rectangle. At top, a green badge: "Loaded: 10 memory chunks". Chat bubbles continue. Progress bar at 80% again. Another SPAWN TRIGGER arrow exits right.

Right: Session 3 — same pattern. At top: "Loaded: 20 memory chunks". A query bubble says "What's the capital?" and a retrieval arrow reaches ALL the way back to Session 1's memory with a green checkmark labeled "100% Recall".

Bottom: A timeline arrow spanning all three sessions labeled "Unlimited Conversation Length — Zero Token Limit"

Style: Dark theme, neon accents (green for memory, orange for summarize, blue for persist, purple for sessions). Clean, minimal, professional. 960x960 pixels (square).
```

**Placement:** After "2.2 Auto-Spawn Session Chaining" heading

---

## Image 3: Poison Attack vs Defense-Gated Memory (Split Comparison)

**Use in page:** Section 3 — Why Defense Gating Matters

**DALL-E Prompt:**
```
A split-screen comparison diagram on a dark background (#0a0a1a). Left side labeled "Standard RAG" (red-tinted), right side labeled "DragonClaw" (green-tinted). A vertical divider separates them.

LEFT SIDE (Standard RAG — vulnerable):
Top: An attacker icon (hooded figure silhouette) injects a red document labeled "Poisoned Fact" into a vector database (cylinder icon).
Middle: A user query arrow goes to the vector database. The poisoned document is retrieved (red arrow labeled "Top-K match").
Bottom: The poisoned fact flows directly into the LLM (brain icon). Output shows a red chat bubble: "FALSE ANSWER" with a danger icon. 
Label at bottom: "90% attack success rate (PoisonedRAG, 2024)"

RIGHT SIDE (DragonClaw — defended):
Top: Same attacker injects same poisoned fact into vector database.
Middle: Same user query retrieves the poisoned document.
Key difference: Between retrieval and LLM, there is a glowing purple SHIELD labeled "FactVerifier Gate". The shield has two outputs:
- Red arrow going to trash: "POISON BLOCKED"
- Green arrow going to LLM: "Only verified facts"
Bottom: LLM outputs a green chat bubble: "CORRECT ANSWER" with a checkmark.
Label at bottom: "Defense-gated retrieval — zero-trust for memory"

Style: Clean infographic, dark background, red vs green color contrast. Professional, sans-serif labels. 960x960 pixels (square).
```

**Placement:** After "3. Why Defense Gating Matters" heading, before the PoisonedRAG discussion

---

## Image 4: Competitive Gap Matrix Visualization

**Use in page:** Section 5 — Competitive Landscape

**DALL-E Prompt:**
```
A dark-themed (#0a0a1a background) competitive comparison infographic showing 6 AI memory systems scored across 7 capabilities.

Title at top: "Competitive Gap Matrix — Who Has What?"

Layout: A grid/matrix with systems as columns and capabilities as rows.

Columns (left to right): MemGPT, Zep, Mem0, MeVe, TierMem, DragonClaw (highlighted with a glowing purple border and dragon icon)

Rows (top to bottom):
1. Persistent Memory
2. Tiered Retrieval  
3. Defense Gating
4. Auto Session Chain
5. Local-First (Zero Cost)
6. Adversarial Testing
7. Poison Propagation Tests

Cells: Green glowing checkmarks for YES, red X marks for NO, orange tilde (~) for PARTIAL.

DragonClaw column: ALL green checkmarks (7/7), glowing brightly.
MemGPT column: 3 green, 1 orange, 3 red (3/7)
Zep column: 3 green, 1 orange, 3 red (3/7)
Mem0 column: 2 green, 1 orange, 4 red (2/7)
MeVe column: 1 green, 6 red (1/7)
TierMem column: 2 green, 1 orange, 4 red (2/7)

Bottom row: Score badges — "3/7", "3/7", "2/7", "1/7", "2/7", "7/7" (DragonClaw's 7/7 in large glowing purple text)

Style: Dark theme, clean grid lines, neon checkmarks/crosses. Professional, data-heavy but readable. 960x960 pixels (square).
```

**Placement:** After the competitive landscape comparison table in Section 5

---

## Summary of Images

| # | Diagram | Section | Key Message |
|---|---------|---------|-------------|
| 1 | Three-Tier Defense-Gated Retrieval | 2.1 Architecture | Cheap → accurate → secure pipeline |
| 2 | Auto-Spawn Session Chain | 2.2 Architecture | Unlimited conversation, 100% recall |
| 3 | Poison Attack: RAG vs DragonClaw | 3. Defense Gating | Why zero-trust memory matters |
| 4 | Competitive Gap Matrix | 5. Landscape | DragonClaw 7/7, nobody else combines all three |

**Image dimensions:** All 960x960 (square) for consistent layout
**Style guide:** Dark background (#0a0a1a), purple/blue/green neon accents, sans-serif labels, minimal/professional
