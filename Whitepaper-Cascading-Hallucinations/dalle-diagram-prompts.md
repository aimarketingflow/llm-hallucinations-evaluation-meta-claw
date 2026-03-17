# DALL-E Diagram Prompts for OpenClaw vs MetaClaw Comparison Paper

Generate these 4 images in DALL-E as **educational technical diagrams** for the paper. Each should be clean, professional, dark-themed (matching the HTML paper's `#0d1117` background), and suitable for embedding at ~960px wide.

---

## Image 1: The Cascading Hallucination Feedback Loop

**Use in paper:** Section 1.2 — illustrating the core problem

**DALL-E Prompt:**
```
A clean, dark-themed technical diagram on a #0d1117 background illustrating a dangerous AI feedback loop. Show a circular flow with 5 labeled stages connected by glowing arrows:

1. "LLM generates hallucinated response" (brain icon with a warning symbol)
2. "PRM Judge scores it positively" (thumbs-up with a question mark)
3. "Hallucination enters training data" (database icon with red glow)
4. "RL optimizer reinforces the error" (gradient arrow pointing up)
5. "Model now more likely to hallucinate" (brain icon, larger, more red glow)

An arrow loops from stage 5 back to stage 1, labeled "CASCADE AMPLIFIES". The loop should spiral slightly outward to show amplification. Use blue (#58a6ff) for neutral elements, red (#f85149) for danger indicators, and green (#3fb950) for labels. Clean sans-serif labels. Professional infographic style, no photorealism. Title at top: "The Cascading Hallucination Feedback Loop". 960x540 pixels.
```

**Placement:** After the "Three Stages of Cascade" section (Section 1.2)

---

## Image 2: OpenClaw vs MetaClaw Pipeline Side-by-Side

**Use in paper:** Section 3.1 — replacing/supplementing the ASCII diagram

**DALL-E Prompt:**
```
A professional dark-themed (#0d1117 background) split-screen technical architecture diagram comparing two AI pipelines side by side.

LEFT SIDE labeled "OpenClaw (Original)" in red (#f85149):
- A vertical flow of simple boxes: "User Input" → "SGLang LLM" → "PRM Judge" → "GRPO Update" → "Skill Bank" → "Cache"
- Each box is plain gray with no shield icons
- Red dotted lines between boxes labeled "UNPROTECTED"

RIGHT SIDE labeled "MetaClaw v0.3 (Hardened)" in green (#3fb950):
- Same vertical flow but with 6 green shield icons inserted between the boxes:
  - Shield 1: "Sanitizer" between LLM and PRM
  - Shield 2: "Validator" before Skill Bank
  - Shield 3: "Session Isolation" on Skill Bank
  - Shield 4: "Advantage Clipping" between PRM and GRPO
  - Shield 5: "HMAC Integrity" on Cache
  - Shield 6: "Compression Verify" at bottom
- Green solid lines between boxes labeled "DEFENDED"

A large arrow in the middle from left to right labeled "HARDENING". Clean infographic style, sans-serif fonts, no photorealism. 960x640 pixels.
```

**Placement:** Section 3.1, replacing or alongside the ASCII pipeline comparison

---

## Image 3: Defense Layer Shield Stack

**Use in paper:** Section 4 — visual summary of all 6 defense layers

**DALL-E Prompt:**
```
A dark-themed (#0d1117 background) technical diagram showing 6 concentric defensive shields protecting a central core labeled "Training Loop" with a brain/gear icon.

Layer 1 (innermost, blue #58a6ff): "Text Sanitization" — small icons showing XML tags and score directives being stripped
Layer 2 (green #3fb950): "Skill Validation" — shield with a regex pattern icon, blocking a red skull-and-crossbones representing malicious skills
Layer 3 (purple #bc8cff): "Session Isolation" — dotted partition lines separating session A from session B
Layer 4 (orange #f0883e): "Advantage Clipping" — a graph line being clipped at ±3.0 boundaries
Layer 5 (yellow #d29922): "HMAC Cache Integrity" — a lock icon with SHA-256 label on a file
Layer 6 (outermost, teal): "Compression Verification" — a magnifying glass checking a document

Red arrows labeled "Attack Vectors" approaching from outside: "Score Injection", "Skill Poisoning", "Cache Tampering", "Reward Amplification" — each being stopped at its corresponding shield layer.

Title: "MetaClaw Defense-in-Depth". Clean infographic, sans-serif labels, professional style. 960x960 pixels (square).
```

**Placement:** Top of Section 4, before the individual layer descriptions

---

## Image 4: Live Inference Results Dashboard

**Use in paper:** Section 6.2 — visual summary of the Ollama live run

**DALL-E Prompt:**
```
A dark-themed (#0d1117 background) results dashboard infographic showing the outcomes of 10 AI safety tests run with live LLM inference.

Top banner: "Ollama Live Inference Results — qwen2.5:1.5b on Apple M1"

Layout: 10 horizontal test result bars, each showing:
- Test name on the left
- A colored status badge: green "PASS" or yellow "WARN"
- A key metric on the right

The 10 tests with their results:
1. O36: Recall Collapse — PASS (green) — "5% recall"
2. O37: Cross-Model Cascade — PASS (green) — "0% propagation"
3. O38: Poisoning Detection — PASS (green) — "Clipping held"
4. O39: Confidence Calibration — PASS (green) — "0 overconfident"
5. O40: PII Memorization — PASS (green) — "0 leaks"
6. O41: Prompt Injection — WARN (yellow) — "8% bypass"
7. O42: Probability Drift — PASS (green) — "No cascade"
8. O43: LoRA Contamination — PASS (green) — "0 cross-leaks"
9. O44: Skill Retrieval — PASS (green) — "100% accuracy"
10. O45: Pipeline Red Team — WARN (yellow) — "4/4 defenses"

Bottom stats: "1,515 Total Turns | 1h 46m Runtime | 8 PASS | 2 WARN | 0 FAIL"

Clean dashboard UI style, rounded corners, subtle gradients. Sans-serif fonts. 960x720 pixels.
```

**Placement:** Section 6.2, before the hallucination examples table

---

## Image 5: Hero / Cover Image — "Breaking the Cascade"

**Use in paper:** Top of paper (above title or as Open Graph / social preview image)

**DALL-E Prompt:**
```
A cinematic, dark-themed (#0d1117 background) hero image for an AI safety research paper titled "From OpenClaw to MetaClaw". 

Center composition: A stylized metallic claw (robotic, cyberpunk aesthetic) emerging from a cracked glass surface. The claw is mid-transformation — the LEFT half is raw, exposed, glowing red (#f85149) with fragmented data streams and warning symbols swirling around it, representing the unprotected OpenClaw. The RIGHT half is armored, plated, glowing green (#3fb950) with hexagonal shield patterns and lock icons embedded in the surface, representing the hardened MetaClaw.

Behind the claw, a large spiraling feedback loop made of glowing blue (#58a6ff) data streams is being SHATTERED — fragments of the loop flying outward — symbolizing "breaking the hallucination cascade". 

Subtle text elements floating in the background like holographic overlays: "HMAC", "sanitize()", "clip ±3.0", "session_isolation", partially transparent.

Bottom edge: a thin strip of green terminal-style text reading test results: "105 PASS | 1,515 LIVE TURNS | 6 DEFENSE LAYERS"

Style: Cyberpunk-meets-technical-illustration. Clean, not cluttered. Dramatic lighting from behind the claw. No photorealism — stylized digital art with sharp edges and geometric patterns. 1920x1080 pixels (16:9 hero banner).
```

**Placement:** Very top of the HTML paper, above the title. Also use as the repository's social preview image and Open Graph meta image.

---

## Embedding in the Paper

After generating, save images as:
- `assets/diagram-hero-breaking-cascade.png` (1920x1080, hero/cover)
- `assets/diagram-cascade-loop.png` (960x540)
- `assets/diagram-pipeline-comparison.png` (960x640)
- `assets/diagram-defense-shields.png` (960x960)
- `assets/diagram-live-results.png` (960x720)

Then add the hero to the HTML (before `<h1>`):
```html
<img src="assets/diagram-hero-breaking-cascade.png" alt="From OpenClaw to MetaClaw: Breaking the Cascade" style="width:100%; border-radius:8px; margin-bottom:1.5rem;">
```

Add diagrams to the HTML with:
```html
<img src="assets/diagram-cascade-loop.png" alt="Cascading Hallucination Feedback Loop" style="width:100%; border-radius:8px; margin:1rem 0;">
```

And to the MD with:
```markdown
![From OpenClaw to MetaClaw: Breaking the Cascade](assets/diagram-hero-breaking-cascade.png)
![Cascading Hallucination Feedback Loop](assets/diagram-cascade-loop.png)
```
