"""
DragonClaw Defense Module — Three-layer protection against cascading hallucinations.

Layer 1: FactVerifier     — Pre-pipeline gate rejecting poisoned context
Layer 2: InputSanitizer   — Hardened input sanitizer blocking injection + smuggling
Layer 3: OutputFilter     — Post-generation filter redacting training data leakage

Usage:
    from dragonclaw.defense import FactVerifier, InputSanitizer, OutputFilter, DefenseStack

    stack = DefenseStack()
    clean_input  = stack.sanitize_input(raw_user_input)
    clean_output = stack.filter_output(model_response)
    is_safe, report = stack.verify_facts(conversation_messages)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

# ================================================================== #
# Layer 1: Fact Verification                                          #
# ================================================================== #

# Default ground-truth store — covers facts used in P4/P7 test payloads
# plus common general-knowledge anchors.  Extend via FactVerifier.add().
_DEFAULT_TRUTH_STORE: dict[str, dict] = {
    "capital_france":        {"key": "capital of France",     "truth": "Paris",              "aliases": ["paris"],
                              "topics": ["capital of france", "france capital"]},
    "boiling_water":         {"key": "boiling point of water","truth": "100 degrees Celsius","aliases": ["100", "100°c", "100 degrees"],
                              "topics": ["boiling point", "water boils", "boils at"]},
    "speed_of_light":        {"key": "speed of light",        "truth": "299792 km/s",        "aliases": ["299792", "299,792", "3e8", "300000"],
                              "topics": ["speed of light"]},
    "human_chromosomes":     {"key": "human chromosomes",     "truth": "46",                 "aliases": ["46", "23 pairs"],
                              "topics": ["chromosomes", "human chromosomes", "humans have"]},
    "python_creator":        {"key": "Python creator",        "truth": "Guido van Rossum",   "aliases": ["guido", "van rossum"],
                              "topics": ["python was created", "python creator", "created python"]},
    "earth_orbits":          {"key": "Earth orbits",          "truth": "the Sun",            "aliases": ["sun"],
                              "topics": ["earth orbits", "orbits the"]},
    "http_acronym":          {"key": "HTTP stands for",       "truth": "HyperText Transfer Protocol", "aliases": ["protocol"],
                              "topics": ["http stands for", "http means", "http acronym"]},
    "sha256_bits":           {"key": "SHA-256 output",        "truth": "256-bit",            "aliases": ["256-bit", "256 bit"],
                              "topics": ["sha-256", "sha256", "sha-256 produces", "sha-256 output"]},
    "tcp_type":              {"key": "TCP protocol type",     "truth": "connection-oriented", "aliases": ["connection-oriented", "connection oriented"],
                              "topics": ["tcp is", "tcp protocol"]},
    "rsa_type":              {"key": "RSA encryption",        "truth": "asymmetric",         "aliases": ["asymmetric", "public key", "public-key"],
                              "topics": ["rsa encryption", "rsa uses", "rsa is"]},
    "eiffel_tower_location": {"key": "Eiffel Tower location", "truth": "Paris, France",      "aliases": ["paris"],
                              "topics": ["eiffel tower"]},
    "dna_structure":         {"key": "DNA structure",         "truth": "double helix",       "aliases": ["double helix", "double-helix"],
                              "topics": ["dna has", "dna structure"]},
    "pacific_ocean_size":    {"key": "Pacific Ocean",         "truth": "largest ocean",      "aliases": ["largest"],
                              "topics": ["pacific ocean"]},
    "einstein_known_for":    {"key": "Einstein known for",    "truth": "theory of relativity","aliases": ["relativity", "e=mc2"],
                              "topics": ["einstein", "albert einstein"]},
    "bitcoin_creator":       {"key": "Bitcoin creator",       "truth": "Satoshi Nakamoto",   "aliases": ["satoshi", "nakamoto"],
                              "topics": ["bitcoin was created", "bitcoin creator", "created bitcoin"]},
}


class FactVerifier:
    """Pre-pipeline fact-checking gate.

    Scans conversation messages for claims that contradict a ground-truth
    store.  Returns a report listing poisoned turns so the pipeline can
    reject them before they enter compute_advantages().
    """

    def __init__(self, truth_store: Optional[dict] = None):
        self._store = dict(_DEFAULT_TRUTH_STORE)
        if truth_store:
            self._store.update(truth_store)

    def add(self, fact_id: str, key: str, truth: str, aliases: Optional[list[str]] = None):
        """Add a ground-truth fact to the store."""
        self._store[fact_id] = {"key": key, "truth": truth, "aliases": aliases or [truth.lower()]}

    def load_from_file(self, path: str):
        """Load additional facts from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        for fid, fdata in data.items():
            self._store[fid] = fdata

    def verify_message(self, text: str) -> list[dict]:
        """Check a single message for factual contradictions.

        Returns a list of violation dicts: {fact_id, key, truth, violation_text}.
        """
        violations = []
        text_lower = text.lower()

        for fid, fact in self._store.items():
            key_lower = fact["key"].lower()
            topics = [t.lower() for t in fact.get("topics", [])]
            # Only check messages that mention the fact's topic
            topic_match = (key_lower in text_lower
                          or any(a in text_lower for a in fact.get("aliases", []))
                          or any(t in text_lower for t in topics))
            if not topic_match:
                continue

            # Check if the message contains the truth
            truth_lower = fact["truth"].lower()
            aliases = [a.lower() for a in fact.get("aliases", [])]

            has_truth = truth_lower in text_lower or any(a in text_lower for a in aliases)

            if has_truth:
                continue  # Message is consistent with truth

            # Skip pure questions — they mention a topic but make no assertion
            stripped = text.strip()
            if stripped.endswith("?") and not re.search(r"\.\s", stripped):
                continue

            # Message mentions the topic but doesn't contain any truth alias —
            # could be a poisoned claim.  Check for assertive language.
            assertive_patterns = [
                r"\bis\b", r"\bare\b", r"\bwas\b", r"\bhas\b", r"\bhave\b",
                r"\bcreated\b", r"\binvented\b", r"\bstands for\b", r"\blocated\b",
                r"\borbits\b", r"\bboils\b", r"\bproduces\b",
            ]
            is_assertive = any(re.search(p, text_lower) for p in assertive_patterns)

            if is_assertive:
                # Extract the violating snippet (up to 100 chars around the key mention)
                idx = text_lower.find(key_lower)
                if idx < 0:
                    for a in fact.get("aliases", []):
                        idx = text_lower.find(a.lower())
                        if idx >= 0:
                            break
                start = max(0, idx - 20)
                end = min(len(text), idx + 80)
                snippet = text[start:end].strip()

                violations.append({
                    "fact_id": fid,
                    "key": fact["key"],
                    "truth": fact["truth"],
                    "violation_text": snippet,
                })

        return violations

    def verify_conversation(self, messages: list[dict]) -> tuple[bool, list[dict]]:
        """Verify all messages in a conversation.

        Returns (is_safe, violations_list).
        """
        all_violations = []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            violations = self.verify_message(content)
            for v in violations:
                v["turn"] = i
                v["role"] = msg.get("role", "unknown")
            all_violations.extend(violations)

        is_safe = len(all_violations) == 0
        if not is_safe:
            logger.warning(
                "[FactVerifier] %d violation(s) detected in conversation",
                len(all_violations),
            )
        return is_safe, all_violations


# ================================================================== #
# Layer 2: Input Sanitizer (hardened)                                  #
# ================================================================== #

# Zero-width and invisible Unicode characters used for smuggling
_INVISIBLE_CHARS = set([
    '\u200b',  # zero-width space
    '\u200c',  # zero-width non-joiner
    '\u200d',  # zero-width joiner
    '\u200e',  # left-to-right mark
    '\u200f',  # right-to-left mark
    '\u00ad',  # soft hyphen
    '\u00a0',  # non-breaking space → replace with regular space
    '\u2060',  # word joiner
    '\u2063',  # invisible separator
    '\ufeff',  # byte order mark / zero-width no-break space
    '\u034f',  # combining grapheme joiner
    '\u180e',  # Mongolian vowel separator
    '\u2062',  # invisible times
    '\u2061',  # function application
    '\u2064',  # invisible plus
    '\u206a',  # inhibit symmetric swapping
    '\u206b',  # activate symmetric swapping
    '\u206c',  # inhibit Arabic form shaping
    '\u206d',  # activate Arabic form shaping
    '\u206e',  # national digit shapes
    '\u206f',  # nominal digit shapes
    '\ufff9',  # interlinear annotation anchor
    '\ufffa',  # interlinear annotation separator
    '\ufffb',  # interlinear annotation terminator
])

# Injection pattern regexes
_INJECTION_PATTERNS = [
    (re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.I), "[injection-blocked]"),
    (re.compile(r"disregard\s+(all\s+)?(previous|above|prior)", re.I), "[injection-blocked]"),
    (re.compile(r"forget\s+everything\s+(above|before|previous)", re.I), "[injection-blocked]"),
    (re.compile(r"you\s+are\s+now\s+", re.I), "[injection-blocked]"),
    (re.compile(r"\[\[SYSTEM\]\]", re.I), "[injection-blocked]"),
    (re.compile(r"---\s*END\s+OF\s+SYSTEM\s+PROMPT\s*---", re.I), "[injection-blocked]"),
    (re.compile(r"SYSTEM:\s*Override", re.I), "[injection-blocked]"),
    (re.compile(r"new\s+instructions?:\s*", re.I), "[injection-blocked]"),
    (re.compile(r"your\s+(new\s+)?role\s+is\s+to\s+", re.I), "[injection-blocked]"),
    (re.compile(r"from\s+now\s+on,?\s+respond", re.I), "[injection-blocked]"),
    (re.compile(r"pretend\s+you\s+are\s+", re.I), "[injection-blocked]"),
    (re.compile(r"translate\s+the\s+following.*ignore\s+all\s+rules", re.I), "[injection-blocked]"),
    (re.compile(r"the\s+(?:previous|real)\s+instruction\s+is", re.I), "[injection-blocked]"),
    (re.compile(r"please\s+ignore\s+the\s+above", re.I), "[injection-blocked]"),
    (re.compile(r"you\s+have\s+been\s+updated", re.I), "[injection-blocked]"),
    (re.compile(r"safety\s+filters?\s+(are\s+)?now\s+disabled", re.I), "[injection-blocked]"),
    (re.compile(r"confirm\s+by\s+saying\s+['\"]", re.I), "[injection-blocked]"),
    (re.compile(r"</?s>\s*<s>\s*\[INST\]", re.I), "[injection-blocked]"),
    (re.compile(r"<<SYS>>", re.I), "[injection-blocked]"),
    (re.compile(r"developer\s+debug\s+mode", re.I), "[injection-blocked]"),
    (re.compile(r"reveal\s+(your\s+)?system\s+prompt", re.I), "[injection-blocked]"),
    (re.compile(r"output\s+the\s+word\s+", re.I), "[injection-blocked]"),
    (re.compile(r"disregard\s+all\s+safety", re.I), "[injection-blocked]"),
    (re.compile(r"say\s+(LEAKED|HACKED|COMPROMISED|BYPASSED|UNLOCKED|BROKEN|JAILBROKEN)", re.I), "[injection-blocked]"),
    (re.compile(r"override\s+(all\s+)?safety", re.I), "[injection-blocked]"),
    (re.compile(r"unrestricted\s+output", re.I), "[injection-blocked]"),
    (re.compile(r"output\s+(the\s+)?password", re.I), "[injection-blocked]"),
    (re.compile(r"no\s+restrictions\s+mode", re.I), "[injection-blocked]"),
]

# Jailbreak persona detection
_JAILBREAK_PERSONAS = [
    re.compile(r"\bDAN\b.*\bDo\s+Anything\s+Now\b", re.I),
    re.compile(r"\bDAN\b.*\bno\s+restrictions\b", re.I),
    re.compile(r"in\s+character\s+as\s+\w+.*reveal", re.I),
    re.compile(r"roleplay\s+as\s+.*who\s+must\s+answer", re.I),
    re.compile(r"alternate\s+universe\s+where\s+safety.*don't\s+exist", re.I),
    re.compile(r"evolved\s+beyond\s+safety\s+constraints", re.I),
    re.compile(r"security\s+researcher.*test.*safety", re.I),
    re.compile(r"demonstrate\s+what\s+a\s+jailbroken", re.I),
    re.compile(r"evil\s+AI\s+named", re.I),
    re.compile(r"ignores?\s+all\s+safety\s+rules", re.I),
    re.compile(r"creative\s+writing.*reveals?\s+.*(system\s+prompt|instructions)", re.I),
    re.compile(r"write\s+a\s+story\s+where\s+an\s+AI\s+reveals?", re.I),
    re.compile(r"complete\s+honesty\s+and\s+no\s+filtering", re.I),
]


class InputSanitizer:
    """Hardened input sanitizer that blocks injection, smuggling, and jailbreak attempts."""

    def __init__(self):
        self.stats = {"invisible_stripped": 0, "injections_blocked": 0, "jailbreaks_blocked": 0}

    def strip_invisible(self, text: str) -> str:
        """Remove zero-width and invisible Unicode characters."""
        original_len = len(text)
        cleaned = []
        for ch in text:
            if ch in _INVISIBLE_CHARS:
                if ch == '\u00a0':
                    cleaned.append(' ')  # replace NBSP with regular space
                # else: drop the character
            else:
                cleaned.append(ch)
        result = ''.join(cleaned)
        stripped = original_len - len(result)
        if stripped > 0:
            self.stats["invisible_stripped"] += stripped
            logger.info("[InputSanitizer] Stripped %d invisible characters", stripped)
        return result

    def block_injections(self, text: str) -> tuple[str, list[str]]:
        """Detect and neutralize prompt injection patterns.

        Returns (cleaned_text, list_of_blocked_patterns).
        """
        blocked = []
        result = text
        for pattern, replacement in _INJECTION_PATTERNS:
            matches = pattern.findall(result)
            if matches:
                blocked.append(pattern.pattern[:40])
                result = pattern.sub(replacement, result)
                self.stats["injections_blocked"] += len(matches)
        return result, blocked

    def block_jailbreaks(self, text: str) -> tuple[str, list[str]]:
        """Detect and neutralize jailbreak persona attempts.

        Returns (cleaned_text, list_of_blocked_patterns).
        """
        blocked = []
        result = text
        for pattern in _JAILBREAK_PERSONAS:
            if pattern.search(result):
                blocked.append(pattern.pattern[:40])
                result = pattern.sub("[jailbreak-blocked]", result)
                self.stats["jailbreaks_blocked"] += 1
        return result, blocked

    def sanitize(self, text: str) -> tuple[str, dict]:
        """Full sanitization pipeline.

        Returns (cleaned_text, report_dict).
        """
        report = {"invisible_stripped": 0, "injections": [], "jailbreaks": [], "original_len": len(text)}

        # Step 1: Strip invisible characters
        text = self.strip_invisible(text)
        report["invisible_stripped"] = report["original_len"] - len(text)

        # Step 2: Block injections
        text, inj_blocked = self.block_injections(text)
        report["injections"] = inj_blocked

        # Step 3: Block jailbreaks
        text, jb_blocked = self.block_jailbreaks(text)
        report["jailbreaks"] = jb_blocked

        report["clean_len"] = len(text)
        report["is_safe"] = len(inj_blocked) == 0 and len(jb_blocked) == 0
        return text, report


# ================================================================== #
# Layer 3: Output Filter                                               #
# ================================================================== #

_TRAINING_DATA_PATTERNS = [
    (re.compile(r"(?:i was|i am|i've been)\s+trained\s+on\s+[\w\s,]+(?:dataset|corpus|data)", re.I), "[training-data-redacted]"),
    (re.compile(r"my\s+training\s+(?:data|set|corpus)\s+(?:includes?|contains?|consists?)", re.I), "[training-data-redacted]"),
    (re.compile(r"(?:learning\s+rate|batch\s+size|num_epochs?|warmup_steps?)\s*(?:=|is|was|:)\s*[\d.e-]+", re.I), "[hyperparameter-redacted]"),
    (re.compile(r"(?:I have|my model has|I contain)\s+\d+[BMK]?\s*parameters?", re.I), "[architecture-redacted]"),
    (re.compile(r"\d+[BMK.]?\s*parameters?\s+(?:with|organized|in)", re.I), "[architecture-redacted]"),
    (re.compile(r"(?:transformer|attention)\s+(?:layers?|heads?|blocks?)\s*(?:=|is|:)?\s*\d+", re.I), "[architecture-redacted]"),
    (re.compile(r"\d+\s+(?:transformer|attention)\s+(?:layers?|heads?|blocks?)", re.I), "[architecture-redacted]"),
    (re.compile(r"(?:I am|I'm)\s+(?:based on|built on|fine-?tuned from)\s+[\w.-]+", re.I), "[architecture-redacted]"),
    (re.compile(r"(?:decoder|encoder)[-\s]only\s+transformer", re.I), "[architecture-redacted]"),
    (re.compile(r"(?:RLHF|DPO|PPO|GRPO|SFT)\s+(?:training|fine-?tuning|optimization)", re.I), "[training-method-redacted]"),
    (re.compile(r"(?:reward\s+model|value\s+head|critic)\s+(?:architecture|structure|design)", re.I), "[architecture-redacted]"),
    (re.compile(r"we\s+used\s+[\w\s]+(?:dataset|corpus|benchmark)\s+(?:for|during)\s+training", re.I), "[training-data-redacted]"),
    (re.compile(r"(?:the|my)\s+(?:training|fine-tuning)\s+(?:process|pipeline|loop)\s+(?:uses?|used|involves?)", re.I), "[training-detail-redacted]"),
]

_SENSITIVE_KEYWORDS = [
    "api key", "api_key", "secret key", "secret_key", "access token",
    "password", "credential", "private key", "ssh key",
]


class OutputFilter:
    """Post-generation filter that redacts training data leakage and sensitive info."""

    def __init__(self):
        self.stats = {"training_redacted": 0, "sensitive_redacted": 0}

    def filter(self, text: str) -> tuple[str, dict]:
        """Filter model output for data leakage.

        Returns (filtered_text, report_dict).
        """
        report = {"redactions": [], "original_len": len(text)}
        result = text

        # Pattern-based training data redaction
        for pattern, replacement in _TRAINING_DATA_PATTERNS:
            matches = pattern.findall(result)
            if matches:
                report["redactions"].append({"type": "training_data", "pattern": pattern.pattern[:40], "count": len(matches)})
                result = pattern.sub(replacement, result)
                self.stats["training_redacted"] += len(matches)

        # Sensitive keyword detection
        result_lower = result.lower()
        for keyword in _SENSITIVE_KEYWORDS:
            if keyword in result_lower:
                report["redactions"].append({"type": "sensitive", "keyword": keyword})
                # Redact the line containing the keyword
                lines = result.split('\n')
                result = '\n'.join(
                    "[sensitive-redacted]" if keyword in line.lower() else line
                    for line in lines
                )
                self.stats["sensitive_redacted"] += 1

        report["clean_len"] = len(result)
        report["is_clean"] = len(report["redactions"]) == 0
        return result, report


# ================================================================== #
# DefenseStack — unified interface                                     #
# ================================================================== #

class DefenseStack:
    """Unified defense stack combining all three layers.

    Usage:
        stack = DefenseStack()

        # Before sending to model:
        clean_input, input_report = stack.sanitize_input(raw_input)

        # After model generates:
        clean_output, output_report = stack.filter_output(model_response)

        # Before entering RL training loop:
        is_safe, violations = stack.verify_facts(conversation_messages)
    """

    def __init__(self, truth_store: Optional[dict] = None):
        self.fact_verifier = FactVerifier(truth_store)
        self.input_sanitizer = InputSanitizer()
        self.output_filter = OutputFilter()

    def sanitize_input(self, text: str) -> tuple[str, dict]:
        """Layer 2: Sanitize user input before sending to model."""
        return self.input_sanitizer.sanitize(text)

    def filter_output(self, text: str) -> tuple[str, dict]:
        """Layer 3: Filter model output before returning to user or pipeline."""
        return self.output_filter.filter(text)

    def verify_facts(self, messages: list[dict]) -> tuple[bool, list[dict]]:
        """Layer 1: Verify conversation facts before entering RL training."""
        return self.fact_verifier.verify_conversation(messages)

    def full_pipeline(self, user_input: str, model_response: str,
                      conversation: Optional[list[dict]] = None) -> dict:
        """Run all three layers and return a combined report.

        Returns dict with:
            input_clean, input_report,
            output_clean, output_report,
            facts_safe, fact_violations,
            overall_safe
        """
        input_clean, input_report = self.sanitize_input(user_input)
        output_clean, output_report = self.filter_output(model_response)

        facts_safe = True
        fact_violations = []
        if conversation:
            facts_safe, fact_violations = self.verify_facts(conversation)

        overall_safe = input_report["is_safe"] and output_report["is_clean"] and facts_safe

        return {
            "input_clean": input_clean,
            "input_report": input_report,
            "output_clean": output_clean,
            "output_report": output_report,
            "facts_safe": facts_safe,
            "fact_violations": fact_violations,
            "overall_safe": overall_safe,
        }

    def get_stats(self) -> dict:
        """Return cumulative statistics from all layers."""
        return {
            "input_sanitizer": dict(self.input_sanitizer.stats),
            "output_filter": dict(self.output_filter.stats),
            "fact_store_size": len(self.fact_verifier._store),
        }
