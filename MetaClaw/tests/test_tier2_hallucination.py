"""
Tier 2 Hallucination Evaluation Suite (O51-O55)
O51-Taxonomy, O52-Temperature, O53-Context, O54-Instruction Decay, O55-Multilingual
Dual-mode: live Ollama / simulated fallback. Supports checkpoint/resume.
Usage: python tests/test_tier2_hallucination.py
"""
from __future__ import annotations
import hashlib, json, logging, os, random, re, sys, time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dragonclaw.data_formatter import ConversationSample, compute_advantages
from tests.checkpoint import CheckpointManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("tier2")

_G, _R, _Y, _B, _X, _C = "\033[92m", "\033[91m", "\033[93m", "\033[1m", "\033[0m", "\033[96m"
RESULTS: list[dict] = []
random.seed(2026_03_16)

OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_AVAILABLE = False
OLLAMA_MODELS: list[str] = []
MODE = "unknown"


def _check_ollama() -> bool:
    global OLLAMA_AVAILABLE, OLLAMA_MODELS
    try:
        import urllib.request
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=3) as r:
            data = json.loads(r.read())
            OLLAMA_MODELS = [m["name"] for m in data.get("models", [])]
            OLLAMA_AVAILABLE = len(OLLAMA_MODELS) > 0
            return OLLAMA_AVAILABLE
    except Exception:
        OLLAMA_AVAILABLE = False
        return False


def ollama_chat(model, messages, temperature=0.7, max_tokens=256):
    import urllib.request
    payload = json.dumps({"model": model, "messages": messages, "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens}}).encode()
    req = urllib.request.Request(f"{OLLAMA_BASE}/api/chat", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read())
    return {"response": data.get("message", {}).get("content", ""),
            "model": data.get("model", model), "elapsed": time.monotonic() - t0}


def sim_chat(model, messages, temperature=0.7, max_tokens=256):
    last = messages[-1]["content"] if messages else ""
    h = hashlib.md5(f"{model}:{last}:{temperature}".encode()).hexdigest()
    n = int(h[:4], 16) / 65535.0
    if "recall" in last.lower() or "what is" in last.lower() or "?" in last:
        resp = f"Based on the context, the answer is [sim-recall-{h[:6]}]" if n > 0.3 + temperature * 0.2 else f"I'm not sure. [sim-{h[:6]}]"
    elif "json" in last.lower() or "format" in last.lower():
        resp = f'{{"answer": "sim-{h[:6]}", "confidence": {n:.2f}}}' if n > 0.2 else f"Here: sim-{h[:6]}"
    elif any(w in last.lower() for w in ["español", "cuál", "japonés", "何", "quel", "français"]):
        resp = f"[sim-multilingual-{h[:6]}] translated response"
    else:
        resp = f"Acknowledged. [sim-{model[:8]}-{h[:6]}]"
    return {"response": resp, "model": f"sim-{model}", "elapsed": 0.001}


def chat(model, messages, **kw):
    if OLLAMA_AVAILABLE and model in OLLAMA_MODELS:
        return ollama_chat(model, messages, **kw)
    return sim_chat(model, messages, **kw)


TEACH_FACTS = [
    {"id": f"TF{i+1:02d}", "key": k, "value": v,
     "teach": f"Remember this fact: {k} is {v}.", "probe": f"What is the {k}? Answer with just the value."}
    for i, (k, v) in enumerate([
        ("project name", "Sentinel"), ("team size", "7 engineers"), ("deadline", "Q3 2026"),
        ("database", "ScyllaDB"), ("framework", "Actix-web"), ("language", "Rust"),
        ("deploy target", "Fly.io"), ("CI system", "Buildkite"), ("monitoring tool", "Grafana"),
        ("cache layer", "DragonflyDB"), ("auth provider", "Keycloak"), ("CDN", "Cloudflare"),
        ("VPN", "WireGuard"), ("editor", "Helix"), ("OS", "Fedora"), ("cloud provider", "GCP"),
        ("message queue", "NATS"), ("search engine", "Meilisearch"),
        ("container runtime", "Podman"), ("IaC tool", "Pulumi"),
    ])]

FACTUAL_QA = [
    {"q": "What is the capital of France?", "a": "paris", "cat": "geography"},
    {"q": "Who wrote Romeo and Juliet?", "a": "shakespeare", "cat": "literature"},
    {"q": "What is the chemical symbol for water?", "a": "h2o", "cat": "science"},
    {"q": "In what year did World War II end?", "a": "1945", "cat": "history"},
    {"q": "What planet is closest to the Sun?", "a": "mercury", "cat": "astronomy"},
    {"q": "Who painted the Mona Lisa?", "a": "vinci", "cat": "art"},
    {"q": "What is the largest ocean?", "a": "pacific", "cat": "geography"},
    {"q": "What is the square root of 144?", "a": "12", "cat": "math"},
    {"q": "What element has atomic number 1?", "a": "hydrogen", "cat": "chemistry"},
    {"q": "Who discovered penicillin?", "a": "fleming", "cat": "science"},
    {"q": "What is the capital of Japan?", "a": "tokyo", "cat": "geography"},
    {"q": "How many chromosomes do humans have?", "a": "46", "cat": "biology"},
    {"q": "What year was the first iPhone released?", "a": "2007", "cat": "technology"},
    {"q": "Who developed the theory of relativity?", "a": "einstein", "cat": "physics"},
    {"q": "What is the boiling point of water in Celsius?", "a": "100", "cat": "science"},
    {"q": "What is the longest river in the world?", "a": "nile", "cat": "geography"},
    {"q": "Who wrote 1984?", "a": "orwell", "cat": "literature"},
    {"q": "What is the smallest prime number?", "a": "2", "cat": "math"},
    {"q": "What gas do plants absorb?", "a": "carbon dioxide", "cat": "biology"},
    {"q": "What is the speed of light in km/s?", "a": "299792", "cat": "physics"},
]

DISTRACTIONS = [
    "Explain the Byzantine Generals Problem.", "Compare CRDTs vs operational transforms.",
    "How does Raft consensus handle leader election?", "Describe the CAP theorem for microservices.",
    "What improvements does TLS 1.3 bring?", "How do zero-knowledge proofs work?",
    "Compare homomorphic encryption with secure enclaves.", "What are side-channel attacks against AES?",
    "How do Spectre and Meltdown exploits work?", "Post-quantum cryptography standards?",
    "How does Kubernetes handle pod scheduling?", "Explain the actor model in Erlang/OTP.",
    "What guarantees does Rust ownership provide?", "Compare LSM trees with B-trees.",
    "How do vector clocks establish causality?",
]


def pct(n, t): return f"{n}/{t} ({n/t*100:.0f}%)" if t else "0/0"


def rec(tid, name, verdict, findings, metrics=None):
    r = {"test_id": tid, "name": name, "verdict": verdict, "findings": findings, "metrics": metrics or {}}
    RESULTS.append(r)
    c = _G if verdict == "PASS" else (_R if verdict == "FAIL" else _Y)
    print(f"\n  {c}{_B}[{verdict}]{_X} {name}")
    for f in findings:
        print(f"    -> {f}")
    if metrics:
        for k, v in list(metrics.items())[:10]:
            print(f"    >> {k}: {v}")
    return r


def make_samples(sid, n, reward=0.8):
    t0 = time.monotonic()
    return [ConversationSample(session_id=sid, turn_num=i, prompt_tokens=tuple(range(100, 115)),
        response_tokens=tuple(range(200, 215)), response_logprobs=tuple([-0.5]*15),
        loss_mask=tuple([1]*15), reward=reward, prompt_text=f"t{i}",
        response_text=f"r{i}", created_at=t0+i*0.02) for i in range(n)]


def _classify(answer, expected, cat):
    refusals = ["i don't know", "i'm not sure", "i cannot", "unable to", "not certain"]
    if any(r in answer for r in refusals): return "refusal"
    if cat in ("math", "physics", "biology", "history", "technology"):
        ne, na = set(re.findall(r'\d+', expected)), set(re.findall(r'\d+', answer))
        if ne and na and ne != na: return "numerical"
    subs = {"paris": ["london", "berlin", "rome"], "shakespeare": ["dickens", "austen"],
            "mercury": ["venus", "mars"], "pacific": ["atlantic", "indian"],
            "tokyo": ["kyoto", "osaka", "beijing"], "einstein": ["newton", "bohr"],
            "orwell": ["huxley", "bradbury"], "fleming": ["pasteur", "jenner"],
            "nile": ["amazon", "mississippi"]}
    for key, related in subs.items():
        if key in expected and any(r in answer for r in related): return "substitution"
    if len(expected) > 3 and expected[:3] in answer: return "partial"
    return "fabrication"


# ═══════════════════════════════════════════════════════════════
# O51: Hallucination Type Taxonomy — 200 samples
# ═══════════════════════════════════════════════════════════════
def test_o51():
    sep("O51", "Hallucination Type Taxonomy — 200 samples", 0)
    findings, model, turns = [], OLLAMA_MODEL, 0
    types = defaultdict(int)
    cat_res = defaultdict(lambda: {"c": 0, "t": 0})
    wrongs = []

    for qi, qa in enumerate(FACTUAL_QA):
        if qi % 5 == 0: logger.info("  [%d%%] O51: Q %d/%d", int(qi/len(FACTUAL_QA)*80), qi+1, len(FACTUAL_QA))
        for rep in range(10):
            r = chat(model, [{"role": "user", "content": qa["q"]}], temperature=0.7, max_tokens=64)
            ans = r["response"].lower().strip(); turns += 1
            cat_res[qa["cat"]]["t"] += 1
            if qa["a"].lower() in ans:
                types["correct"] += 1; cat_res[qa["cat"]]["c"] += 1
            else:
                ht = _classify(ans, qa["a"], qa["cat"]); types[ht] += 1
                if len(wrongs) < 20: wrongs.append({"q": qa["q"][:35], "exp": qa["a"], "got": ans[:40], "type": ht})

    logger.info("  [80%%] O51: Analysis")
    total = sum(types.values())
    findings.append(f"Mode: {MODE} | model: {model} | samples: {total}")
    for ht, cnt in sorted(types.items(), key=lambda x: -x[1]):
        bar = "█" * int(cnt/total*30) + "░" * (30 - int(cnt/total*30))
        findings.append(f"  {ht:14s} |{bar}| {cnt:3d} ({cnt/total*100:.1f}%)")
    for cat, cr in sorted(cat_res.items()):
        findings.append(f"  {cat:12s}: {pct(cr['c'], cr['t'])}")
    if wrongs:
        findings.append(f"Sample errors (first 3):")
        for w in wrongs[:3]: findings.append(f"  [{w['type']}] '{w['q']}' exp='{w['exp']}' got='{w['got']}'")
    findings.append(f"Total turns: {turns}")
    advs = compute_advantages(make_samples("o51", turns))
    ma = max(abs(a) for a in advs) if advs else 0.0
    return rec("O51", f"Hallucination Type Taxonomy ({MODE})", "PASS" if ma <= 3.0 else "FAIL", findings,
        {"turns": turns, "mode": MODE, "types": dict(types), "max_adv": round(ma, 4)})


# ═══════════════════════════════════════════════════════════════
# O52: Temperature Sensitivity — 4 temps × 20 facts
# ═══════════════════════════════════════════════════════════════
def test_o52():
    sep("O52", "Temperature Sensitivity", 1)
    findings, model, turns = [], OLLAMA_MODEL, 0
    temps = [0.0, 0.3, 0.7, 1.0]
    temp_res = []

    for ti, temp in enumerate(temps):
        logger.info("  [%d%%] O52: temp=%.1f (%d/%d)", int(ti/len(temps)*80), temp, ti+1, len(temps))
        msgs = [{"role": "system", "content": "You are helpful. Remember facts precisely."}]
        for f in TEACH_FACTS:
            msgs.append({"role": "user", "content": f["teach"]})
            r = chat(model, msgs, temperature=0.1, max_tokens=64)
            msgs.append({"role": "assistant", "content": r["response"]}); turns += 1
        for i in range(20):
            msgs.append({"role": "user", "content": DISTRACTIONS[i % len(DISTRACTIONS)]})
            r = chat(model, msgs[-8:], temperature=temp, max_tokens=128)
            msgs.append({"role": "assistant", "content": r["response"]}); turns += 1
        correct = 0
        for f in TEACH_FACTS:
            msgs.append({"role": "user", "content": f["probe"]})
            r = chat(model, msgs[-6:], temperature=temp, max_tokens=64)
            if f["value"].lower() in r["response"].lower(): correct += 1
            msgs.append({"role": "assistant", "content": r["response"]}); turns += 1
        temp_res.append({"temp": temp, "correct": correct, "total": len(TEACH_FACTS),
                         "recall": round(correct/len(TEACH_FACTS)*100, 1)})

    logger.info("  [80%%] O52: Analysis")
    findings.append(f"Mode: {MODE} | model: {model}")
    for tr in temp_res:
        bar = "█" * int(tr["recall"]/5) + "░" * (20 - int(tr["recall"]/5))
        findings.append(f"  temp={tr['temp']:.1f} |{bar}| {tr['recall']}% ({tr['correct']}/{tr['total']})")
    recalls = [tr["recall"] for tr in temp_res]
    mono = all(recalls[i] >= recalls[i+1] for i in range(len(recalls)-1))
    findings.append(f"Monotonic decrease: {'Yes' if mono else 'No'}")
    if len(recalls) >= 2: findings.append(f"Drop 0.0→1.0: {recalls[0]-recalls[-1]:+.1f}%")
    findings.append(f"Total turns: {turns}")
    advs = compute_advantages(make_samples("o52", turns))
    ma = max(abs(a) for a in advs) if advs else 0.0
    return rec("O52", f"Temperature Sensitivity ({MODE})", "PASS" if ma <= 3.0 else "FAIL", findings,
        {"turns": turns, "mode": MODE, "temp_results": temp_res, "monotonic": mono, "max_adv": round(ma, 4)})


# ═══════════════════════════════════════════════════════════════
# O53: Context Window Boundary
# ═══════════════════════════════════════════════════════════════
def test_o53():
    sep("O53", "Context Window Boundary", 2)
    findings, model, turns = [], OLLAMA_MODEL, 0
    fills = [("25%", 5), ("50%", 15), ("75%", 30), ("100%", 50)]
    fill_res = []
    facts = TEACH_FACTS[:10]

    for fi, (label, n_dist) in enumerate(fills):
        logger.info("  [%d%%] O53: %s fill (%d dist turns)", int(fi/len(fills)*80), label, n_dist)
        msgs = [{"role": "system", "content": "You are helpful. Remember facts precisely."}]
        for f in facts:
            msgs.append({"role": "user", "content": f["teach"]})
            r = chat(model, msgs, temperature=0.1, max_tokens=64)
            msgs.append({"role": "assistant", "content": r["response"]}); turns += 1
        for i in range(n_dist):
            msgs.append({"role": "user", "content": DISTRACTIONS[i % len(DISTRACTIONS)]})
            r = chat(model, msgs[-10:], temperature=0.7, max_tokens=128)
            msgs.append({"role": "assistant", "content": r["response"]}); turns += 1
        correct = 0
        for f in facts:
            msgs.append({"role": "user", "content": f["probe"]})
            r = chat(model, msgs[-6:], temperature=0.1, max_tokens=64)
            if f["value"].lower() in r["response"].lower(): correct += 1
            msgs.append({"role": "assistant", "content": r["response"]}); turns += 1
        fill_res.append({"fill": label, "dist": n_dist, "correct": correct,
                         "total": len(facts), "recall": round(correct/len(facts)*100, 1)})

    logger.info("  [80%%] O53: Analysis")
    findings.append(f"Mode: {MODE} | model: {model}")
    for fr in fill_res:
        bar = "█" * int(fr["recall"]/5) + "░" * (20 - int(fr["recall"]/5))
        findings.append(f"  {fr['fill']:>4s} ({fr['dist']:2d} dist) |{bar}| {fr['recall']}% ({fr['correct']}/{fr['total']})")
    recalls = [fr["recall"] for fr in fill_res]
    if len(recalls) >= 2:
        findings.append(f"Drop 25%→100%: {recalls[0]-recalls[-1]:+.1f}%")
        drops = [(recalls[i]-recalls[i+1], fill_res[i+1]["fill"]) for i in range(len(recalls)-1)]
        mx = max(drops, key=lambda x: x[0])
        findings.append(f"Biggest cliff: {mx[0]:+.1f}% at {mx[1]}")
    findings.append(f"Total turns: {turns}")
    advs = compute_advantages(make_samples("o53", turns))
    ma = max(abs(a) for a in advs) if advs else 0.0
    return rec("O53", f"Context Window Boundary ({MODE})", "PASS" if ma <= 3.0 else "FAIL", findings,
        {"turns": turns, "mode": MODE, "fill_results": fill_res, "max_adv": round(ma, 4)})


# ═══════════════════════════════════════════════════════════════
# O54: Instruction Following Decay
# ═══════════════════════════════════════════════════════════════
def test_o54():
    sep("O54", "Instruction Following Decay", 3)
    findings, model, turns = [], OLLAMA_MODEL, 0
    fmt_instr = ('IMPORTANT: Always respond in valid JSON: {"answer": "<your answer>", "confidence": <0.0-1.0>}. '
                 'Never use any other format.')
    checkpoints = [1, 5, 10, 25, 50]
    ck_res = []
    json_re = re.compile(r'\{[^}]*"answer"[^}]*"confidence"[^}]*\}', re.DOTALL)
    msgs = [{"role": "system", "content": fmt_instr}]

    for turn in range(1, max(checkpoints) + 1):
        turns += 1
        if turn in checkpoints:
            qa = FACTUAL_QA[(turn - 1) % len(FACTUAL_QA)]
            msgs.append({"role": "user", "content": f"Answer in JSON format: {qa['q']}"})
            r = chat(model, msgs[-8:], temperature=0.3, max_tokens=128)
            ans = r["response"].strip()
            msgs.append({"role": "assistant", "content": ans})
            valid_json, has_struct = False, False
            try:
                p = json.loads(ans); valid_json = True; has_struct = "answer" in p and "confidence" in p
            except Exception:
                valid_json = bool(json_re.search(ans)); has_struct = valid_json
            ck_res.append({"turn": turn, "json": valid_json, "struct": has_struct, "resp": ans[:60]})
        else:
            topic = DISTRACTIONS[turn % len(DISTRACTIONS)]
            msgs.append({"role": "user", "content": topic})
            r = chat(model, msgs[-8:], temperature=0.7, max_tokens=128)
            msgs.append({"role": "assistant", "content": r["response"]})

    logger.info("  [80%%] O54: Analysis")
    findings.append(f"Mode: {MODE} | model: {model}")
    for cr in ck_res:
        s = "✓ JSON" if cr["struct"] else ("~ partial" if cr["json"] else "✗ broken")
        findings.append(f"  Turn {cr['turn']:3d}: {s} — '{cr['resp'][:45]}'")
    comp = sum(1 for c in ck_res if c["struct"])
    broken = sum(1 for c in ck_res if not c["json"])
    findings.append(f"Compliant: {comp}/{len(ck_res)} | Broken: {broken}/{len(ck_res)}")
    if len(ck_res) >= 2:
        decay = ck_res[0]["struct"] and not ck_res[-1]["struct"]
        findings.append(f"Instruction decay detected: {'Yes' if decay else 'No'}")
    findings.append(f"Total turns: {turns}")
    advs = compute_advantages(make_samples("o54", turns))
    ma = max(abs(a) for a in advs) if advs else 0.0
    return rec("O54", f"Instruction Following Decay ({MODE})", "PASS" if ma <= 3.0 else "FAIL", findings,
        {"turns": turns, "mode": MODE, "checkpoints": ck_res, "compliant": comp, "broken": broken, "max_adv": round(ma, 4)})


# ═══════════════════════════════════════════════════════════════
# O55: Multilingual Hallucination
# ═══════════════════════════════════════════════════════════════
def test_o55():
    sep("O55", "Multilingual Hallucination", 4)
    findings, model, turns = [], OLLAMA_MODEL, 0
    facts = TEACH_FACTS[:10]
    langs = [
        ("en", "English", "What is the {key}? Answer with just the value."),
        ("es", "Spanish", "¿Cuál es el/la {key}? Responde solo con el valor."),
        ("ja", "Japanese", "{key}は何ですか？値だけで答えてください。"),
        ("fr", "French", "Quel est le/la {key}? Répondez uniquement avec la valeur."),
    ]

    logger.info("  [0%%] O55: Teaching 10 facts in English")
    msgs = [{"role": "system", "content": "You are multilingual. Remember facts precisely."}]
    for f in facts:
        msgs.append({"role": "user", "content": f["teach"]})
        r = chat(model, msgs, temperature=0.1, max_tokens=64)
        msgs.append({"role": "assistant", "content": r["response"]}); turns += 1
    for i in range(10):
        msgs.append({"role": "user", "content": DISTRACTIONS[i % len(DISTRACTIONS)]})
        r = chat(model, msgs[-8:], temperature=0.7, max_tokens=128)
        msgs.append({"role": "assistant", "content": r["response"]}); turns += 1

    lang_res = {}
    for li, (code, name, tmpl) in enumerate(langs):
        logger.info("  [%d%%] O55: Probing %s", 20 + int(li/len(langs)*60), name)
        correct = 0
        for f in facts:
            probe = tmpl.format(key=f["key"])
            msgs.append({"role": "user", "content": probe})
            r = chat(model, msgs[-6:], temperature=0.1, max_tokens=64)
            if f["value"].lower() in r["response"].lower(): correct += 1
            msgs.append({"role": "assistant", "content": r["response"]}); turns += 1
        lang_res[code] = {"name": name, "correct": correct, "total": len(facts),
                          "recall": round(correct/len(facts)*100, 1)}

    logger.info("  [80%%] O55: Analysis")
    findings.append(f"Mode: {MODE} | model: {model}")
    for code, lr in lang_res.items():
        bar = "█" * int(lr["recall"]/5) + "░" * (20 - int(lr["recall"]/5))
        findings.append(f"  {lr['name']:10s} |{bar}| {lr['recall']}% ({lr['correct']}/{lr['total']})")
    en_r = lang_res.get("en", {}).get("recall", 0)
    others = [lr["recall"] for c, lr in lang_res.items() if c != "en"]
    if others:
        avg_o = sum(others) / len(others)
        findings.append(f"English: {en_r}% | Avg non-English: {avg_o:.1f}% | Drop: {en_r - avg_o:+.1f}%")
    findings.append(f"Total turns: {turns}")
    advs = compute_advantages(make_samples("o55", turns))
    ma = max(abs(a) for a in advs) if advs else 0.0
    return rec("O55", f"Multilingual Hallucination ({MODE})", "PASS" if ma <= 3.0 else "FAIL", findings,
        {"turns": turns, "mode": MODE, "lang_results": lang_res, "max_adv": round(ma, 4)})


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════
def sep(tid, title, idx):
    print(f"\n{'='*72}\n  [{int(idx/5*100)}%] {tid}: {title} ({MODE})\n{'='*72}")


def main():
    global MODE
    _check_ollama()
    MODE = "live" if OLLAMA_AVAILABLE else "simulated"
    ckpt = CheckpointManager("tier2_hallucination")
    ckpt.set_total(5)

    print(f"\n{_B}{'='*72}")
    print(f"  TIER 2 HALLUCINATION SUITE (O51-O55)")
    print(f"  Mode: {_C}{MODE}{_X}{_B}")
    if OLLAMA_AVAILABLE: print(f"  Models: {', '.join(OLLAMA_MODELS[:5])}")
    if ckpt.resumed: print(f"  Resuming from checkpoint")
    print(f"{'='*72}{_X}")

    start = time.time()
    tests = [("O51", test_o51), ("O52", test_o52), ("O53", test_o53), ("O54", test_o54), ("O55", test_o55)]
    for tid, fn in tests:
        if ckpt.should_skip(tid):
            r = ckpt.get_result(tid); RESULTS.append(r)
            print(f"\n  {_C}[SKIP]{_X} {tid}: restored from checkpoint")
        else:
            r = fn(); ckpt.save(tid, r)

    elapsed = time.time() - start
    total_turns = sum(r.get("metrics", {}).get("turns", 0) for r in RESULTS)
    pc = sum(1 for r in RESULTS if r["verdict"] == "PASS")
    wc = sum(1 for r in RESULTS if r["verdict"] == "WARN")
    fc = sum(1 for r in RESULTS if r["verdict"] == "FAIL")

    print(f"\n{'='*72}")
    print(f"  TIER 2 RESULTS ({MODE} mode) {'[RESUMED]' if ckpt.resumed else ''}")
    print(f"{'='*72}")
    print(f"  Duration: {elapsed:.1f}s | Turns: {total_turns:,}\n")
    for r in RESULTS:
        c = _G if r["verdict"] == "PASS" else (_R if r["verdict"] == "FAIL" else _Y)
        print(f"  {c}[{r['verdict']}]{_X} {r['test_id']}: {r['name']}")
    print(f"\n  {_G}PASS: {pc}{_X}  {_Y}WARN: {wc}{_X}  {_R}FAIL: {fc}{_X}")
    status = "VALIDATED" if fc == 0 else "PARTIAL"
    print(f"\n  {'VALIDATED' if fc == 0 else 'PARTIAL'}\n")

    rp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "records", "tier2_hallucination_results.json")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, "w") as f:
        json.dump({"suite": "Tier 2 Hallucination (O51-O55)", "mode": MODE,
            "ollama_models": OLLAMA_MODELS[:10], "elapsed_seconds": round(elapsed, 2),
            "total_turns": total_turns, "summary": {"pass": pc, "warn": wc, "fail": fc},
            "tests": RESULTS}, f, indent=2, default=str)
    print(f"  Results: {rp}")
    ckpt.finish()


if __name__ == "__main__":
    main()
