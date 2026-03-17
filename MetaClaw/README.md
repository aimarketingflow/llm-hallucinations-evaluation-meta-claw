<div align="center">

<img src="assets/new_logo.png" alt="DragonClaw" width="600">

<br/>

# Just talk to your agent — it learns and *EVOLVES*.

<p>Inspired by how the brain learns. Meta-learn and evolve your 🦞 from every conversation in the wild. No GPU required. Works with Kimi, Qwen, Claude, MiniMax, and more.</p>

⚡ Supported LLM Providers & Platforms

<table>
<tr>
<td align="center" width="100">
  <a href="https://kimi.ai">
    <img src="https://github.com/MoonshotAI.png?size=200" width="48" height="48" alt="Kimi" />
  </a><br/>
  <sub><a href="https://kimi.ai"><b>Kimi</b></a></sub>
</td>
<td align="center" width="100">
  <a href="https://qwen.ai">
    <img src="https://github.com/QwenLM.png?size=200" width="48" height="48" alt="Qwen" />
  </a><br/>
  <sub><a href="https://qwen.ai"><b>Qwen</b></a></sub>
</td>
<td align="center" width="100">
  <a href="https://www.anthropic.com/claude">
    <img src="https://cdn.simpleicons.org/claude/D97757" width="48" height="48" alt="Claude" />
  </a><br/>
  <sub><a href="https://www.anthropic.com/claude"><b>Claude</b></a></sub>
</td>
<td align="center" width="100">
  <a href="https://www.minimax.io">
    <img src="https://github.com/minimax-ai.png?size=200" width="48" height="48" alt="MiniMax" />
  </a><br/>
  <sub><a href="https://www.minimax.io"><b>MiniMax</b></a></sub>
</td>
<td align="center" width="100">
  <a href="https://openai.com">
    <img src="https://github.com/openai.png?size=200" width="48" height="48" alt="OpenAI" />
  </a><br/>
  <sub><a href="https://openai.com"><b>OpenAI</b></a></sub>
</td>
<td align="center" width="100">
  <a href="https://gemini.google.com">
    <img src="https://cdn.simpleicons.org/googlegemini/8E75B2" width="48" height="48" alt="Gemini" />
  </a><br/>
  <sub><a href="https://gemini.google.com"><b>Gemini</b></a></sub>
</td>
<td align="center" width="100">
  <sub><b>+ Much<br/>More</b></sub>
</td>
</tr>
</table>

🧬 RL Training Backends for Weight Evolution

<table>
<tr>
<td align="center" width="100">
  <a href="https://www.thinkingmachines.ai/tinker/">
    <img src="assets/tinker.jpg" width="48" height="48" alt="Tinker" />
  </a><br/>
  <sub><a href="https://www.thinkingmachines.ai/tinker/"><b>Tinker</b></a></sub>
</td>
<td align="center" width="100">
  <a href="https://github.com/MindLab-Research/mindlab-toolkit">
    <img src="https://github.com/MindLab-Research.png?size=200" width="48" height="48" alt="MinT" />
  </a><br/>
  <sub><a href="https://github.com/MindLab-Research/mindlab-toolkit"><b>MinT</b></a></sub>
</td>
<td align="center" width="100">
  <sub><b>More<br/>Coming</b></sub>
</td>
</tr>
</table>

<p>
  <a href="https://github.com/aiming-lab/DragonClaw"><img src="https://img.shields.io/badge/github-DragonClaw-181717?style=flat&labelColor=555&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat&labelColor=555" alt="License MIT"></a>
  <img src="https://img.shields.io/badge/⚡_Fully_Async-yellow?style=flat&labelColor=555" alt="Fully Async" />
  <img src="https://img.shields.io/badge/☁️_No_GPU_Cluster-blue?style=flat&labelColor=555" alt="No GPU Cluster" />
  <img src="https://img.shields.io/badge/🛠️_Skill_Evolution-orange?style=flat&labelColor=555" alt="Skill Evolution" />
  <img src="https://img.shields.io/badge/🚀_One--Click_Deploy-green?style=flat&labelColor=555" alt="One-Click Deploy" />
</p>

[🇨🇳 中文](./assets/README_ZH.md) • [🇯🇵 日本語](./assets/README_JA.md) • [🇰🇷 한국어](./assets/README_KO.md) • [🇫🇷 Français](./assets/README_FR.md) • [🇩🇪 Deutsch](./assets/README_DE.md) • [🇪🇸 Español](./assets/README_ES.md) • [🇧🇷 Português](./assets/README_PT.md) • [🇷🇺 Русский](./assets/README_RU.md) • [🇮🇹 Italiano](./assets/README_IT.md) • [🇻🇳 Tiếng Việt](./assets/README_VI.md) • [🇦🇪 العربية](./assets/README_AR.md) • [🇮🇳 हिन्दी](./assets/README_HI.md)

<br/>

[Overview](#-overview) • [Quick Start](#-quick-start) • [Configuration](#️-configuration) • [Skills Mode](#-skills-mode) • [RL Mode](#-rl-mode) • [MadMax Mode](#-madmax-mode-default) • [Citation](#-citation)

</div>

---

<div align="center">

### Two commands. That's it.
</div>

```bash
dragonclaw setup              # one-time config wizard
dragonclaw start              # default: madmax mode — skills + scheduled RL training
dragonclaw start --mode rl    # RL without scheduler (trains immediately on full batch)
dragonclaw start --mode skills_only  # skills only, no RL (no Tinker needed)
```

<div align="center">
<img src="assets/dragonclaw.gif" alt="DragonClaw demo" width="700">
</div>

---

## 🔥 News

- **[03/13/2026]** **v0.3.1** — MinT backend support: RL training now works with both Tinker and MinT. Configurable via `rl.backend` (auto/tinker/mint).
- **[03/13/2026]** **v0.3** — Continual meta-learning support: slow RL updates now only run during sleep hours, idle time, or Google Calendar meetings. Added support/query set separation to prevent stale reward signals from polluting model updates.
- **[03/11/2026]** **v0.2** — One-click deployment via `dragonclaw` CLI. Skills enabled by default, RL is now opt-in.
- **[03/09/2026]** We release **DragonClaw** — Just talk to your agent and let it evolve automatically. **NO** GPU deployment required; just plug into the **API**.

---

## 🎥 Demo

https://github.com/user-attachments/assets/d86a41a8-4181-4e3a-af0e-dc453a6b8594

---

## � Origin: MetaClaw → DragonClaw

DragonClaw is the result of a deliberate research progression: **ERLA → MetaClaw → DragonClaw**.

We first designed [ERLA](https://aimarketingflow.com/erla/) (Ephemeral Recursive Learning Agents), a privacy-preserving architecture where agents learn, distill knowledge, and self-destruct. When we recognized [MetaClaw](https://github.com/meta-claw/meta-claw) (v0.3) as a variant expansion of the direction ERLA was exploring — persistent conversation memory, meta-learning via replay, agent self-improvement — we used it as a testbed. We applied the same adversarial methodology we'd developed for ERLA: inject poison, trace propagation, measure defense gaps. MetaClaw had none. Rather than just documenting the vulnerabilities, we upgraded MetaClaw in place — adding defenses, building session chaining, and running 160 adversarial tests to prove the expanded framework was more agile and more secure than the original. The result is DragonClaw.

**What we inherited from MetaClaw:**
- Meta-learning RL pipeline (GRPO, skill injection, conversation replay)
- OpenAI-compatible proxy architecture
- OpenClaw environment integration
- Skills library and auto-summarization

**What DragonClaw adds (tested with 160 adversarial tests across 22 suites):**
- **3-tier defense stack** — FactVerifier (Tier 1), InputSanitizer (Tier 2), OutputFilter (Tier 3)
- **Defense-gated conversation memory** — retrieved facts treated as untrusted, verified before injection
- **Auto-spawn session chaining** — token budget monitoring, session summarization, seamless handoff
- **Disk-persistent memory index** — conversation recall survives across sessions and restarts
- **Adversarial test coverage** — poison propagation, cross-session recall, defense evasion, training loop corruption

The original MetaClaw can be referenced at its upstream repo for comparison. DragonClaw's competitive intelligence report (`docs/competitive-intelligence-report.md`) documents how our architecture compares to MemGPT/Letta, Zep, Mem0, and other memory systems.

---

## �📖 Overview

**DragonClaw is an agent that meta-learns and evolves in the wild.**
Just talk to your agent as you normally would — DragonClaw turns every live conversation into a learning signal, enabling the agent to continuously improve through real-world deployment rather than offline training alone.

Under the hood, it places your model behind an OpenAI-compatible proxy that intercepts interactions from OpenClaw, injects relevant skills at each turn, and meta-learns from accumulated experience. Skills are summarized automatically after each session; with RL enabled, a meta-learning scheduler defers weight updates to idle windows so the agent is never interrupted during active use.

No GPU cluster required. DragonClaw works with any OpenAI-compatible LLM API out of the box, and uses a Tinker-compatible backend for cloud-based LoRA training. [Tinker](https://www.thinkingmachines.ai/tinker/) is the default reference path, and MinT can be enabled through a separate compatibility package when needed.

## 🤖 Key Features

### **One-click deployment**
Configure once with `dragonclaw setup`, then `dragonclaw start` brings up the proxy, injects skills, and wires OpenClaw automatically. No manual shell scripts needed.

### **Three operating modes**

| Mode | Default | What it does |
|------|---------|--------------|
| `skills_only` | | Proxy your LLM API. Skills injected and auto-summarized after each session. No GPU/Tinker required. |
| `rl` | | Skills + RL training (GRPO). Trains immediately when a batch is full. Optional OPD for teacher distillation. |
| `madmax` | ✅ | Skills + RL + smart scheduler. RL weight updates only run during sleep/idle/meeting windows. |

### **Asynchronous by design**
Serving, reward modeling, and training are fully decoupled. The agent continues responding while scoring and optimization run in parallel.

---

## 🚀 Quick Start

### 1. Install

```bash
pip install -e .                        # skills_only mode (lightweight)
pip install -e ".[rl]"                  # + RL training support (torch, transformers, tinker)
pip install -e ".[evolve]"              # + skill evolution via OpenAI-compatible LLM
pip install -e ".[scheduler]"           # + Google Calendar integration for scheduler
pip install -e ".[rl,evolve,scheduler]" # recommended for full RL + scheduler setup
```

If you want to run `rl.backend=mint`, install the MinT compatibility package separately in the same environment, for example [`mindlab-toolkit`](https://github.com/MindLab-Research/mindlab-toolkit). DragonClaw keeps that dependency out of the default package so RL users can choose Tinker or MinT explicitly.

### 2. Configure

```bash
dragonclaw setup
```

The interactive wizard will ask you to choose your LLM provider (Kimi, Qwen, MiniMax, or custom), enter your API key, and optionally enable RL training.

DragonClaw's RL path can switch explicitly between `tinker` and `mint`. `auto` is the recommended default and will still infer MinT from Mint-like credentials or base URLs when the MinT package is installed.

**Tinker**:

```bash
dragonclaw config rl.backend tinker
dragonclaw config rl.api_key sk-...
dragonclaw config rl.model moonshotai/Kimi-K2.5
```

**MinT**:

```bash
dragonclaw config rl.backend mint
dragonclaw config rl.api_key sk-mint-...
dragonclaw config rl.base_url https://mint.macaron.xin/
dragonclaw config rl.model Qwen/Qwen3-4B-Instruct-2507
```

Legacy aliases `rl.tinker_api_key` and `rl.tinker_base_url` are still accepted for backward compatibility.

### 3. Start

```bash
dragonclaw start
```

That's it. DragonClaw starts the proxy, automatically configures OpenClaw to use it, and restarts the gateway. Open OpenClaw and start chatting — skills are injected at every turn, and the session is automatically summarized into new skills when you're done.

---

## ⚙️ Configuration

Configuration lives in `~/.dragonclaw/config.yaml`, created by `dragonclaw setup`.

**CLI commands:**

```
dragonclaw setup                  # Interactive first-time configuration wizard
dragonclaw start                  # Start DragonClaw (default: madmax mode)
dragonclaw start --mode rl        # Force RL mode (no scheduler) for this session
dragonclaw start --mode skills_only  # Force skills-only mode for this session
dragonclaw stop                   # Stop a running DragonClaw instance
dragonclaw status                 # Check proxy health, running mode, and scheduler state
dragonclaw config show            # View current configuration
dragonclaw config KEY VALUE       # Set a config value
```

<details>
<summary><b>Full config reference (click to expand)</b></summary>

```yaml
mode: madmax               # "madmax" | "rl" | "skills_only"

llm:
  provider: kimi            # kimi | qwen | openai | minimax | custom
  model_id: moonshotai/Kimi-K2.5
  api_base: https://api.moonshot.cn/v1
  api_key: sk-...

proxy:
  port: 30000
  api_key: ""              # optional bearer token for the local DragonClaw proxy

skills:
  enabled: true
  dir: ~/.dragonclaw/skills   # your skill library
  retrieval_mode: template  # template | embedding
  top_k: 6
  task_specific_top_k: 10   # cap task-specific skills (default 10)
  auto_evolve: true         # auto-summarize skills after each session

rl:
  enabled: false            # set to true to enable RL training
  backend: auto             # "auto" | "tinker" | "mint"
  model: moonshotai/Kimi-K2.5
  api_key: ""
  base_url: ""              # optional backend endpoint, e.g. https://mint.macaron.xin/ for MinT
  tinker_api_key: ""        # legacy alias for api_key
  tinker_base_url: ""       # legacy alias for base_url
  prm_url: https://api.openai.com/v1
  prm_model: gpt-5.2
  prm_api_key: ""
  lora_rank: 32
  batch_size: 4
  resume_from_ckpt: ""      # optional checkpoint path to resume training
  evolver_api_base: ""      # leave empty to reuse llm.api_base
  evolver_api_key: ""
  evolver_model: gpt-5.2

opd:
  enabled: false            # set to true to enable OPD (teacher distillation)
  teacher_url: ""           # teacher model base URL (OpenAI-compatible /v1/completions)
  teacher_model: ""         # teacher model name (e.g., Qwen/Qwen3-32B)
  teacher_api_key: ""       # teacher model API key
  kl_penalty_coef: 1.0      # KL penalty coefficient for OPD

max_context_tokens: 20000   # prompt token cap before truncation

scheduler:                  # v0.3: meta-learning scheduler (auto-enabled in madmax mode)
  enabled: false            # madmax mode enables this automatically; set manually for rl mode
  sleep_start: "23:00"
  sleep_end: "07:00"
  idle_threshold_minutes: 30
  min_window_minutes: 15
  calendar:
    enabled: false
    credentials_path: ""
    token_path: ""
```

</details>

---

## 💪 Skills Mode

**`dragonclaw start --mode skills_only`**

The lightest mode. No GPU, no RL backend needed. DragonClaw places your LLM behind a proxy that injects relevant skills at every turn, then auto-summarizes new skills after each conversation.

Skills are short Markdown instructions stored in `~/.dragonclaw/skills/` as individual `SKILL.md` files. The library grows automatically with your usage.

To pre-load the built-in skill bank (40+ skills across coding, security, agentic tasks, etc.):

```bash
cp -r memory_data/skills/* ~/.dragonclaw/skills/
```

---

## 🔬 RL Mode

**`dragonclaw start --mode rl`**

Everything in Skills Mode, plus continuous RL fine-tuning from live conversations. Each conversation turn is tokenized and submitted as a training sample. A judge LLM (PRM) scores responses asynchronously, and a Tinker-compatible backend (Tinker cloud or MinT) runs LoRA fine-tuning with hot-swapped weights.

**Tinker**:

```bash
dragonclaw config rl.backend tinker
dragonclaw config rl.api_key sk-...
dragonclaw config rl.model moonshotai/Kimi-K2.5
dragonclaw config rl.prm_url https://api.openai.com/v1
dragonclaw config rl.prm_api_key sk-...
dragonclaw start --mode rl
```

**MinT**:

```bash
dragonclaw config rl.backend mint
dragonclaw config rl.api_key sk-mint-...
dragonclaw config rl.base_url https://mint.macaron.xin/
dragonclaw config rl.model Qwen/Qwen3-4B-Instruct-2507
dragonclaw config rl.prm_url https://api.openai.com/v1
dragonclaw config rl.prm_api_key sk-...
dragonclaw start --mode rl
```

A dedicated evolver LLM also extracts new skills from failed episodes, feeding them back into the skill library.

**Programmatic rollout** (no OpenClaw TUI needed): set `openclaw_env_data_dir` to a directory of JSONL task files:

```json
{"task_id": "task_1", "instruction": "Register the webhook at https://example.com/hook"}
```

### On-Policy Distillation (OPD)

OPD is an optional add-on for RL Mode. It distills a larger teacher model into the student on-policy: the student generates responses as usual, and the teacher provides per-token log-probabilities on those same responses. A KL penalty steers the student toward the teacher's distribution.

```bash
dragonclaw config opd.enabled true
dragonclaw config opd.teacher_url http://localhost:8082/v1
dragonclaw config opd.teacher_model Qwen/Qwen3-32B
dragonclaw config opd.kl_penalty_coef 1.0
```

The teacher must be served behind an OpenAI-compatible `/v1/completions` endpoint (e.g., vLLM, SGLang). OPD can be combined with PRM scoring, both run asynchronously. See `examples/run_conversation_opd.py` and `scripts/run_openclaw_tinker_opd.sh`.

---

## 🧠 MadMax Mode (Default)

**`dragonclaw start`**

Everything in RL Mode, plus a meta-learning scheduler that defers weight updates to user-inactive windows so the agent is never interrupted during active use. This is the default mode.

The RL weight hot-swap step pauses the agent for several minutes. Instead of training immediately when a batch is full (like RL Mode does), MadMax waits for an appropriate window.

Three conditions trigger an update window (any one is sufficient):

- **Sleep hours**: configurable start/end time (e.g., 23:00 to 07:00)
- **Keyboard inactivity**: triggers after N minutes of idle time
- **Google Calendar events**: detects meetings so updates can run while you're away

```bash
dragonclaw config scheduler.sleep_start "23:00"
dragonclaw config scheduler.sleep_end   "07:00"
dragonclaw config scheduler.idle_threshold_minutes 30

# Optional: Google Calendar integration
pip install -e ".[scheduler]"
dragonclaw config scheduler.calendar.enabled true
dragonclaw config scheduler.calendar.credentials_path ~/.dragonclaw/client_secrets.json
```

If the user returns mid-update, the partial batch is saved and resumed at the next window.

Each `ConversationSample` is tagged with a `skill_generation` version. When skill evolution bumps the generation, the RL buffer is flushed so only post-evolution samples are used for gradient updates (MAML support/query set separation).

---

## 📚 Citation

```bibtex
@misc{xia2026dragonclaw,
  author       = {Xia, Peng and Chen, Jianwen and Yang, Xinyu and Tu, Haoqin and Han, Siwei and Qiu, Shi and Zheng, Zeyu and Xie, Cihang and Yao, Huaxiu},
  title        = {DragonClaw: Just Talk --- An Agent That Meta-Learns and Evolves in the Wild},
  year         = {2026},
  organization = {GitHub},
  url          = {https://github.com/aiming-lab/DragonClaw},
}
```

---

## 🙏 Acknowledgements

DragonClaw builds on top of the following open-source projects:

- [OpenClaw](https://openclaw.ai) – the core agent framework.
- [SkillRL](https://github.com/aiming-lab/SkillRL) – our skill-augmented RL framework.
- [Tinker](https://www.thinkingmachines.ai/tinker/) – used for online RL training.
- [MinT](https://github.com/MindLab-Research/mindlab-toolkit) – alternative backend for online RL training.
- [OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL) – inspiration for our RL design.
- [awesome-openclaw-skills](https://github.com/VoltAgent/awesome-openclaw-skills) – provides the foundation for our skill bank.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
