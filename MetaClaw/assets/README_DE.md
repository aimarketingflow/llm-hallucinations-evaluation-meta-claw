<div align="center">

<img src="new_logo.png" alt="DragonClaw" width="600">

<br/>

# Sprich einfach mit deinem Agenten — er lernt und *ENTWICKELT* sich weiter.

<p>Inspiriert davon, wie das Gehirn lernt. Meta-lernen und entwickeln Sie Ihren 🦞 aus jedem Gespräch. Keine GPU nötig. Kompatibel mit Kimi, Qwen, Claude, MiniMax und mehr.</p>

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
    <img src="tinker.jpg" width="48" height="48" alt="Tinker" />
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
  <img src="https://img.shields.io/badge/⚡_Vollständig_Async-yellow?style=flat&labelColor=555" alt="Fully Async" />
  <img src="https://img.shields.io/badge/☁️_Kein_GPU--Cluster-blue?style=flat&labelColor=555" alt="No GPU Cluster" />
  <img src="https://img.shields.io/badge/🛠️_Skill--Evolution-orange?style=flat&labelColor=555" alt="Skill Evolution" />
  <img src="https://img.shields.io/badge/🚀_Ein--Klick--Deployment-green?style=flat&labelColor=555" alt="One-Click Deploy" />
</p>

<br/>

[🇺🇸 English](../README.md) • [🇨🇳 中文](./README_ZH.md) • [🇯🇵 日本語](./README_JA.md) • [🇰🇷 한국어](./README_KO.md) • [🇫🇷 Français](./README_FR.md) • [🇪🇸 Español](./README_ES.md) • [🇧🇷 Português](./README_PT.md) • [🇷🇺 Русский](./README_RU.md) • [🇮🇹 Italiano](./README_IT.md) • [🇻🇳 Tiếng Việt](./README_VI.md) • [🇦🇪 العربية](./README_AR.md) • [🇮🇳 हिन्दी](./README_HI.md)

<br/>

[Übersicht](#-übersicht) • [Schnellstart](#-schnellstart) • [Konfiguration](#️-konfiguration) • [Skills-Modus](#-skills-modus) • [RL-Modus](#-rl-modus) • [MadMax-Modus](#-madmax-modus-standard) • [Zitierung](#-zitierung)

</div>

---

<div align="center">

### Zwei Befehle. Das ist alles.
</div>

```bash
dragonclaw setup              # Einmaliger Konfigurationsassistent
dragonclaw start              # Standard: MadMax-Modus, Skills + geplantes RL-Training
dragonclaw start --mode rl    # RL ohne Scheduler (trainiert sofort bei vollem Batch)
dragonclaw start --mode skills_only  # Nur Skills, kein RL (kein Tinker nötig)
```

<div align="center">
<img src="dragonclaw.gif" alt="DragonClaw demo" width="700">
</div>

---

## 🔥 Neuigkeiten

- **[13.03.2026]** **v0.3.1** MinT-Backend-Unterstützung: RL-Training funktioniert jetzt mit Tinker und MinT. Konfigurierbar über `rl.backend` (auto/tinker/mint).
- **[13.03.2026]** **v0.3** Kontinuierliche Meta-Learning-Unterstützung: RL-Gewichtsupdates laufen nur noch während Schlafenszeiten, Leerlaufphasen oder Google-Calendar-Meetings. Support/Query-Set-Trennung hinzugefügt, um veraltete Belohnungssignale von Modell-Updates fernzuhalten.
- **[11.03.2026]** **v0.2** Ein-Klick-Deployment über `dragonclaw` CLI. Skills standardmäßig aktiviert, RL jetzt optional.
- **[09.03.2026]** **DragonClaw** veröffentlicht. Sprich einfach mit deinem Agenten und lass ihn automatisch weiterentwickeln. **Kein** GPU-Deployment erforderlich; einfach an die **API** anschließen.

---

## 🎥 Demo

https://github.com/user-attachments/assets/d86a41a8-4181-4e3a-af0e-dc453a6b8594

---

## 📖 Übersicht

**DragonClaw ist ein Agent, der in realen Einsatzszenarien meta-lernt und sich weiterentwickelt.**
Sprich einfach wie gewohnt mit deinem Agenten. DragonClaw verwandelt jedes Live-Gespräch in ein Lernsignal und ermöglicht dem Agenten, sich durch den realen Einsatz kontinuierlich zu verbessern, statt nur auf Offline-Training zu setzen.

Unter der Haube kapselt es dein Modell hinter einem OpenAI-kompatiblen Proxy, fängt Interaktionen über OpenClaw ab, injiziert relevante Skills bei jedem Schritt und meta-lernt aus den gesammelten Erfahrungen. Nach jeder Session werden Skills automatisch zusammengefasst; mit aktiviertem RL verschiebt ein Meta-Learning-Scheduler die Gewichtsaktualisierungen in inaktive Zeitfenster, damit der Agent während der aktiven Nutzung nie unterbrochen wird.

Kein GPU-Cluster nötig. DragonClaw funktioniert mit jeder OpenAI-kompatiblen LLM-API und nutzt ein Tinker-kompatibles Backend für Cloud-basiertes LoRA-Training. [Tinker](https://www.thinkingmachines.ai/tinker/) ist der Standard-Referenzpfad; bei Bedarf kann MinT über ein separates Kompatibilitätspaket aktiviert werden.

## 🤖 Hauptfunktionen

### **Ein-Klick-Deployment**
Einmal mit `dragonclaw setup` konfigurieren, dann startet `dragonclaw start` den Proxy, injiziert Skills und verbindet OpenClaw automatisch. Keine manuellen Shell-Skripte nötig.

### **Drei Betriebsmodi**

| Modus | Standard | Funktion |
|-------|---------|----------|
| `skills_only` | | Proxy für deine LLM-API. Skills werden injiziert und nach jeder Session automatisch zusammengefasst. Kein GPU/Tinker erforderlich. |
| `rl` | | Skills + RL-Training (GRPO). Trainiert sofort, wenn ein Batch voll ist. Optional OPD für Lehrer-Destillation. |
| `madmax` | ✅ | Skills + RL + Smart-Scheduler. RL-Gewichtsupdates laufen nur in Schlaf-/Leerlauf-/Meeting-Fenstern. |

### **Asynchron by Design**
Serving, Reward Modeling und Training sind vollständig entkoppelt. Der Agent antwortet weiterhin, während Bewertung und Optimierung parallel laufen.

---

## 🚀 Schnellstart

### 1. Installation

```bash
pip install -e .                        # skills_only-Modus (leichtgewichtig)
pip install -e ".[rl]"                  # + RL-Trainingsunterstützung (torch, transformers, tinker)
pip install -e ".[evolve]"              # + Skill-Evolution via OpenAI-kompatibler LLM
pip install -e ".[scheduler]"           # + Google Calendar Integration für Scheduler
pip install -e ".[rl,evolve,scheduler]" # empfohlen: vollständiges RL + Scheduler-Setup
```

Wenn du `rl.backend=mint` verwenden willst, installiere das MinT-Kompatibilitätspaket separat in derselben Umgebung, zum Beispiel [`mindlab-toolkit`](https://github.com/MindLab-Research/mindlab-toolkit). DragonClaw hält diese Abhängigkeit absichtlich aus dem Standardpaket heraus, damit RL-Nutzer Tinker oder MinT explizit wählen können.

### 2. Konfiguration

```bash
dragonclaw setup
```

Der interaktive Assistent führt dich durch die Auswahl des LLM-Anbieters (Kimi, Qwen, MiniMax oder benutzerdefiniert), API-Schlüssel und optionale RL-Aktivierung.

Der RL-Pfad von DragonClaw kann explizit zwischen `tinker` und `mint` wechseln. `auto` ist die empfohlene Voreinstellung und kann MinT weiterhin aus Mint-ähnlichen Credentials oder Base-URLs ableiten, wenn das MinT-Paket installiert ist.

**Tinker** (Standard):

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

Die Legacy-Aliase `rl.tinker_api_key` und `rl.tinker_base_url` werden weiterhin aus Kompatibilitätsgründen akzeptiert.

### 3. Start

```bash
dragonclaw start
```

Das war's. DragonClaw startet den Proxy, konfiguriert OpenClaw automatisch und startet das Gateway neu. Öffne OpenClaw und beginne zu chatten. Skills werden bei jedem Schritt injiziert, und die Session wird automatisch zu neuen Skills zusammengefasst, wenn du fertig bist.

---

## ⚙️ Konfiguration

Die Konfiguration liegt in `~/.dragonclaw/config.yaml`, erstellt durch `dragonclaw setup`.

**CLI-Befehle:**

```
dragonclaw setup                  # Interaktiver Erstkonfigurations-Assistent
dragonclaw start                  # DragonClaw starten (Standard: MadMax-Modus)
dragonclaw start --mode rl        # RL-Modus für diese Session erzwingen (ohne Scheduler)
dragonclaw start --mode skills_only  # Nur-Skills-Modus für diese Session erzwingen
dragonclaw stop                   # Laufende DragonClaw-Instanz stoppen
dragonclaw status                 # Proxy-Status, laufenden Modus und Scheduler prüfen
dragonclaw config show            # Aktuelle Konfiguration anzeigen
dragonclaw config KEY VALUE       # Konfigurationswert setzen
```

<details>
<summary><b>Vollständige Konfigurationsreferenz (zum Aufklappen klicken)</b></summary>

```yaml
mode: madmax               # "madmax" | "rl" | "skills_only"

llm:
  provider: kimi            # kimi | qwen | openai | minimax | custom
  model_id: moonshotai/Kimi-K2.5
  api_base: https://api.moonshot.cn/v1
  api_key: sk-...

proxy:
  port: 30000
  api_key: ""              # optionales Bearer-Token für den lokalen DragonClaw-Proxy

skills:
  enabled: true
  dir: ~/.dragonclaw/skills   # deine Skill-Bibliothek
  retrieval_mode: template  # template | embedding
  top_k: 6
  task_specific_top_k: 10   # Obergrenze für aufgabenspezifische Skills (Standard 10)
  auto_evolve: true         # Skills nach jeder Session automatisch zusammenfassen

rl:
  enabled: false            # auf true setzen, um RL-Training zu aktivieren
  backend: auto             # "auto" | "tinker" | "mint"
  model: moonshotai/Kimi-K2.5
  api_key: ""
  base_url: ""              # optionaler Backend-Endpunkt, z.B. https://mint.macaron.xin/ für MinT
  tinker_api_key: ""        # Legacy-Alias für api_key
  tinker_base_url: ""       # Legacy-Alias für base_url
  prm_url: https://api.openai.com/v1
  prm_model: gpt-5.2
  prm_api_key: ""
  lora_rank: 32
  batch_size: 4
  resume_from_ckpt: ""      # optionaler Checkpoint-Pfad zum Fortsetzen des Trainings
  evolver_api_base: ""      # leer lassen, um llm.api_base wiederzuverwenden
  evolver_api_key: ""
  evolver_model: gpt-5.2

opd:
  enabled: false            # auf true setzen, um OPD (Lehrer-Destillation) zu aktivieren
  teacher_url: ""           # Basis-URL des Lehrermodells (OpenAI-kompatibles /v1/completions)
  teacher_model: ""         # Name des Lehrermodells (z.B. Qwen/Qwen3-32B)
  teacher_api_key: ""       # API-Schlüssel des Lehrermodells
  kl_penalty_coef: 1.0      # KL-Strafkoeffizient für OPD

max_context_tokens: 20000   # Token-Obergrenze vor Kürzung

scheduler:                  # v0.3: Meta-Learning-Scheduler (auto-aktiviert im MadMax-Modus)
  enabled: false            # MadMax-Modus aktiviert automatisch; für RL-Modus manuell setzen
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

## 💪 Skills-Modus

**`dragonclaw start --mode skills_only`**

Der leichteste Modus. Kein GPU, kein RL-Backend nötig. DragonClaw kapselt dein LLM hinter einem Proxy, der bei jedem Schritt relevante Skills injiziert und nach jedem Gespräch automatisch neue Skills zusammenfasst.

Skills sind kurze Markdown-Anweisungen in `~/.dragonclaw/skills/` als einzelne `SKILL.md`-Dateien. Die Bibliothek wächst automatisch mit der Nutzung.

Um die eingebaute Skill-Bank vorzuladen (40+ Skills für Coding, Security, agentische Aufgaben usw.):

```bash
cp -r memory_data/skills/* ~/.dragonclaw/skills/
```

---

## 🔬 RL-Modus

**`dragonclaw start --mode rl`**

Alles aus dem Skills-Modus, plus kontinuierliches RL-Fine-Tuning aus Live-Gesprächen. Jeder Gesprächszug wird tokenisiert und als Trainingsbeispiel eingereicht. Ein Richter-LLM (PRM) bewertet Antworten asynchron, und ein Tinker-kompatibles Backend (Tinker Cloud oder MinT) führt LoRA-Fine-Tuning mit Hot-Swap-Gewichten durch.

**Tinker** (Standard):

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

Ein dediziertes Evolver-LLM extrahiert außerdem neue Skills aus fehlgeschlagenen Episoden und speist sie zurück in die Skill-Bibliothek.

**Programmatisches Rollout** (keine OpenClaw TUI nötig): `openclaw_env_data_dir` auf ein Verzeichnis mit JSONL-Aufgabendateien setzen:

```json
{"task_id": "task_1", "instruction": "Register the webhook at https://example.com/hook"}
```

### On-Policy Distillation (OPD)

OPD ist ein optionales Add-on für den RL-Modus. Es destilliert ein größeres Lehrermodell on-policy in den Schüler: Der Schüler generiert Antworten wie gewohnt, und der Lehrer liefert token-weise Log-Wahrscheinlichkeiten für dieselben Antworten. Eine KL-Strafe lenkt den Schüler zur Verteilung des Lehrers hin.

```bash
dragonclaw config opd.enabled true
dragonclaw config opd.teacher_url http://localhost:8082/v1
dragonclaw config opd.teacher_model Qwen/Qwen3-32B
dragonclaw config opd.kl_penalty_coef 1.0
```

Das Lehrermodell muss hinter einem OpenAI-kompatiblen `/v1/completions`-Endpunkt (z.B. vLLM, SGLang) betrieben werden. OPD kann mit PRM-Bewertung kombiniert werden, beide laufen asynchron. Siehe `examples/run_conversation_opd.py` und `scripts/run_openclaw_tinker_opd.sh`.

---

## 🧠 MadMax-Modus (Standard)

**`dragonclaw start`**

Alles aus dem RL-Modus, plus ein Meta-Learning-Scheduler, der Gewichtsupdates in Benutzer-Inaktivitätsfenster verschiebt, damit der Agent während der aktiven Nutzung nie unterbrochen wird. Dies ist der Standardmodus.

Der RL-Gewichts-Hot-Swap-Schritt pausiert den Agenten für mehrere Minuten. Anstatt sofort zu trainieren, wenn ein Batch voll ist (wie im RL-Modus), wartet MadMax auf ein geeignetes Zeitfenster.

Drei Bedingungen lösen ein Update-Fenster aus (eine reicht aus):

- **Schlafenszeiten**: konfigurierbarer Start-/Endzeitpunkt (z.B. 23:00 bis 07:00)
- **Tastatur-Inaktivität**: wird nach N Minuten Leerlauf ausgelöst
- **Google-Calendar-Events**: erkennt Meetings, sodass Updates laufen können, während du unterwegs bist

```bash
dragonclaw config scheduler.sleep_start "23:00"
dragonclaw config scheduler.sleep_end   "07:00"
dragonclaw config scheduler.idle_threshold_minutes 30

# Optional: Google Calendar Integration
pip install -e ".[scheduler]"
dragonclaw config scheduler.calendar.enabled true
dragonclaw config scheduler.calendar.credentials_path ~/.dragonclaw/client_secrets.json
```

Wenn der Benutzer während eines Updates zurückkehrt, wird der partielle Batch gespeichert und im nächsten Fenster fortgesetzt.

Jedes `ConversationSample` wird mit einer `skill_generation`-Version getaggt. Wenn die Skill-Evolution die Generation erhöht, wird der RL-Buffer geleert, sodass nur Post-Evolutions-Samples für Gradient-Updates verwendet werden (MAML Support/Query-Set-Trennung).

---

## 📚 Zitierung

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

## 🙏 Danksagungen

DragonClaw baut auf folgenden Open-Source-Projekten auf:

- [OpenClaw](https://openclaw.ai) - das zentrale Agent-Framework.
- [SkillRL](https://github.com/aiming-lab/SkillRL) - unser skill-erweitertes RL-Framework.
- [Tinker](https://www.thinkingmachines.ai/tinker/) - für Online-RL-Training verwendet.
- [MinT](https://github.com/MindLab-Research/mindlab-toolkit) - alternatives Backend für Online-RL-Training.
- [OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL) - Inspiration für unser RL-Design.
- [awesome-openclaw-skills](https://github.com/VoltAgent/awesome-openclaw-skills) - stellt die Grundlage für unsere Skill-Bank bereit.

---

## 📄 Lizenz

Dieses Projekt ist unter der [MIT-Lizenz](LICENSE) lizenziert.
