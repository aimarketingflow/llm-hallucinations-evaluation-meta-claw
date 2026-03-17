<div align="center">

<img src="new_logo.png" alt="DragonClaw" width="600">

<br/>

# Просто разговаривайте с вашим агентом, и он будет учиться и *ЭВОЛЮЦИОНИРОВАТЬ*.

<p>Вдохновлено тем, как учится мозг. Мета-обучение и эволюция вашего 🦞 в каждом реальном диалоге. GPU не требуется. Поддерживает Kimi, Qwen, Claude, MiniMax и другие.</p>

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
  <img src="https://img.shields.io/badge/⚡_Полностью_асинхронно-yellow?style=flat&labelColor=555" alt="Fully Async" />
  <img src="https://img.shields.io/badge/☁️_Без_GPU_кластера-blue?style=flat&labelColor=555" alt="No GPU Cluster" />
  <img src="https://img.shields.io/badge/🛠️_Эволюция_навыков-orange?style=flat&labelColor=555" alt="Skill Evolution" />
  <img src="https://img.shields.io/badge/🚀_Развертывание_в_один_клик-green?style=flat&labelColor=555" alt="One-Click Deploy" />
</p>

<br/>

[🇺🇸 English](../README.md) • [🇨🇳 中文](./README_ZH.md) • [🇯🇵 日本語](./README_JA.md) • [🇰🇷 한국어](./README_KO.md) • [🇫🇷 Français](./README_FR.md) • [🇩🇪 Deutsch](./README_DE.md) • [🇪🇸 Español](./README_ES.md) • [🇧🇷 Português](./README_PT.md) • [🇮🇹 Italiano](./README_IT.md) • [🇻🇳 Tiếng Việt](./README_VI.md) • [🇸🇦 العربية](./README_AR.md) • [🇮🇳 हिन्दी](./README_HI.md)

<br/>

[Обзор](#-обзор) • [Быстрый старт](#-быстрый-старт) • [Конфигурация](#️-конфигурация) • [Режим навыков](#-режим-навыков) • [Режим RL](#-режим-rl) • [Режим MadMax](#-режим-madmax-по-умолчанию) • [Цитирование](#-цитирование)

</div>

---

<div align="center">

### Две команды. Это все.
</div>

```bash
dragonclaw setup              # одноразовый мастер настройки
dragonclaw start              # по умолчанию: режим madmax, навыки + плановое RL-обучение
dragonclaw start --mode rl    # RL без планировщика (обучение сразу по заполнении batch)
dragonclaw start --mode skills_only  # только навыки, без RL (Tinker не нужен)
```

<div align="center">
<img src="dragonclaw.gif" alt="DragonClaw demo" width="700">
</div>

---

## 🔥 Новости

- **[13.03.2026]** **v0.3.1** Поддержка бэкенда MinT: RL-обучение теперь работает как с Tinker, так и с MinT. Настраивается через `rl.backend` (auto/tinker/mint).
- **[13.03.2026]** **v0.3** Поддержка непрерывного мета-обучения: медленные RL-обновления запускаются только во время сна, простоя или встреч в Google Calendar. Добавлено разделение на support/query множества для предотвращения загрязнения обновлений модели устаревшими сигналами вознаграждения.
- **[11.03.2026]** **v0.2** Развертывание в один клик через CLI `dragonclaw`. Навыки включены по умолчанию, RL теперь опционален.
- **[09.03.2026]** Выпущен **DragonClaw**. Просто общайтесь с агентом, и он будет эволюционировать автоматически. GPU-развертывание **не требуется**, достаточно подключить **API**.

---

## 🎥 Демонстрация

https://github.com/user-attachments/assets/d86a41a8-4181-4e3a-af0e-dc453a6b8594

---

## 📖 Обзор

**DragonClaw это агент, который мета-обучается и эволюционирует в реальных условиях.**
Просто общайтесь с агентом, как обычно. DragonClaw превращает каждый живой диалог в обучающий сигнал, позволяя агенту непрерывно совершенствоваться через реальное развертывание, а не только через офлайн-обучение.

Под капотом DragonClaw размещает вашу модель за OpenAI-совместимым прокси, который перехватывает взаимодействия из OpenClaw, внедряет релевантные навыки на каждом шаге и мета-обучается на накопленном опыте. После каждой сессии навыки автоматически суммируются; при включенном RL планировщик мета-обучения откладывает обновление весов до окон простоя, чтобы агент не прерывался во время активного использования.

GPU-кластер не требуется. DragonClaw работает с любым OpenAI-совместимым LLM API «из коробки» и использует Tinker-совместимый бэкенд для облачного LoRA-дообучения. [Tinker](https://www.thinkingmachines.ai/tinker/) является путем по умолчанию, а MinT можно подключить через отдельный пакет совместимости при необходимости.

## 🤖 Ключевые возможности

### **Развертывание в один клик**
Настройте один раз с помощью `dragonclaw setup`, затем `dragonclaw start` запускает прокси, внедряет навыки и автоматически подключает OpenClaw. Ручные shell-скрипты не нужны.

### **Три режима работы**

| Режим | По умолчанию | Описание |
|-------|-------------|----------|
| `skills_only` | | Прокси для вашего LLM API. Навыки внедряются, после каждой сессии автоматически суммируются. GPU/Tinker не требуются. |
| `rl` | | Навыки + RL-обучение (GRPO). Обучение запускается сразу при заполнении batch. Опциональный OPD для дистилляции учителя. |
| `madmax` | ✅ | Навыки + RL + интеллектуальный планировщик. Обновление весов RL происходит только во время сна/простоя/встреч. |

### **Полностью асинхронная архитектура**
Обслуживание, моделирование вознаграждений и обучение полностью разделены. Агент продолжает отвечать, пока оценка и оптимизация выполняются параллельно.

---

## 🚀 Быстрый старт

### 1. Установка

```bash
pip install -e .                        # режим skills_only (легковесный)
pip install -e ".[rl]"                  # + поддержка RL-обучения (torch, transformers, tinker)
pip install -e ".[evolve]"              # + эволюция навыков через OpenAI-совместимый LLM
pip install -e ".[scheduler]"           # + интеграция с Google Calendar для планировщика
pip install -e ".[rl,evolve,scheduler]" # рекомендуется для полной конфигурации RL + планировщик
```

Если вы хотите использовать `rl.backend=mint`, установите пакет совместимости MinT отдельно в том же окружении, например [`mindlab-toolkit`](https://github.com/MindLab-Research/mindlab-toolkit). DragonClaw не включает эту зависимость в пакет по умолчанию, чтобы пользователи RL могли явно выбирать между Tinker и MinT.

### 2. Настройка

```bash
dragonclaw setup
```

Интерактивный мастер предложит выбрать LLM-провайдера (Kimi, Qwen, MiniMax или пользовательский), ввести API-ключ и опционально включить RL-обучение.

RL-путь DragonClaw позволяет явно переключаться между `tinker` и `mint`. Рекомендуемое значение по умолчанию: `auto`. При установленном пакете MinT он по-прежнему способен распознать MinT по учетным данным или base URL в стиле Mint.

**Tinker** (по умолчанию):

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

Устаревшие псевдонимы `rl.tinker_api_key` и `rl.tinker_base_url` по-прежнему поддерживаются для обратной совместимости.

### 3. Запуск

```bash
dragonclaw start
```

Это все. DragonClaw запускает прокси, автоматически настраивает OpenClaw и перезапускает шлюз. Откройте OpenClaw и начните диалог: навыки внедряются на каждом шаге, а по завершении сессии автоматически суммируются в новые навыки.

---

## ⚙️ Конфигурация

Файл конфигурации находится в `~/.dragonclaw/config.yaml` и создается командой `dragonclaw setup`.

**Команды CLI:**

```
dragonclaw setup                  # Интерактивный мастер первоначальной настройки
dragonclaw start                  # Запуск DragonClaw (по умолчанию: режим madmax)
dragonclaw start --mode rl        # Принудительно включить режим RL (без планировщика) для этой сессии
dragonclaw start --mode skills_only  # Принудительно включить режим только навыков для этой сессии
dragonclaw stop                   # Остановить работающий экземпляр DragonClaw
dragonclaw status                 # Проверить состояние прокси, режим работы и статус планировщика
dragonclaw config show            # Просмотр текущей конфигурации
dragonclaw config KEY VALUE       # Установить значение конфигурации
```

<details>
<summary><b>Полная справка по конфигурации (нажмите, чтобы развернуть)</b></summary>

```yaml
mode: madmax               # "madmax" | "rl" | "skills_only"

llm:
  provider: kimi            # kimi | qwen | openai | minimax | custom
  model_id: moonshotai/Kimi-K2.5
  api_base: https://api.moonshot.cn/v1
  api_key: sk-...

proxy:
  port: 30000
  api_key: ""              # необязательный bearer-токен для локального прокси DragonClaw

skills:
  enabled: true
  dir: ~/.dragonclaw/skills   # каталог вашей библиотеки навыков
  retrieval_mode: template  # template | embedding
  top_k: 6
  task_specific_top_k: 10   # лимит навыков для конкретных задач (по умолчанию 10)
  auto_evolve: true         # автоматическое суммирование навыков после каждой сессии

rl:
  enabled: false            # установите true для включения RL-обучения
  backend: auto             # "auto" | "tinker" | "mint"
  model: moonshotai/Kimi-K2.5
  api_key: ""
  base_url: ""              # необязательная точка доступа бэкенда, например https://mint.macaron.xin/ для MinT
  tinker_api_key: ""        # устаревший псевдоним для api_key
  tinker_base_url: ""       # устаревший псевдоним для base_url
  prm_url: https://api.openai.com/v1
  prm_model: gpt-5.2
  prm_api_key: ""
  lora_rank: 32
  batch_size: 4
  resume_from_ckpt: ""      # необязательный путь к контрольной точке для возобновления обучения
  evolver_api_base: ""      # оставьте пустым для использования llm.api_base
  evolver_api_key: ""
  evolver_model: gpt-5.2

opd:
  enabled: false            # установите true для включения OPD (дистилляция учителя)
  teacher_url: ""           # base URL модели-учителя (OpenAI-совместимый /v1/completions)
  teacher_model: ""         # имя модели-учителя (например, Qwen/Qwen3-32B)
  teacher_api_key: ""       # API-ключ модели-учителя
  kl_penalty_coef: 1.0      # коэффициент KL-штрафа для OPD

max_context_tokens: 20000   # лимит токенов промпта перед усечением

scheduler:                  # v0.3: планировщик мета-обучения (автоматически включается в режиме madmax)
  enabled: false            # в режиме madmax включается автоматически; для режима rl установите вручную
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

## 💪 Режим навыков

**`dragonclaw start --mode skills_only`**

Самый легкий режим. Не требуется ни GPU, ни RL-бэкенд. DragonClaw размещает ваш LLM за прокси, который внедряет релевантные навыки на каждом шаге, а затем автоматически суммирует новые навыки после каждого диалога.

Навыки представляют собой короткие Markdown-инструкции, хранящиеся в `~/.dragonclaw/skills/` в виде отдельных файлов `SKILL.md`. Библиотека навыков растет автоматически вместе с использованием.

Для предварительной загрузки встроенного банка навыков (более 40 навыков по программированию, безопасности, агентным задачам и др.):

```bash
cp -r memory_data/skills/* ~/.dragonclaw/skills/
```

---

## 🔬 Режим RL

**`dragonclaw start --mode rl`**

Все возможности режима навыков плюс непрерывное RL-дообучение на основе живых диалогов. Каждый шаг диалога токенизируется и отправляется как обучающий пример. Модель-судья (PRM) асинхронно оценивает ответы, а Tinker-совместимый бэкенд (Tinker cloud или MinT) выполняет LoRA-дообучение с горячей заменой весов.

**Tinker** (по умолчанию):

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

Специализированная модель-эволюционер также извлекает новые навыки из неудачных эпизодов и возвращает их в библиотеку навыков.

**Программный rollout** (без TUI OpenClaw): установите `openclaw_env_data_dir` на каталог с JSONL-файлами задач:

```json
{"task_id": "task_1", "instruction": "Register the webhook at https://example.com/hook"}
```

### Дистилляция с политикой на лету (OPD)

OPD является опциональным дополнением к режиму RL. Он дистиллирует большую модель-учителя в модель-ученика на его собственной политике: ученик генерирует ответы как обычно, а учитель предоставляет потокенные логарифмические вероятности для тех же ответов. KL-штраф направляет ученика к распределению учителя.

```bash
dragonclaw config opd.enabled true
dragonclaw config opd.teacher_url http://localhost:8082/v1
dragonclaw config opd.teacher_model Qwen/Qwen3-32B
dragonclaw config opd.kl_penalty_coef 1.0
```

Учитель должен быть развернут за OpenAI-совместимой точкой доступа `/v1/completions` (например, vLLM, SGLang). OPD можно комбинировать с оценкой PRM, оба процесса выполняются асинхронно. См. `examples/run_conversation_opd.py` и `scripts/run_openclaw_tinker_opd.sh`.

---

## 🧠 Режим MadMax (По умолчанию)

**`dragonclaw start`**

Все возможности режима RL плюс планировщик мета-обучения, который откладывает обновление весов до окон неактивности пользователя, чтобы агент не прерывался во время активного использования. Это режим по умолчанию.

Шаг горячей замены весов RL приостанавливает агента на несколько минут. Вместо того чтобы обучаться сразу при заполнении batch (как в режиме RL), MadMax ожидает подходящего окна.

Три условия запускают окно обновления (достаточно любого одного):

- **Часы сна**: настраиваемое время начала/окончания (например, 23:00 до 07:00)
- **Неактивность клавиатуры**: срабатывает после N минут простоя
- **События Google Calendar**: обнаруживает встречи, чтобы обновления выполнялись, пока вы отсутствуете

```bash
dragonclaw config scheduler.sleep_start "23:00"
dragonclaw config scheduler.sleep_end   "07:00"
dragonclaw config scheduler.idle_threshold_minutes 30

# Необязательно: интеграция с Google Calendar
pip install -e ".[scheduler]"
dragonclaw config scheduler.calendar.enabled true
dragonclaw config scheduler.calendar.credentials_path ~/.dragonclaw/client_secrets.json
```

Если пользователь возвращается во время обновления, частичный batch сохраняется и возобновляется в следующем окне.

Каждый `ConversationSample` помечается версией `skill_generation`. Когда эволюция навыков увеличивает поколение, RL-буфер очищается, и для градиентных обновлений используются только пост-эволюционные примеры (разделение MAML support/query множеств).

---

## 📚 Цитирование

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

## 🙏 Благодарности

DragonClaw построен на основе следующих проектов с открытым исходным кодом:

- [OpenClaw](https://openclaw.ai), основной фреймворк агента.
- [SkillRL](https://github.com/aiming-lab/SkillRL), наш фреймворк RL с расширением навыков.
- [Tinker](https://www.thinkingmachines.ai/tinker/), используется для онлайн RL-обучения.
- [MinT](https://github.com/MindLab-Research/mindlab-toolkit), альтернативный бэкенд для онлайн RL-обучения.
- [OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL), вдохновение для нашего дизайна RL.
- [awesome-openclaw-skills](https://github.com/VoltAgent/awesome-openclaw-skills), основа для нашего банка навыков.

---

## 📄 Лицензия

Этот проект распространяется под лицензией [MIT](LICENSE).
