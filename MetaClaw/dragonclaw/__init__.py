"""
DragonClaw — OpenClaw skill injection and RL training, one-click deployment.

Integrates:
  - OpenClaw online dialogue data collection (FastAPI proxy)
  - Skill injection and auto-summarization (skills_only mode)
  - Tinker-compatible cloud LoRA RL training (rl mode, optional)

Quick start:
    dragonclaw setup    # configure LLM, skills, RL toggle
    dragonclaw start    # one-click launch
"""

from .config import DragonClawConfig
from .config_store import ConfigStore
from .api_server import DragonClawAPIServer
from .rollout import AsyncRolloutWorker
from .prm_scorer import PRMScorer
from .skill_manager import SkillManager
from .skill_evolver import SkillEvolver
from .launcher import DragonClawLauncher

# RL-only imports (guarded to avoid hard dep on torch/backend SDKs in skills_only mode)
try:
    from .data_formatter import ConversationSample, batch_to_datums, compute_advantages
    from .trainer import DragonClawTrainer
except ImportError:
    pass

__all__ = [
    "DragonClawConfig",
    "ConfigStore",
    "DragonClawAPIServer",
    "AsyncRolloutWorker",
    "PRMScorer",
    "SkillManager",
    "SkillEvolver",
    "DragonClawLauncher",
]
