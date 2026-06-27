"""
HSCodeComp Agent — Agent 工厂模块

提供核心逻辑：
  - 配置加载（多模型预设）
  - 日志初始化
  - System Prompt 构建
  - AIAgent 实例创建
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from _resources import PROJECT_DIR, get_config_path, get_prompt_path, get_logs_dir


# ═══════════════════════════════════════════════════════════════
# Environment
# ═══════════════════════════════════════════════════════════════

def load_env() -> None:
    """Load .env file and set HERMES_HOME."""
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_DIR / ".env")
    except ImportError:
        pass
    os.environ.setdefault("HERMES_HOME", str(Path.home() / ".hermes"))


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

def load_config(model_preset: Optional[str] = None) -> dict:
    """Load config.yaml with environment variable expansion.

    Args:
        model_preset: 模型预设名称 (如 'qwen3.7-max', 'glm-5.2')
                      为 None 时使用 config.yaml 中的 default_model
    """
    import yaml

    config_path = get_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Expand $ENV_VAR references
    def expand(val):
        if isinstance(val, str) and val.startswith("$"):
            env_key = val[1:]
            return os.environ.get(env_key, "")
        return val

    # Determine which model preset to use
    preset_name = model_preset or os.environ.get("MODEL_PRESET", "") or config.get("default_model", "qwen3.7-max")
    models = config.get("models", {})

    if preset_name not in models:
        available = list(models.keys())
        raise ValueError(f"未知模型预设: '{preset_name}'。可用: {available}")

    model_config = models[preset_name]

    # Expand env vars in model config
    config["model"] = model_config.get("model", preset_name)
    config["base_url"] = expand(model_config.get("base_url", ""))
    config["api_key"] = expand(model_config.get("api_key", ""))
    config["enable_thinking"] = model_config.get("enable_thinking", False)
    config["context_length"] = model_config.get("context_length", 131072)
    config["model_preset"] = preset_name

    return config


# ═══════════════════════════════════════════════════════════════
# Logging Setup
# ═══════════════════════════════════════════════════════════════

def setup_logging(config: dict) -> logging.Logger:
    """Configure structured logging."""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    log_dir = get_logs_dir()

    # Session log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"session_{timestamp}.log"

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "[%(asctime)s] [%(name)-20s] [%(levelname)-5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    root_logger.addHandler(file_handler)

    # Console handler
    if log_config.get("console_rich", True):
        try:
            from rich.logging import RichHandler
            console_handler = RichHandler(
                level=log_level, show_path=False, markup=True, rich_tracebacks=True
            )
            root_logger.addHandler(console_handler)
        except ImportError:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(file_fmt)
            root_logger.addHandler(console_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(file_fmt)
        root_logger.addHandler(console_handler)

    logger = logging.getLogger("hscode")
    logger.info("📝 日志文件: %s", log_file)
    return logger


# ═══════════════════════════════════════════════════════════════
# System Prompt Building
# ═══════════════════════════════════════════════════════════════

def build_system_prompt(gri_rules: bool = True) -> str:
    """构建 HSCode 分类专用的 system prompt.

    Args:
        gri_rules: 是否追加 GRI 六条归类总规则（默认 True）
    """
    prompt_path = get_prompt_path()
    if prompt_path.exists():
        prompt = prompt_path.read_text(encoding="utf-8")
    else:
        # Fallback: 简化版
        prompt = (
            "你是一位专业的商品编码智能归类助手。"
            "根据商品信息按照美国海关 HSCode 标准进行分类，"
            "输出完整的 10 位 HSCode。使用 LaTeX 格式 \\boxed{} 输出最终答案。"
        )

    if gri_rules:
        gri_path = PROJECT_DIR / "prompts" / "gri_rules.md"
        if gri_path.exists():
            prompt += "\n\n" + gri_path.read_text(encoding="utf-8")

    return prompt


# ═══════════════════════════════════════════════════════════════
# Agent Creation
# ═══════════════════════════════════════════════════════════════

def create_agent(config: dict, logger: logging.Logger,
                 reasoning_callback=None, thinking_callback=None,
                 gri_rules: bool = True):
    """Create and configure the Hermes AIAgent instance for HSCode classification.

    Args:
        config: 完整配置字典（来自 load_config()）
        logger: 日志 logger
        reasoning_callback: 可选，reasoning 流式回调
        thinking_callback: 可选，thinking 流式回调
        gri_rules: 是否在 system prompt 中注入 GRI 规则（默认 True）
    """

    # ── 覆盖 hermes 框架硬编码的 DEFAULT_AGENT_IDENTITY ──
    try:
        import agent.prompt_builder as _prompt_builder
        import agent.system_prompt as _system_prompt
        _prompt_builder.DEFAULT_AGENT_IDENTITY = ""
        _system_prompt.DEFAULT_AGENT_IDENTITY = ""
        logger.debug("✅ 已清除 hermes DEFAULT_AGENT_IDENTITY（含 system_prompt 引用）")
    except ImportError:
        logger.warning("⚠️  无法导入 agent.prompt_builder / agent.system_prompt")

    # Build system prompt
    system_message = build_system_prompt(gri_rules=gri_rules)
    logger.info("✅ System prompt 已加载 (%d chars, GRI=%s)", len(system_message), gri_rules)

    # Agent configuration
    agent_config = config.get("agent", {})
    enabled_toolsets = agent_config.get("enabled_toolsets", ["web"])

    # Reasoning config
    reasoning_cfg = agent_config.get("reasoning")
    if reasoning_cfg and not isinstance(reasoning_cfg, dict):
        reasoning_cfg = None

    # Request overrides (for DashScope enable_thinking)
    request_overrides = {}
    if config.get("enable_thinking"):
        request_overrides.setdefault("extra_body", {})["enable_thinking"] = True

    # Create AIAgent
    from run_agent import AIAgent

    # ── 注册本地 HTS 树查询工具 ──
    from custom_hts_tools import register_hts_tools
    register_hts_tools()
    logger.info("✅ 已注册本地 HTS 树查询工具 (toolset='hts')")

    # ── 根据模型名自动判断 api_mode ──
    model_name = config.get("model", "qwen3.7-max")
    api_mode = None
    if "claude" in model_name.lower():
        api_mode = "anthropic_messages"
    elif "gpt" in model_name.lower():
        api_mode = "chat_completions"

    agent = AIAgent(
        base_url=config.get("base_url", ""),
        api_key=config.get("api_key", ""),
        model=model_name,
        api_mode=api_mode,
        max_iterations=agent_config.get("max_iterations", 30),
        tool_delay=agent_config.get("tool_delay", 0.1),
        enabled_toolsets=enabled_toolsets,
        ephemeral_system_prompt=system_message,
        verbose_logging=agent_config.get("verbose", False),
        quiet_mode=True,
        load_soul_identity=False,
        skip_context_files=True,
        skip_memory=True,
        reasoning_config=reasoning_cfg,
        reasoning_callback=reasoning_callback,
        thinking_callback=thinking_callback,
        request_overrides=request_overrides or None,
    )

    # Context length override
    ctx_len = config.get("context_length")
    if ctx_len is not None:
        try:
            agent._config_context_length = int(ctx_len)
            logger.info("📏 context_length=%d", int(ctx_len))
        except (TypeError, ValueError):
            pass

    logger.info(
        "🤖 HSCode Agent 已初始化 (model=%s, preset=%s, toolsets=%s, "
        "reasoning=%s, enable_thinking=%s)",
        config.get("model"), config.get("model_preset"),
        enabled_toolsets,
        bool(reasoning_cfg and reasoning_cfg.get("enabled", True)),
        bool(config.get("enable_thinking")),
    )

    return agent
