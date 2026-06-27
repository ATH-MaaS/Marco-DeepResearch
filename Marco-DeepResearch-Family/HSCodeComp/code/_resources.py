"""
HSCodeComp Agent — 资源定位模块

提供统一的资源文件路径解析。
"""

from __future__ import annotations

from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# Project Root Resolution
# ═══════════════════════════════════════════════════════════════

PROJECT_DIR: Path = Path(__file__).resolve().parent

# 数据集路径
DATA_DIR: Path = PROJECT_DIR / "data"
TEST_DATA_PATH: Path = DATA_DIR / "test_data.jsonl"


def get_project_dir() -> Path:
    return PROJECT_DIR


def get_config_path() -> Path:
    return PROJECT_DIR / "config.yaml"


def get_prompt_path() -> Path:
    return PROJECT_DIR / "prompts" / "hscode_system.md"


def get_output_dir() -> Path:
    d = PROJECT_DIR / "output"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_logs_dir() -> Path:
    d = PROJECT_DIR / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_test_data_path() -> Path:
    """返回 HSCodeComp 测试数据路径"""
    return TEST_DATA_PATH
