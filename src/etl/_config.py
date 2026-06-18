"""Shared loader for the ETL pipeline configuration (``configs/etl.yaml``).

Centralizes the data directory locations so each pipeline step reads them
from a single place instead of taking command-line arguments. Directory
paths in the YAML are written relative to the project root and resolved to
absolute ``Path`` objects here.
"""

from pathlib import Path

import yaml

# 專案根目錄：src/etl/_config.py 往上三層
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = PROJECT_ROOT / "configs" / "etl.yaml"


def load_config() -> dict:
    """Loads ``configs/etl.yaml`` with directory paths resolved to absolute Paths.

    Returns:
        The parsed config dict. Each value under the ``dirs`` mapping is
        converted from a project-root-relative string into an absolute
        ``Path``.
    """
    config = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    config["dirs"] = {name: PROJECT_ROOT / rel for name, rel in config["dirs"].items()}
    return config
