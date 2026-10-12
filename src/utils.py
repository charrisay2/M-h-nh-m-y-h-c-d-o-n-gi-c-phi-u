"""Tiện ích dùng chung: seed, logging, IO, nạp config."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int = 42) -> None:
    """Cố định seed để kết quả reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str = "pipeline", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def load_config(path: str | Path | None = None) -> dict:
    """Nạp config.yaml và chuẩn hóa danh sách mã (hỗ trợ nhiều dòng / chuỗi CSV)."""
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for market, mcfg in cfg.get("markets", {}).items():
        flat: list[str] = []
        for item in mcfg.get("tickers", []):
            if isinstance(item, str):
                flat.extend(t.strip() for t in item.split(",") if t.strip())
            else:
                flat.append(str(item))
        # Loại trùng, giữ nguyên thứ tự
        seen, tickers = set(), []
        for t in flat:
            if t not in seen:
                seen.add(t)
                tickers.append(t)
        mcfg["tickers"] = tickers
    return cfg


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: str | Path) -> None:
    p = ensure_dir(Path(path).parent) / Path(path).name
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def fmt_pct(x: float, nd: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{nd}f}%"


def fmt_num(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"
