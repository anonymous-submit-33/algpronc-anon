"""Run logging: one directory per run under $ALGPRONC_OUT_DIR, JSONL per task."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

_CONSOLE_CONFIGURED = False


def get_logger(name: str = "algpronc") -> logging.Logger:
    global _CONSOLE_CONFIGURED
    logger = logging.getLogger(name)
    if not _CONSOLE_CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
        root = logging.getLogger("algpronc")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _CONSOLE_CONFIGURED = True
    return logger


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "item") and getattr(obj, "ndim", 1) == 0:
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


class RunLogger:
    """Writes `<out_dir>/<run_id>/{config.yaml,tasks.jsonl,summary.json}`."""

    def __init__(self, run_id: Any, out_dir: str | os.PathLike | None = None) -> None:
        if hasattr(run_id, "run_id"):  
            cfg = run_id
            out_dir = out_dir or cfg.resolved_out_dir()
            run_id = cfg.run_id
        base = Path(out_dir or os.environ.get("ALGPRONC_OUT_DIR", "./runs"))
        self.run_id = run_id
        self.dir = base / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tasks_path = self.dir / "tasks.jsonl"
        self.config_path = self.dir / "config.yaml"
        self.summary_path = self.dir / "summary.json"
        self.tasks_path.write_text("")
        self.log = get_logger("algpronc.run")

    def dump_config(self, cfg_dict: dict[str, Any]) -> None:
        with open(self.config_path, "w") as fh:
            yaml.safe_dump(_jsonable(cfg_dict), fh, sort_keys=False)

    def log_task(self, row: dict[str, Any]) -> None:
        with open(self.tasks_path, "a") as fh:
            fh.write(json.dumps(_jsonable(row)) + "\n")

    def write_json(self, filename: str, obj: Any) -> Path:
        path = self.dir / filename
        with open(path, "w") as fh:
            json.dump(_jsonable(obj), fh, indent=2)
        return path

    def write_summary(self, summary: dict[str, Any]) -> None:
        with open(self.summary_path, "w") as fh:
            json.dump(_jsonable(summary), fh, indent=2)

    def info(self, msg: str) -> None:
        self.log.info(msg)
