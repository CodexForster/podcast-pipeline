from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict


def ensure_structure(project_root: Path, path_config: Dict[str, str]) -> Dict[str, Path]:
    resolved = {}
    for key, rel in path_config.items():
        path = (project_root / rel).resolve()
        path.mkdir(parents=True, exist_ok=True)
        resolved[key] = path
    return resolved


def copy_if_needed(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

