from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from src.main import main as src_main

    return src_main()


if __name__ == "__main__":
    raise SystemExit(main())

