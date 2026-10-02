"""Tee a script's stdout/stderr to a file under log/.

Usage:
    from scriptlog import start
    log_path, _fh = start("eval")      # -> log/eval_YYYYmmdd_HHMMSS.log

Keeps console output intact while also writing a transcript to disk, so every
run lands in one predictable place instead of relying on shell redirection.
"""
from __future__ import annotations

import os
import sys
import time


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self._streams:
            st.flush()


def start(name: str, log_dir: str = "log") -> tuple[str, "object"]:
    """Begin logging; returns (path, handle). Call again to switch files."""
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    fh = open(path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    return path, fh
