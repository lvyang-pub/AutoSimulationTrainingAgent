"""Download the Unitree Go2 MJCF model from mujoco_menagerie.

Network note: `git clone` against github.com is blocked/reset on this machine,
but `raw.githubusercontent.com` and `api.github.com` work over HTTPS. So we fetch
files individually via urllib instead of cloning.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

REPO = "google-deepmind/mujoco_menagerie"
REF = "main"
SUBDIR = "unitree_go2"
HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "go2")

RAW = f"https://raw.githubusercontent.com/{REPO}/{REF}/{SUBDIR}/{{name}}"
API = f"https://api.github.com/repos/{REPO}/contents/{SUBDIR}/{{path}}"


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "autosim-fetch"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _list_dir(path: str) -> list[dict]:
    """List a directory in the repo via the GitHub contents API."""
    url = API.format(path=path) if path else API.format(path="")
    if not path:
        url = f"https://api.github.com/repos/{REPO}/contents/{SUBDIR}"
    return json.loads(_get(url))


def _download(rel: str, dest: str) -> None:
    """Download one file (rel path is relative to SUBDIR)."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  -> {rel}")
    data = _get(RAW.format(name=rel))
    with open(dest, "wb") as fh:
        fh.write(data)


def fetch(force: bool = False) -> str:
    """Fetch the model into DEST. Returns path to scene.xml. Idempotent."""
    scene = os.path.join(DEST, "scene.xml")
    if os.path.exists(scene) and not force:
        print(f"[fetch_go2] already cached at {DEST} (use force=True to refresh)")
        return scene

    os.makedirs(DEST, exist_ok=True)
    print(f"[fetch_go2] fetching {SUBDIR} -> {DEST}")
    # top-level files (xml + README etc.)
    for entry in _list_dir(""):
        if entry["type"] == "file" and entry["name"].endswith((".xml", ".yaml", ".json")):
            _download(entry["name"], os.path.join(DEST, entry["name"]))
    # assets/ meshes
    try:
        for entry in _list_dir("assets"):
            if entry["type"] == "file":
                _download(f"assets/{entry['name']}", os.path.join(DEST, "assets", entry["name"]))
    except Exception as exc:  # noqa: BLE001 - assets dir optional in some revisions
        print(f"[fetch_go2] warning: could not list assets/: {exc}")

    if not os.path.exists(scene):
        raise RuntimeError("download finished but scene.xml is missing")
    print("[fetch_go2] done")
    return scene


if __name__ == "__main__":
    path = fetch(force="--force" in sys.argv)
    # verify it loads
    import mujoco

    m = mujoco.MjModel.from_xml_path(path)
    print(f"[fetch_go2] OK scene.xml loads: nq={m.nq} nu={m.nu} nbody={m.nbody}")
