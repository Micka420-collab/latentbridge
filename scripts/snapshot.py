"""Project restore-points for the self-improvement loop.

Every modification gets a snapshot you can roll back to if the change turns out
worse or useless. Git-backed, so it works for Hermes, Arbor, or you.

    python scripts/snapshot.py save "try VQ bridge"   # before editing
    python scripts/snapshot.py list                    # see restore points
    python scripts/snapshot.py restore <id>            # revert files to a snapshot

Always run on an improve/<topic> branch (see AGENTS.md), never on main.
Restore is non-destructive to history: it reverts the working files to the
snapshot but keeps every snapshot, so you can move back and forth.
"""
from __future__ import annotations

import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(*args, capture=True):
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=capture, text=True)


def branch():
    return git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


def save(label):
    if branch() in ("main", "master"):
        print("REFUSED: you are on main. First: git checkout -b improve/<topic>")
        sys.exit(2)
    git("add", "-A")
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode
    if staged == 0:
        print("nothing to snapshot (no changes since last snapshot)")
        return
    git("commit", "-m", f"snapshot: {label}")
    sha = git("rev-parse", "--short", "HEAD").stdout.strip()
    print(f"snapshot saved: {sha}  ({label})  on branch {branch()}")
    print(f"  restore later with:  python scripts/snapshot.py restore {sha}")


def list_snaps():
    out = git("log", "--oneline", "--grep", "snapshot:", "-n", "30").stdout.strip()
    print("restore points (newest first):")
    print(out or "  (none yet)")


def restore(ref):
    git("restore", "--source", ref, "--worktree", "--staged", "--", ".", capture=False)
    print(f"restored working files to {ref}. History is intact (snapshots kept).")
    print("re-measure with:  python scripts/measure.py --config configs/life.yaml")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "save":
        save(sys.argv[2] if len(sys.argv) > 2 else "wip")
    elif cmd in ("list", "ls"):
        list_snaps()
    elif cmd == "restore":
        if len(sys.argv) < 3:
            print("usage: snapshot.py restore <id>")
            sys.exit(2)
        restore(sys.argv[2])
    else:
        print(f"unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
