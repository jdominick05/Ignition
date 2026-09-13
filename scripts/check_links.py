#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
scripts/check_links.py
Verify every markdown link in the repo resolves, and check repository markdown files.
Exit code 1 if anything is broken, so it can gate a commit.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.M)
EXTERNAL = re.compile(r"^(https?|mailto|ftp):", re.I)


def slug(text: str) -> str:
    """GitHub's heading -> anchor transform."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return text.replace(" ", "-")


def anchors(path: Path) -> set:
    try:
        body = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    seen, out = {}, set()
    for _, text in HEADING.findall(body):
        base = slug(text)
        n = seen.get(base, 0)
        out.add(base if n == 0 else f"{base}-{n}")
        seen[base] = n + 1
    return out


def md_files(root: Path):
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "*.md", "*/*.md", "*/*/*.md"],
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()
        tracked = sorted(set(root / line for line in out if line.strip()))
        if tracked:
            return tracked
    except Exception:
        pass
    return sorted(p for p in root.glob("**/*.md") if ".git" not in p.parts and "venv" not in p.parts)


def check_file(path: Path, all_anchors: dict):
    errs = []
    try:
        body = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return [f"{path}: cannot read ({e})"]

    for raw in LINK.findall(body):
        if EXTERNAL.match(raw) or raw.startswith("file:///"):
            continue
        target, _, frag = raw.partition("#")
        if not target:
            if frag and frag not in all_anchors.get(path, set()):
                errs.append(f"{path}: broken same-file anchor #{frag}")
            continue

        resolved = (path.parent / target).resolve()
        if not resolved.exists():
            errs.append(f"{path}: link to {target} does not exist (resolved {resolved})")
            continue

        if frag and resolved.suffix.lower() == ".md":
            target_anchors = all_anchors.get(resolved)
            if target_anchors is None:
                target_anchors = anchors(resolved)
                all_anchors[resolved] = target_anchors
            if frag not in target_anchors:
                errs.append(f"{path}: anchor #{frag} not found in {target}")

    return errs


def main():
    files = md_files(ROOT)
    all_anchors = {f: anchors(f) for f in files}
    all_errs = []

    for f in files:
        all_errs.extend(check_file(f, all_anchors))

    if all_errs:
        print("Link check failed:")
        for e in all_errs:
            print(f"  {e}")
        sys.exit(1)

    print(f"OK checked {len(files)} markdown files: all links resolve.")


if __name__ == "__main__":
    main()
