#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""README.md word budget: at most 250 words of prose.

Counted, as a reader meets them: headings, paragraphs, list items, table cells, link text, image alt text and inline
code. Not counted: fenced code blocks (the commands to paste), the generated badge block, HTML comments and tags, and
URLs. A word is a whitespace-separated token holding at least one letter or digit, so `|`, `-` and `→` alone are free.
Measurements belong in docs/PERFORMANCE.md, which the badges read; the README walks a user from install to a model.

  python .github/checks/readme_words.py [--limit 250] [--file README.md] [--show]

Exits 1 when the file holds more words than the limit. Standard library only.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIMIT = 250
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def prose(text: str) -> str:
    """The text a reader reads as prose, with code blocks, the badge block, comments, tags and URLs removed."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"^<!-- badges:start.*?^<!-- badges:end -->[ \t]*$", "", text, flags=re.S | re.M)
    kept, fence = [], None
    for line in text.split("\n"):
        run = FENCE.match(line)
        if fence is None:
            if run:
                fence = run.group(1)
            else:
                kept.append(line)
        elif run and run.group(1)[0] == fence[0] and len(run.group(1)) >= len(fence) \
                and not line.strip()[len(run.group(1)):].strip():
            fence = None
    text = "\n".join(kept)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"^\s*\[[^\]]+\]:\s*\S+.*$", " ", text, flags=re.M)  # reference-style link definitions
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links and images keep their text
    text = re.sub(r"<(?:https?://|mailto:)[^>]*>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"https?://\S+", " ", text)


def words(text: str) -> list:
    return [token for token in prose(text).split() if re.search(r"[A-Za-z0-9]", token)]


def main(argv: list | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description="Count README.md's prose words against a limit.")
    parser.add_argument("--limit", type=int, default=LIMIT, help=f"most words allowed (default {LIMIT})")
    parser.add_argument("--file", type=Path, default=ROOT / "README.md", help="Markdown file to count")
    parser.add_argument("--show", action="store_true", help="print the counted words")
    args = parser.parse_args(argv)

    counted = words(args.file.read_bytes().decode("utf-8"))
    if args.show:
        print(" ".join(counted))
    over = len(counted) > args.limit
    print(f"{'FAIL' if over else 'ok  '} {args.file.name}: {len(counted)} words of prose, limit {args.limit}")
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
