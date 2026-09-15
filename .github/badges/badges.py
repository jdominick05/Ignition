#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""README badges whose numbers img.shields.io reads live from README.md.

The badges are listed in badges.toml next to this file. A number badge names the README section its number comes
from, an RE2 pattern and a replacement; img.shields.io runs the pattern over README.md on GitHub each time the badge
is fetched, so a push that changes a number in the README changes its badge. Nothing here commits or publishes.

  check [--require-re2]  the badge block in README.md matches badges.toml, and every pattern matches exactly once,
                         inside its section, with the same groups under Python's re and RE2 (google-re2)
  write                  regenerate the block between the badges markers in README.md, then check
  live --ref COMMIT      fetch every number badge from img.shields.io against README.md at COMMIT and compare it
                         with what check computes from that file

Standard library only; RE2 is checked when google-re2 is importable.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    sys.exit("badges.py needs Python 3.11 or newer (tomllib)")

try:
    import re2  # google-re2: the regex engine img.shields.io runs
except ImportError:
    re2 = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = HERE / "badges.toml"
START = "<!-- badges:start: generated from .github/badges/badges.toml by .github/badges/badges.py -->"
END = "<!-- badges:end -->"
SHIELDS = "https://img.shields.io"
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
GROUP = re.compile(r"\$(\d+)")
KINDS = {"message": ("label", "color"), "image": (), "search": ("label", "color", "section", "replace")}


def load_spec(path: Path = SPEC) -> dict:
    spec = tomllib.loads(path.read_text(encoding="utf-8"))
    for key in ("repository", "branch", "readme"):
        if not isinstance(spec.get(key), str) or not spec[key]:
            raise ValueError(f"{path.name}: missing {key}")
    for n, badge in enumerate(spec.get("badge", []), 1):
        kinds = [kind for kind in KINDS if kind in badge]
        if len(kinds) != 1:
            raise ValueError(f"{path.name}: badge {n} needs exactly one of message, image or search")
        for key in ("alt",) + KINDS[kinds[0]]:
            if not isinstance(badge.get(key), str) or not badge[key]:
                raise ValueError(f"{path.name}: badge {n} needs {key}")
        if re.search(r"[\[\]]", badge["alt"]):
            raise ValueError(f"{path.name}: badge {n}'s alt text may not contain brackets")
    return spec


def read_text(path: Path) -> str:
    """The file's text with LF line endings, whatever the checkout wrote."""
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def number_badges(spec: dict) -> list:
    return [badge for badge in spec["badge"] if "search" in badge]


def raw_url(spec: dict, ref: str | None = None) -> str:
    return f"https://raw.githubusercontent.com/{spec['repository']}/{ref or spec['branch']}/{spec['readme']}"


def _static_part(text: str) -> str:
    return urllib.parse.quote(text.replace("-", "--").replace("_", "__"), safe="")


def static_url(label: str, message: str, color: str) -> str:
    return f"{SHIELDS}/badge/{_static_part(label)}-{_static_part(message)}-{color}"


def regex_url(spec: dict, badge: dict, ref: str | None = None) -> str:
    query = {"url": raw_url(spec, ref), "search": badge["search"], "replace": badge["replace"],
             "label": badge["label"], "color": badge["color"]}
    return f"{SHIELDS}/badge/dynamic/regex?" + urllib.parse.urlencode(query, safe="", quote_via=urllib.parse.quote)


def slug(title: str) -> str:
    """GitHub's anchor for a heading, computed as scripts/check_links.py computes it."""
    title = re.sub(r"<[^>]+>", "", title)
    title = re.sub(r"`([^`]*)`", r"\1", title)
    title = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", title)
    return re.sub(r"[^\w\s-]", "", title.strip().lower()).replace(" ", "-")


def badge_markdown(spec: dict, badge: dict) -> str:
    if "message" in badge:
        image, link = static_url(badge["label"], badge["message"], badge["color"]), badge.get("link")
    elif "image" in badge:
        image, link = badge["image"], badge.get("link")
    else:
        image, link = regex_url(spec, badge), badge.get("link", "#" + slug(badge["section"]))
    markdown = f"![{badge['alt']}]({image})"
    return f"[{markdown}]({link})" if link else markdown


def render_block(spec: dict) -> str:
    rows: dict = {}
    for badge in spec["badge"]:
        rows.setdefault(badge.get("row", 1), []).append(badge_markdown(spec, badge))
    lines = [START]
    for n, row in enumerate(sorted(rows)):
        lines += ([""] if n else []) + rows[row]
    return "\n".join(lines + [END])


def current_block(text: str) -> str | None:
    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if line.startswith("<!-- badges:start")]
    ends = [i for i, line in enumerate(lines) if line.strip() == END]
    if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
        return None
    return "\n".join(lines[starts[0]:ends[0] + 1])


def headings(text: str) -> list:
    """(level, title, offset) of every ATX heading outside fenced code blocks."""
    found, fence, offset = [], None, 0
    for line in text.split("\n"):
        run = FENCE.match(line)
        if fence is None and run:
            fence = run.group(1)
        elif fence is not None:
            if run and run.group(1)[0] == fence[0] and len(run.group(1)) >= len(fence) \
                    and not line.strip()[len(run.group(1)):].strip():
                fence = None
        else:
            heading = HEADING.match(line)
            if heading:
                found.append((len(heading.group(1)), heading.group(2), offset))
        offset += len(line) + 1
    return found


def section_span(text: str, title: str) -> tuple[int, int] | None:
    """Offsets of the section under the one heading titled `title`, up to the next heading at its level or above."""
    found = headings(text)
    hits = [i for i, (_, heading, _) in enumerate(found) if heading == title]
    if len(hits) != 1:
        return None
    level, _, start = found[hits[0]]
    end = next((offset for depth, _, offset in found[hits[0] + 1:] if depth <= level), len(text))
    return start, end


@dataclass
class Result:
    label: str
    message: str | None
    problems: list = field(default_factory=list)


def evaluate(badge: dict, text: str, require_re2: bool = False) -> Result:
    """What the badge renders from `text`, or why it would not render a number from its section."""
    label, search, replace = badge["label"], badge["search"], badge["replace"]
    problems = [f"{key} is not ASCII" for key in ("label", "search", "replace") if not badge[key].isascii()]
    if re.search(r"\$(?!\d)", replace):
        problems.append("replace may only use $1, $2, ...: img.shields.io gives other $ forms their JavaScript meaning")
    span = section_span(text, badge["section"])
    if span is None:
        problems.append(f"section {badge['section']!r} is not exactly one heading in the README")
    engines = [("re", re)]
    if re2 is not None:
        engines.append(("re2", re2))
    elif require_re2:
        problems.append("google-re2 is not installed, so RE2 was not checked (pip install google-re2)")
    matches = {}
    for name, engine in engines:
        try:
            matches[name] = list(engine.finditer(search, text))
        except Exception as exc:  # noqa: BLE001 - either engine's pattern error
            problems.append(f"{name} rejects the pattern: {exc}")
            continue
        if len(matches[name]) != 1:
            problems.append(f"{name} finds {len(matches[name])} matches; a badge needs exactly 1")
    if problems:
        return Result(label, None, problems)

    match = matches["re"][0]
    groups = match.groups()
    if "re2" in matches and tuple(matches["re2"][0].groups()) != groups:
        problems.append(f"re and re2 capture different groups: {groups} against {tuple(matches['re2'][0].groups())}")
    if not span[0] <= match.start() < match.end() <= span[1]:
        problems.append(f"the match runs outside section {badge['section']!r}")
    refs = [int(n) for n in GROUP.findall(replace)]
    if not refs:
        problems.append("replace uses no group, so the badge would not show a number from the README")
    for n in sorted({n for n in refs if not 1 <= n <= len(groups)}):
        problems.append(f"replace uses ${n} but the pattern has {len(groups)} groups")
    if problems:
        return Result(label, None, problems)

    message = GROUP.sub(lambda ref: groups[int(ref.group(1)) - 1] or "", replace)
    if not re.search(r"[0-9]", message):
        return Result(label, None, [f"the badge would read {message!r}, with no number in it"])
    return Result(label, message)


def check(spec: dict, text: str, require_re2: bool = False) -> bool:
    ok = True
    block = current_block(text)
    if block is None:
        print(f"FAIL README: no single badges block from a '<!-- badges:start' line to '{END}'")
        ok = False
    elif block != render_block(spec):
        print("FAIL README: the badges block differs from badges.toml; run: python .github/badges/badges.py write")
        ok = False
    for badge in number_badges(spec):
        result = evaluate(badge, text, require_re2)
        if result.problems:
            ok = False
            print(f"FAIL {result.label}")
            for problem in result.problems:
                print(f"     {problem}")
        else:
            print(f"ok   {result.label}: {result.message}")
    return ok


def write(spec: dict, path: Path) -> None:
    raw = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    block = current_block(text)
    if block is None:
        raise SystemExit(f"{path.name}: no single badges block to replace; add a '<!-- badges:start' line and "
                         f"'{END}' around the badges first")
    text = text.replace(block, render_block(spec), 1)
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))


def fetch(url: str, attempts: int = 3) -> str:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ignition-readme-badges"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8")
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def live(spec: dict, ref: str) -> bool:
    """Compare img.shields.io's rendering at `ref` with this module's; a commit's raw file is not cached stale."""
    text = fetch(raw_url(spec, ref)).replace("\r\n", "\n")
    ok = True
    for badge in number_badges(spec):
        expected = evaluate(badge, text)
        title = re.search(r"<title>(.*?)</title>", fetch(regex_url(spec, badge, ref)), re.S)
        shown = html.unescape(title.group(1)) if title else "(no title in the SVG)"
        wanted = f"{badge['label']}: {expected.message}" if expected.message else None
        if shown == wanted:
            print(f"ok   {shown}")
            continue
        ok = False
        print(f"FAIL {badge['label']}: img.shields.io shows {shown!r}, expected {wanted!r}")
        for problem in expected.problems:
            print(f"     {problem}")
    return ok


def main(argv: list | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description="Check or regenerate the README badges.")
    commands = parser.add_subparsers(dest="command", required=True)
    check_cmd = commands.add_parser("check", help="verify the badge block and every number badge's pattern")
    check_cmd.add_argument("--require-re2", action="store_true", help="fail when google-re2 is not installed")
    commands.add_parser("write", help="regenerate the badge block in README.md, then check")
    live_cmd = commands.add_parser("live", help="compare img.shields.io's rendering at a commit")
    live_cmd.add_argument("--ref", required=True, help="commit (or branch) whose README.md to render")
    args = parser.parse_args(argv)

    spec = load_spec()
    readme = ROOT / spec["readme"]
    if args.command == "write":
        write(spec, readme)
        ok = check(spec, read_text(readme))
    elif args.command == "check":
        ok = check(spec, read_text(readme), args.require_re2)
    else:
        ok = live(spec, args.ref)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
