#!/usr/bin/env python3
# Copyright (C) 2026 The Ignition contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Refuse a bare native invocation in install.ps1, which silently corrupts a function's return value.

In PowerShell a function's uncaptured output IS its return value. `Get-Python` called
`Install-WithWinget` as a bare statement, so on a machine where winget actually installed
something its console text joined the returned path in an array; `[string]` flattened that
array into one line, and step 7 tried to run

    Found Python 3.13 [Python.Python.3.13] ... Successfully installed C:\\...\\python.exe

as a command name. It only reproduced on a fresh machine, because only there did winget
have anything to say, which is why re-running the installer "fixed" it.

The rule: a native call written as a statement must send its output somewhere explicit.
Assigning it (`$x = & ...`) or piping it (`| Out-Host`, `| Out-Null`) are all fine; leaving
it bare is not. Exit 1 on a violation, so this can gate a commit.
"""
import re
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "install.ps1"
# A statement that starts with the call operator: optional indent, then "& $" or "& <word>.exe".
BARE_CALL = re.compile(r"^\s*&\s+\$")
FUNCTION = re.compile(r"^\s*function\s+[-\w]+")
SINKS = ("| Out-Host", "| Out-Null", "| Out-File", "> $null")


def main() -> int:
    if not SCRIPT.is_file():
        print(f"FAIL {SCRIPT} not found")
        return 1

    # Only inside a function does a stray output line become a return value. At the top level it
    # simply prints, which several steps rely on, so those are left alone.
    problems = []
    depth, in_function = 0, False
    for number, line in enumerate(SCRIPT.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.split("#", 1)[0].rstrip()
        if FUNCTION.match(stripped):
            in_function, depth = True, 0
        if in_function:
            depth += stripped.count("{") - stripped.count("}")
            if depth <= 0 and "{" not in stripped and "}" in stripped:
                in_function = False
        if not in_function or not BARE_CALL.match(stripped):
            continue
        if any(sink in stripped for sink in SINKS):
            continue
        problems.append((number, stripped.strip()))

    if problems:
        print(f"FAIL {len(problems)} bare native invocation(s) in install.ps1.")
        print("     Each writes to its function's output stream and can corrupt what that")
        print("     function returns. Pipe to Out-Host to keep it visible, or assign it.")
        for number, text in problems:
            print(f"  install.ps1:{number}  {text}")
        return 1

    print("ok   install.ps1: no bare native invocation can reach a function's return value")
    return 0


if __name__ == "__main__":
    sys.exit(main())
