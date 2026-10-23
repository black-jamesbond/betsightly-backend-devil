#!/usr/bin/env python3
"""Inspect isolated VPS three-way merge conflicts without changing any files.

Always review output for locally embedded credentials before sharing.
Usage:
  python deploy/review_vps_conflicts.py /private/reconciled-worktree
  python deploy/review_vps_conflicts.py /private/reconciled-worktree --file sportybet_jackpot.py
  python deploy/review_vps_conflicts.py /private/reconciled-worktree --file dashboard/app.py --summary

This program reads :1 (common base), :2 (release), :3 (VPS), and the
working conflict file. It NEVER calls checkout/add/commit/merge/abort.
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
from pathlib import Path

ALLOWED = {
    "sportybet_jackpot.py",
    "dashboard/app.py",
    "dashboard/templates/index.html",
}


def command(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def read_stage(root: Path, stage: int, path: str) -> str:
    try:
        return command(root, "show", f":{stage}:{path}")
    except subprocess.CalledProcessError:
        return ""


def functions(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ["<source contains syntax errors>"]
    found = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if isinstance(node, ast.ClassDef):
                found.append("class " + node.name)
            else:
                found.append(node.name)
    return found


def routes(source: str) -> list[str]:
    # The Flask routes are literal decorators: report paths, not code.
    return re.findall(r'@app\.route\(["\\']([^"\\']+)', source)


def hunks(source: str) -> list[dict]:
    """Parse standard diff3/zdiff3 Git conflict markers in exact on-disk order."""
    lines = source.splitlines()
    parsed = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("<<<<<<< "):
            i += 1
            continue
        start = i
        i += 1
        ours, base, theirs = [], [], []
        dest = ours
        while i < len(lines):
            line = lines[i]
            if line.startswith("||||||| "):
                dest = base
                i += 1
                continue
            if line.startswith("=======") and line.strip() == "=======":
                dest = theirs
                i += 1
                continue
            if line.startswith(">>>>>>> "):
                parsed.append({
                    "line": start + 1,
                    "release": ours,
                    "base": base,
                    "vps": theirs,
                })
                break
            dest.append(line)
            i += 1
        else:
            raise RuntimeError(f"Unterminated Git conflict starting line {start + 1}")
        i += 1
    return parsed


def report(root: Path, path: str, summary: bool) -> None:
    path = str(path)
    if path not in ALLOWED:
        raise ValueError("unexpected path: " + path)
    file = root / path
    if not file.is_file():
        raise ValueError("conflict file missing: " + str(file))
    source = file.read_text(encoding="utf-8")
    parts = hunks(source)
    print(f"=== {path} ===")
    print("Conflict hunks:", len(parts))
    stage2 = read_stage(root, 2, path)
    stage3 = read_stage(root, 3, path)
    print("Release-only functions:", ", ".join(sorted(set(functions(stage2)) -
                                                   set(functions(stage3)))))
    print("VPS-only functions:", ", ".join(sorted(set(functions(stage3)) -
                                               set(functions(stage2)))))
    if path == "dashboard/app.py":
        print("Release-only routes:", ", ".join(sorted(set(routes(stage2)) -
                                                    set(routes(stage3)))))
        print("VPS-only routes:", ", ".join(sorted(set(routes(stage3)) -
                                                set(routes(stage2)))))
    print("Release side length:", len(stage2.splitlines()))
    print("VPS side length:", len(stage3.splitlines()))
    for index, hunk in enumerate(parts, 1):
        print(f"\n--- HUNK {index} / {len(parts)} (worktree line {hunk['line']}) ---")
        if summary:
            print(f"Release {len(hunk['release'])} lines; "
                  f"VPS {len(hunk['vps'])} lines; "
                  f"base {len(hunk['base'])} lines")
            for key in ("release", "vps"):
                header = [s.strip() for s in hunk[key] if
                          re.match(r"\s*(def |async def |class |@app.route|function |const |let )", s)]
                print(key + " symbols: " + "; ".join(header[:15]))
        else:
            for key in ("release", "vps"):
                print(f"[{key.upper()} -- {len(hunk[key])} lines]")
                for row in hunk[key]:
                    # Redact common inline assignments. Still review the output.
                    masked = re.sub(
                        r"(?i)(api[_-]?key|password|client_secret|access_token|"
                        r"authorization|secret_key)\s*[:=]\s*(['\\"])[^'\\"]+\2",
                        r"\1=<REDACTED>", row,
                    )
                    print(masked)
    print(f"\n=== END {path} ===")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate", type=Path)
    p.add_argument("--file", choices=sorted(ALLOWED))
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()
    root = args.candidate.resolve()
    if not (root / ".git").exists():
        p.error("Candidate is not a Git worktree")
    files = [args.file] if args.file else sorted(ALLOWED)
    for file in files:
        report(root, file, args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
