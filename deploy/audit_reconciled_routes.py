#!/usr/bin/env python3
"""Check a three-way BetSightly dashboard resolution before any VPS deployment.

Reads the local merge index stage 2 (release), stage 3 (VPS), and candidate
file, with no network calls or code execution. Requires all Flask route
identities to survive unless deliberately reviewed and documented.
Does NOT modify a file, stage a change or restart a service.

Usage:
  python deploy/audit_reconciled_routes.py /path/to/reconciled-release-worktree
"""
from __future__ import annotations

import ast
import subprocess
import sys
from collections import Counter
from pathlib import Path

PATH = "dashboard/app.py"
EXPECTED = {
    "/api/booking/status", "/api/group-code", "/api/build/code",
    "/api/operations/health", "/api/settlement-report",
}


def git(root: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(root), *args],
                       check=True, capture_output=True, text=True)
    return p.stdout


def ast_routes(source: str) -> tuple[list[tuple[str, str]], set[str]]:
    tree = ast.parse(source)
    routes = []
    definitions = set()
    for n in tree.body:
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        definitions.add(n.name)
        for deco in n.decorator_list:
            if not (isinstance(deco, ast.Call) and
                    isinstance(deco.func, ast.Attribute) and
                    deco.func.attr == "route" and deco.args and
                    isinstance(deco.args[0], ast.Constant) and
                    isinstance(deco.args[0].value, str)):
                continue
            path = deco.args[0].value
            methods = {"GET"}  # Flask route default, unless methods= declared.
            for kw in deco.keywords:
                if kw.arg == "methods" and isinstance(kw.value, (ast.List, ast.Tuple)):
                    methods = {
                        node.value.upper() for node in kw.value.elts
                        if isinstance(node, ast.Constant) and
                        isinstance(node.value, str)
                    }
            for method in sorted(methods):
                routes.append((path, method))
    return routes, definitions


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    filename = root / PATH
    try:
        candidate_source = filename.read_text(encoding="utf-8")
        conflict_paths = git(root, "diff", "--name-only", "--diff-filter=U")
    except (OSError, subprocess.CalledProcessError) as exc:
        print("ABORT: candidate worktree or index inaccessible:", exc)
        return 2

    if conflict_paths.strip() or any(x in candidate_source for x in (
            "<<<<<<< ", "=======", ">>>>>>> ")):
        print("BLOCKED: unresolved Git conflicts remain")
        print("Paths:", ", ".join(conflict_paths.splitlines()))
        return 1

    try:
        actual, definitions = ast_routes(candidate_source)
    except SyntaxError as exc:
        print(f"BLOCKED: reconciled Flask file has Python syntax error at {exc.lineno}")
        return 1

    wanted = set()
    for stage, label in ((2, "tested release"), (3, "live VPS")):
        try:
            source = git(root, "show", f":{stage}:{PATH}")
        except subprocess.CalledProcessError:
            # Git discards conflict stages after the file was staged. The full
            # version inventory must be captured BEFORE git add.
            print(f"NOTICE: {label} index stage is gone; "
                  "route-union check requires pre-resolution report")
            continue
        try:
            routes, _ = ast_routes(source)
        except SyntaxError:
            print(f"BLOCKED: {label} route source is not parseable")
            return 1
        wanted.update(routes)

    found = set(actual)
    missing = sorted(wanted - found)
    duplicated = sorted(
        route for route, count in Counter(actual).items() if count > 1
    )
    expected_missing = sorted(
        p for p in EXPECTED if (p, "GET") not in found
    )

    print("=== VPS MERGED DASHBOARD ROUTE AUDIT ===")
    print("Candidate routes:", len(actual))
    print("Union of available release/VPS staged routes:", len(wanted))
    print("Candidate top-level functions:", len(definitions))
    print("Required booking and operations paths missing:", expected_missing)
    print("Release/VPS routes lost:", missing)
    print("Duplicate route+method mappings:", duplicated)
    if missing or duplicated or expected_missing:
        print("BLOCKED: reconcile routes before release")
        return 1
    print("Route audit PASS (still requires behavioral and live tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
