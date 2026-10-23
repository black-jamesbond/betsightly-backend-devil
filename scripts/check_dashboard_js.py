#!/usr/bin/env python3
"""Syntax-check Flask dashboard's inline JavaScript without browser execution."""
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
html = (root / "dashboard/templates/index.html").read_text(encoding="utf-8")
blocks = re.findall(r"<script(?:\s[^>]*)?>([\s\S]*?)</script>", html, re.IGNORECASE)
if not blocks:
    raise SystemExit("No inline dashboard JavaScript was found")
with tempfile.TemporaryDirectory(prefix="betsightly-js-") as scratch:
    for index, source in enumerate(blocks):
        path = Path(scratch) / f"dashboard-{index}.js"
        # Flask/Jinja injects values before the browser parses this script.
        # Replace only those template expressions with a JS string for syntax checks.
        sanitized = re.sub(r"\{\{[\s\S]*?\}\}", '""', source)
        path.write_text(sanitized, encoding="utf-8")
        result = subprocess.run(["node", "--check", str(path)], check=False)
        if result.returncode:
            raise SystemExit(result.returncode)
print(f"Validated {len(blocks)} inline dashboard script(s)")
