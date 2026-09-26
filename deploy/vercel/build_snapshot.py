"""
Build the static snapshot that gets deployed to Vercel.

The dashboard is a Flask app that reads ~500 MB of local history, writes caches,
and runs 7-minute background jobs — none of which Vercel's serverless platform
can host. What it *can* host is the page itself with every response pre-baked,
which is what "assess it online" needs: a fast, shareable, read-only copy of
exactly what the dashboard shows.

So this script pulls each endpoint the page calls from the running dashboard,
writes them into data/, patches the page's one api() helper to read those files
instead of the server, disables the controls that need a backend, and drops the
result in this folder ready for `vercel deploy`.

Usage (with the dashboard running on 5050):
    python deploy/vercel/build_snapshot.py [--date YYYY-MM-DD] [--port 5050]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# Endpoint -> file name. Query strings are stripped: the snapshot is one date.
FILES = {
    "predictions": "predictions.json",
    "hourly-breakdown": "hourly-breakdown.json",
    "build": "build.json",
    "jackpot": "jackpot.json",
    "live": "live.json",
    "fetch-status": "fetch-status.json",
    "model-info": "model-info.json",
}


def get(path: str, timeout: int = 600):
    url = f"http://127.0.0.1:{PORT}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except Exception as exc:
        return None, str(exc).encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    global PORT
    PORT = args.port
    DATA.mkdir(parents=True, exist_ok=True)

    print(f"building the snapshot for {args.date} from localhost:{PORT}")
    status, page = get("/")
    if status != 200:
        print(f"  the dashboard is not answering on {PORT} — start it first")
        return 1
    html = page.decode("utf-8")

    baked = {}
    for endpoint, name in FILES.items():
        st, body = get(f"/api/{endpoint}?date={args.date}")
        if st != 200:
            print(f"  {endpoint}: HTTP {st} — skipped")
            continue
        (DATA / name).write_bytes(body)
        baked[endpoint] = len(body)
        print(f"  {endpoint:<20} -> data/{name}  ({len(body) // 1024} KB)")

    # The one variation the UI asks for by query: the Over/Under 1.5 filter.
    st, body = get(f"/api/build?date={args.date}&market=over_1.5")
    if st == 200:
        (DATA / "build_over15.json").write_bytes(body)
        print(f"  {'build (1.5 only)':<20} -> data/build_over15.json  ({len(body) // 1024} KB)")

    # The version probe must agree with the page's own hash or the freshness
    # guard would reload the page in a loop.
    m = re.search(r"window\._PAGE_HASH = \"?([0-9a-f]+)\"?;", html)
    page_hash = m.group(1) if m else ""
    (DATA / "version.json").write_text(json.dumps({
        "running": page_hash, "on_disk": page_hash, "stale": False, "page": page_hash,
        "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "uptime_min": 0,
        "reload_enabled": False, "network": {"ok": True, "error": "", "checked": 0},
        "restart_cmd": "", "static_snapshot": True,
    }), encoding="utf-8")
    (DATA / "generate-status.json").write_text(
        json.dumps({"status": "idle"}), encoding="utf-8")

    # --- patch the page -----------------------------------------------------
    old_api = re.search(r"function api\(path\) \{.*?\n\}", html, re.S)
    if not old_api:
        print("  could not find the api() helper — page layout changed?")
        return 1
    new_api = """function api(path) {
  // Static snapshot: every endpoint the page asks for is a pre-baked file.
  const clean = String(path).split('?')[0].replace(/^\\/api\\//, '');
  let file = clean.replace(/\\//g, '_');
  if (clean === 'build' && String(path).includes('market=over_1.5')) file = 'build_over15';
  return 'data/' + file + '.json';
}"""
    html = html[:old_api.start()] + new_api + html[old_api.end():]

    stamp = datetime.now().strftime("%d %b %Y %H:%M")
    banner = f"""
<div style="position:sticky;top:0;z-index:99;background:#1b2a1f;border-bottom:1px solid #2f4a36;
     color:#cfe8d6;font:13px/1.5 system-ui,sans-serif;padding:8px 14px">
  <b>Read-only snapshot</b> of the BetSightly dashboard, generated {stamp}.
  Every game, prediction, ticket and price below is real data captured at that moment.
  <span style="color:#9fb8a6">Fetch &amp; Predict, booking codes, live scores and the
  date picker need the app running on the machine it was built from.</span>
</div>"""
    html = html.replace("<body>", "<body>" + banner, 1)

    # Anything that needs a backend is intercepted instead of silently failing.
    guard = """
<script>
(function () {
  const needsServer = /generateForDate|getBuildCode|getGroupCode|refreshPredictions|_buildRebuild|switchTab\\('pipeline'\\)/;
  document.addEventListener('click', function (e) {
    const el = e.target.closest('[onclick]');
    if (el && needsServer.test(el.getAttribute('onclick') || '')) {
      e.preventDefault(); e.stopPropagation();
      el.title = 'Not available in the static snapshot — run the dashboard locally for this.';
    }
  }, true);
  // Grey out the controls that cannot work here so the page reads honestly.
  const mark = function () {
    document.querySelectorAll('[onclick]').forEach(function (el) {
      if (needsServer.test(el.getAttribute('onclick') || '')) el.style.opacity = '0.45';
    });
  };
  document.addEventListener('DOMContentLoaded', mark);
  new MutationObserver(mark).observe(document.documentElement, {childList: true, subtree: true});
})();
</script>"""
    html = html.replace("</body>", guard + "\n</body>", 1)

    # Lock the date picker: one snapshot, one date.
    html = html.replace("document.getElementById('datePicker').value = today;",
                        f"document.getElementById('datePicker').value = '{args.date}';\n"
                        f"document.getElementById('datePicker').disabled = true;", 1)
    html = html.replace('value="' + args.date + '"', 'value="' + args.date + '"', 1)

    (HERE / "index.html").write_text(html, encoding="utf-8")
    print(f"  wrote index.html ({len(html) // 1024} KB, page hash {page_hash})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
