# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Baracoda Group (Baracoda AI Labs)
"""Claude token counts through Anthropic's free /v1/messages/count_tokens endpoint.
Requires the environment variable ANTHROPIC_API_KEY (never written to disk by this script).
1) Calibrates the per-message overhead by difference: count(x), count(xx), count(xxx) for several texts x
   ending with a newline; overhead = 2*count(x) - count(xx). Raw responses are saved.
2) Counts every sentence of every corpus in its own request and subtracts the overhead.
Outputs results/claude_calibration.json and results/claude_counts.json (with request timestamps).
Anthropic describes the endpoint as an estimate; these are not invoices."""
import json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from common import RESULTS, load

MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
KEY = os.environ.get("ANTHROPIC_API_KEY")


def count(text):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": text}]}).encode()
    for a in range(12):
        req = urllib.request.Request("https://api.anthropic.com/v1/messages/count_tokens", data=body,
                                     headers={"x-api-key": KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        try:
            return json.load(urllib.request.urlopen(req, timeout=60))["input_tokens"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529):
                time.sleep(min(60, 2 ** a)); continue
            raise
    raise RuntimeError("too many retries")


def main():
    if not KEY:
        raise SystemExit("Set ANTHROPIC_API_KEY")
    probes = ["Bonjour.\n", "The cat sat on the mat.\n", "Le chat est sur le tapis.\n", "1234\n", "Hello world\n", "a\n",
              "Tous les êtres humains naissent libres.\n"]
    cal = [{"text": x, "c1": count(x), "c2": count(x * 2), "c3": count(x * 3)} for x in probes]
    for c in cal:
        c["overhead_2"] = 2 * c["c1"] - c["c2"]; c["overhead_3"] = (3 * c["c1"] - c["c3"]) / 2
    overheads = {c["overhead_2"] for c in cal} | {c["overhead_3"] for c in cal}
    if len(overheads) != 1:
        raise SystemExit(f"Inconsistent overhead estimates: {overheads}")
    oh = int(overheads.pop())
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "claude_calibration.json").write_text(json.dumps({"model": MODEL, "overhead": oh, "probes": cal,
        "at": datetime.now(timezone.utc).isoformat()}, indent=1, ensure_ascii=False))
    out = {"model": MODEL, "overhead": oh, "started": datetime.now(timezone.utc).isoformat(), "counts": {}}
    with ThreadPoolExecutor(8) as ex:
        for name, L in {**load("dev"), **load("final")}.items():
            out["counts"][name] = [c - oh for c in ex.map(count, L)]
            print(name, sum(out["counts"][name]), flush=True)
    out["finished"] = datetime.now(timezone.utc).isoformat()
    (RESULTS / "claude_counts.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
