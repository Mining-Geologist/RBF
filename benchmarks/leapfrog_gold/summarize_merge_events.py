"""Convert passive SubDomainer JSONL event data into an auditable merge graph.

Use: python benchmarks/leapfrog_gold/summarize_merge_events.py trace/merge-events.jsonl
Only runs on local diagnostic output. Does not import Leapfrog.
"""
from __future__ import annotations
import argparse
import collections
import csv
import json
from pathlib import Path

def summarize(path: Path, output: Path) -> dict:
    records, errors = [], []
    with path.open(encoding="utf-8") as src:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    records.append(payload)
            except json.JSONDecodeError as exc:
                errors.append({"line": line_no, "error": str(exc)})
    counts = collections.Counter(item.get("method", "unknown") for item in records
                                 if item.get("type") == "return")
    rows = []
    for event in records:
        if event.get("type") != "return" or not event.get("method", "").endswith(".merge_domains"):
            continue
        before = event.get("before", {})
        after = event.get("after", {})
        rows.append({
            "sequence": event.get("sequence", ""),
            "method": event["method"],
            "arguments": json.dumps(event.get("arguments", [])),
            "removed_domains": "|".join(event.get("removed_domains", [])),
            "created_domains": "|".join(event.get("created_domains", [])),
            "domains_before": before.get("domains_size", ""),
            "domains_after": after.get("domains_size", ""),
            "queue_before": before.get("domainq_size", ""),
            "queue_after": after.get("domainq_size", ""),
            "elapsed_ms": event.get("elapsed_ns", 0) / 1e6,
        })
    output.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output / "merge_graph.csv").open("w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {"events": len(records), "methods": dict(counts),
               "merge_returns": len(rows), "errors": errors,
               "first_merge": rows[0] if rows else None,
               "last_merge": rows[-1] if rows else None,
               "warning": None if rows else "No observed merge decisions; cannot infer merge graph."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("merge-graph-results"))
    args = parser.parse_args()
    print(json.dumps(summarize(args.input, args.output), indent=2))

if __name__ == "__main__":
    main()
