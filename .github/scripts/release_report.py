"""Render the CI build/test report and merge it into a release body.

    python release_report.py render --din CSV --test-log NAME=LOG ... --sums SHA256SUMS.txt --rustc RUSTC_VV
    python release_report.py merge OLD_BODY.md REPORT.md   # prints the new body

The report sits between two HTML comment markers, so re-running the workflow replaces it
instead of appending a second copy; anything the author wrote outside the markers is kept.
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict

START = "<!-- hamt-ci-report:start -->"
END = "<!-- hamt-ci-report:end -->"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
RESULT = re.compile(r"test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored")


def test_counts(path):
    passed = failed = ignored = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RESULT.search(ANSI.sub("", line))
            if m:
                passed += int(m.group(2))
                failed += int(m.group(3))
                ignored += int(m.group(4))
    return passed, failed, ignored


def din_section(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        return ["_DIN EN 15026 benchmark produced no comparison rows._"]
    out = []
    for case in dict.fromkeys(r["case"] for r in rows):
        rs = [r for r in rows if r["case"] == case]
        days = sorted({int(r["day"]) for r in rs})
        tol_t = next(r["tolerance"] for r in rs if r["quantity"] == "temperature")
        tol_w = float(next(r["tolerance"] for r in rs if r["quantity"] == "water_content"))
        out += [
            f"### DIN EN 15026 benchmark (Annex B) — {case}",
            "",
            f"Criteria: temperature |ΔT| ≤ {tol_t} K; water content |Δw|/w_ref ≤ {tol_w:.1%}.",
            "",
            "| Day | max \\|ΔT\\| [K] | at depth [m] | max \\|Δw\\|/w_ref | \\|Δw\\| [kg/m³] | at depth [m] | Result |",
            "|---:|---:|---:|---:|---:|---:|:---|",
        ]
        by_day = defaultdict(list)
        for r in rs:
            by_day[int(r["day"])].append(r)
        for d in days:
            t = max((r for r in by_day[d] if r["quantity"] == "temperature"), key=lambda r: float(r["abs_diff"]))
            w = max((r for r in by_day[d] if r["quantity"] == "water_content"), key=lambda r: float(r["rel_diff"]))
            ok = all(r["pass"] == "true" for r in by_day[d])
            out.append(
                f"| {d} | {float(t['abs_diff']):.3f} | {t['depth_m']} | {float(w['rel_diff']):.2%} "
                f"| {float(w['abs_diff']):.2f} | {w['depth_m']} | {'pass' if ok else '**FAIL**'} |"
            )
        last = days[-1]
        out += ["", f"<details><summary>Final profile, day {last}</summary>", ""]
        out += ["| Depth [m] | T [°C] | T_ref [°C] | \\|ΔT\\| [K] | Result |", "|---:|---:|---:|---:|:---|"]
        for r in by_day[last]:
            if r["quantity"] == "temperature":
                out.append(f"| {r['depth_m']} | {float(r['sim']):.2f} | {float(r['ref']):.2f} | {float(r['abs_diff']):.3f} | {'pass' if r['pass'] == 'true' else '**FAIL**'} |")
        out += ["", "| Depth [m] | w [kg/m³] | w_ref [kg/m³] | \\|Δw\\| [kg/m³] | \\|Δw\\|/w_ref | Result |", "|---:|---:|---:|---:|---:|:---|"]
        for r in by_day[last]:
            if r["quantity"] == "water_content":
                out.append(f"| {r['depth_m']} | {float(r['sim']):.2f} | {float(r['ref']):.2f} | {float(r['abs_diff']):.2f} | {float(r['rel_diff']):.2%} | {'pass' if r['pass'] == 'true' else '**FAIL**'} |")
        out += ["", "</details>", ""]
    return out


def render(a):
    rustc = dict(l.split(": ", 1) for l in open(a.rustc, encoding="utf-8").read().splitlines() if ": " in l)
    server, repo, run_id = (os.environ.get(k, "") for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    sha = os.environ.get("GITHUB_SHA", "")
    lines = [
        START,
        "## Build & test",
        "",
        f"- Commit: `{sha[:7]}` · Toolchain: `rustc {rustc.get('release', '?')}` ({rustc.get('commit-date', '?')}) · `{os.environ.get('ImageOS', os.environ.get('RUNNER_OS', ''))}`",
    ]
    for spec in a.test_log:
        name, path = spec.split("=", 1)
        p, f, i = test_counts(path)
        lines.append(f"- `{name}` tests: {p} passed, {f} failed, {i} ignored")
    if run_id:
        lines.append(f"- CI run: {server}/{repo}/actions/runs/{run_id}")
    lines.append("")
    lines += din_section(a.din)
    lines += ["### Assets", "", "| File | SHA256 |", "|:---|:---|"]
    for l in open(a.sums, encoding="utf-8"):
        if l.strip():
            digest, name = l.split(maxsplit=1)
            lines.append(f"| `{name.strip().lstrip('*')}` | `{digest}` |")
    lines += ["", END, ""]
    sys.stdout.write("\n".join(lines))


def merge(a):
    body = open(a.old, encoding="utf-8").read()
    report = open(a.report, encoding="utf-8").read().strip()
    body = re.sub(re.escape(START) + r".*?" + re.escape(END), "", body, flags=re.S).rstrip()
    sys.stdout.write((body + "\n\n" if body else "") + report + "\n")


sys.stdout.reconfigure(encoding="utf-8", newline="\n")  # Windows runners default to cp1252

p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="cmd", required=True)
r = sub.add_parser("render")
r.add_argument("--din", required=True)
r.add_argument("--test-log", action="append", default=[])
r.add_argument("--sums", required=True)
r.add_argument("--rustc", required=True)
m = sub.add_parser("merge")
m.add_argument("old")
m.add_argument("report")
args = p.parse_args()
render(args) if args.cmd == "render" else merge(args)
