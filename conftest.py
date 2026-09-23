# Root-level pytest plugin.
#
# Automatically saves a JSON + Markdown test report to
# benchmarks/reports/TEST_RESULTS_<timestamp>.json and appends
# a one-line summary to benchmarks/reports/AUDIT_TRAIL.jsonl after
# every pytest session, regardless of pass/fail status.
#
# No user action required -- pytest loads this automatically.

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


_REPORTS_DIR = Path(__file__).resolve().parent / "benchmarks" / "reports"
_AUDIT_TRAIL = _REPORTS_DIR / "AUDIT_TRAIL.jsonl"


def _git_short_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
            cwd=Path(__file__).resolve().parent,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class _AuditPlugin:
    """Pytest plugin that persists test results after every session."""

    def __init__(self) -> None:
        self._results: list[dict] = []
        self._sha = _git_short_sha()

    def pytest_runtest_logreport(self, report) -> None:
        # Record only the final call phase (not setup / teardown noise)
        if report.when != "call":
            return
        self._results.append({
            "nodeid": report.nodeid,
            "outcome": report.outcome,
            "duration_s": round(report.duration, 6),
        })

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ts_safe = ts.replace(":", "-")

        total   = len(self._results)
        n_pass  = sum(1 for r in self._results if r["outcome"] == "passed")
        n_fail  = sum(1 for r in self._results if r["outcome"] == "failed")
        n_error = sum(1 for r in self._results if r["outcome"] == "error")
        n_skip  = sum(1 for r in self._results if r["outcome"] == "skipped")

        # 1 -- Full JSON test record
        json_path = _REPORTS_DIR / f"TEST_RESULTS_{ts_safe}.json"
        record = {
            "timestamp": ts,
            "git_sha": self._sha,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.platform(),
            "exit_status": int(exitstatus),
            "summary": {
                "total": total,
                "passed": n_pass,
                "failed": n_fail,
                "error": n_error,
                "skipped": n_skip,
            },
            "tests": self._results,
        }
        json_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 2 -- Markdown summary report
        md_path = _REPORTS_DIR / f"TEST_RESULTS_{ts_safe}.md"
        status_str = "ALL PASSED" if n_fail == 0 and n_error == 0 else "FAILURES DETECTED"
        md_lines = [
            "# Automated Test Suite Results",
            "",
            f"**Status**: `{status_str}`  ",
            f"**Timestamp**: `{ts}`  ",
            f"**Git SHA**: `{self._sha}`  ",
            f"**Python**: `{record['python_version']}`  ",
            "",
            "## Summary",
            "",
            f"| Total | Passed | Failed | Error | Skipped |",
            f"|------:|-------:|-------:|------:|--------:|",
            f"| {total} | {n_pass} | {n_fail} | {n_error} | {n_skip} |",
            "",
        ]
        if n_fail > 0 or n_error > 0:
            md_lines += [
                "## Failed / Errored Tests",
                "",
                "| Test Node ID | Outcome |",
                "|:-------------|:--------|",
            ]
            for r in self._results:
                if r["outcome"] not in ("passed", "skipped"):
                    md_lines.append(f"| `{r['nodeid']}` | `{r['outcome']}` |")
        md_path.write_text("\n".join(md_lines), encoding="utf-8")

        # 3 -- Append to audit trail (never overwritten; append-only log)
        audit_entry = {
            "timestamp": ts,
            "type": "pytest_session",
            "git_sha": self._sha,
            "score": f"{n_pass}/{total}",
            "all_passed": (n_fail == 0 and n_error == 0),
            "failed": n_fail,
            "error": n_error,
            "report": md_path.name,
            "json": json_path.name,
        }
        with _AUDIT_TRAIL.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(audit_entry) + "\n")

        print(f"\n[TEST AUDIT] {n_pass}/{total} passed -> {json_path.name}")


def pytest_configure(config) -> None:
    config.pluginmanager.register(_AuditPlugin(), "_audit_plugin")
