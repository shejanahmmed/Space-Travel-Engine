"""Central Benchmark Persistence Framework.

Every benchmark script MUST import and use this module to record results.
Calling `BenchmarkSession.save()` writes a timestamped JSON + Markdown
report to `benchmarks/reports/` and appends a one-line entry to
`benchmarks/reports/AUDIT_TRAIL.jsonl` — the authoritative audit log.

Usage pattern in any benchmark script:
    from benchmarks.save_results import BenchmarkSession, BenchmarkResult

    session = BenchmarkSession(
        milestone="B15",
        name="My New Benchmark Suite",
        version="0.1.0",
    )
    session.add(BenchmarkResult(
        id="B15-01",
        name="Scenario name",
        evaluated_value="42.0 m/s",
        reference_value="42.0 m/s (Source 2024)",
        discrepancy="0.0% error",
        passed=True,
        notes="Optional technical note.",
    ))
    session.save()  # <-- always called, never optional

Design constraints:
- `save()` is unconditional; results are written even on partial failure.
- JSON record is machine-readable for future automated cross-validation.
- AUDIT_TRAIL.jsonl enables diffs between engine versions.
- Deterministic local file I/O with zero network dependencies.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Canonical reports directory -- always relative to THIS file's location
_REPORTS_DIR = Path(__file__).resolve().parent / "reports"
_AUDIT_TRAIL = _REPORTS_DIR / "AUDIT_TRAIL.jsonl"


def _git_short_sha() -> str:
    """Return current Git short SHA for reproducibility tracing, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


@dataclass
class BenchmarkResult:
    """Single benchmark scenario result.

    Attributes:
        id:               Identifier e.g. 'B14-01'.
        name:             Human-readable scenario name.
        evaluated_value:  Stringified computed value with units.
        reference_value:  Authoritative reference value with source citation.
        discrepancy:      Quantified difference (absolute or relative).
        passed:           True iff within the benchmark's acceptance criterion.
        notes:            Optional technical commentary on methodology or caveats.
    """
    id: str
    name: str
    evaluated_value: str
    reference_value: str
    discrepancy: str
    passed: bool
    notes: str = ""


@dataclass
class BenchmarkSession:
    """Container for a full benchmark run.

    Args:
        milestone:  Milestone tag, e.g. 'M14', 'M15-P1'. Used in filenames.
        name:       Full human-readable benchmark suite name.
        version:    Engine version string, defaults to '0.1.0'.
    """
    milestone: str
    name: str
    version: str = "0.1.0"
    results: List[BenchmarkResult] = field(default_factory=list)
    _timestamp: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        self._timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def add(self, result: BenchmarkResult) -> None:
        """Append a completed benchmark result to the session."""
        self.results.append(result)

    @property
    def all_passed(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)

    @property
    def score(self) -> str:
        n = len(self.results)
        p = sum(1 for r in self.results if r.passed)
        return f"{p}/{n}"

    def save(self) -> Path:
        """Persist results unconditionally to JSON, Markdown, and AUDIT_TRAIL.

        This method is unconditional: it is always called at the end of every
        benchmark session regardless of pass/fail status. The audit trail must
        record failures too, so that regressions are detectable.

        Returns:
            Path to the written Markdown report.
        """
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        safe_tag = self.milestone.replace(" ", "_").replace("/", "-").upper()
        md_path   = _REPORTS_DIR / f"{safe_tag}_BENCHMARK_REPORT.md"
        json_path = _REPORTS_DIR / f"{safe_tag}_BENCHMARK_DATA.json"

        # 1. Write JSON record (machine-readable, version-control friendly)
        json_record = {
            "milestone": self.milestone,
            "name": self.name,
            "version": self.version,
            "timestamp": self._timestamp,
            "git_sha": _git_short_sha(),
            "python_version": _python_version(),
            "platform": platform.platform(),
            "score": self.score,
            "all_passed": self.all_passed,
            "results": [asdict(r) for r in self.results],
        }
        json_path.write_text(
            json.dumps(json_record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 2. Write Markdown report (human-readable)
        md_path.write_text(self._build_markdown(json_record), encoding="utf-8")

        # 3. Append one-line entry to the audit trail (never overwritten)
        audit_entry = {
            "timestamp": self._timestamp,
            "milestone": self.milestone,
            "score": self.score,
            "all_passed": self.all_passed,
            "git_sha": json_record["git_sha"],
            "report": md_path.name,
            "json": json_path.name,
        }
        with _AUDIT_TRAIL.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(audit_entry) + "\n")

        print(f"[SAVED] {md_path}")
        print(f"[SAVED] {json_path}")
        print(f"[APPENDED] {_AUDIT_TRAIL}")
        return md_path

    def _build_markdown(self, meta: dict) -> str:
        status_str = "PASSED & CERTIFIED" if self.all_passed else "PARTIALLY FAILED"
        lines = [
            f"# {self.milestone} Benchmark Report",
            f"## {self.name}",
            "",
            f"**Audit Status**: `{status_str}`  ",
            f"**Score**: `{self.score} PASSED`  ",
            f"**Timestamp**: `{self._timestamp}`  ",
            f"**Engine Version**: `{self.version}`  ",
            f"**Git SHA**: `{meta['git_sha']}`  ",
            f"**Python**: `{meta['python_version']}`  ",
            "",
            "---",
            "",
            "## Results Matrix",
            "",
            "| ID | Scenario | Evaluated Value | Reference | Discrepancy | Status |",
            "|:---|:---------|:----------------|:----------|:------------|:-------|",
        ]
        for r in self.results:
            status_icon = "PASS" if r.passed else "FAIL"
            lines.append(
                f"| `{r.id}` | **{r.name}** | `{r.evaluated_value}` "
                f"| {r.reference_value} | {r.discrepancy} | `{status_icon}` |"
            )
            if r.notes:
                lines.append(f"|  | *{r.notes}* | | | | |")

        lines.extend([
            "",
            "---",
            "",
            "```",
            f"AUDIT SIGNATURE: {self.milestone} VERIFIED SCIENTIFIC RECORD",
            f"TIMESTAMP:       {self._timestamp}",
            f"GIT SHA:         {meta['git_sha']}",
            f"SCORE:           {self.score}",
            "CERTIFIED BY:    Relativistic Space Travel Computational Engine Audit Framework",
            "```",
        ])
        return "\n".join(lines)
