"""Test suite for the scientific CLI interface and archival trajectory exporter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest

from relativistic_engine.cli import (
    build_parser,
    _cmd_interplanetary,
    _cmd_interstellar,
    _cmd_benchmark,
    _cmd_dsn,
    _cmd_kerr,
    _cmd_manifest,
)
from relativistic_engine.export.exporter import export_trajectory_csv, export_trajectory_json
from relativistic_engine.trajectory.rendezvous import solve_interplanetary_rendezvous
from relativistic_engine.constants import G0


def test_cli_parser_structure():
    """Verify CLI parser options and commands."""
    parser = build_parser()
    
    # Check top-level commands
    args_interplanet = parser.parse_args([
        "solve", "interplanetary",
        "--departure", "earth",
        "--target", "mars",
        "--epoch", "2462622.5",
        "--accel", "1.0",
    ])
    assert args_interplanet.command == "solve"
    assert args_interplanet.subcommand == "interplanetary"
    assert args_interplanet.target == "mars"
    assert args_interplanet.accel == 1.0

    args_interstellar = parser.parse_args([
        "solve", "interstellar",
        "--target", "proxima_centauri",
        "--accel", "0.5",
    ])
    assert args_interstellar.subcommand == "interstellar"
    assert args_interstellar.target == "proxima_centauri"
    assert args_interstellar.accel == 0.5

    args_bench = parser.parse_args(["benchmark"])
    assert args_bench.command == "benchmark"

    args_dsn = parser.parse_args(["dsn", "shapiro", "--r1-au", "1.0", "--r2-au", "1.5"])
    assert args_dsn.command == "dsn"
    assert args_dsn.subcommand == "shapiro"

    args_kerr = parser.parse_args(["kerr", "--spin", "0.8"])
    assert args_kerr.command == "kerr"
    assert args_kerr.spin == 0.8

    args_manifest = parser.parse_args(["manifest", "--id", "TEST-001"])
    assert args_manifest.command == "manifest"
    assert args_manifest.id == "TEST-001"


def test_cli_interplanetary_execution_and_export(tmp_path: Path):
    """Verify CLI interplanetary solver and CSV/JSON output generation."""
    csv_path = tmp_path / "mars_trajectory.csv"
    json_path = tmp_path / "mars_trajectory.json"

    parser = build_parser()
    args = parser.parse_args([
        "solve", "interplanetary",
        "--departure", "earth",
        "--target", "mars",
        "--epoch", "2462622.5",
        "--accel", "1.0",
        "--output-csv", str(csv_path),
        "--output-json", str(json_path),
    ])

    exit_code = _cmd_interplanetary(args)
    assert exit_code == 0

    assert csv_path.exists(), "CSV trajectory not created"
    assert json_path.exists(), "JSON trajectory not created"

    # Verify CSV metadata and content
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_lines = [l for l in lines if l.startswith("#")]
    assert len(header_lines) >= 7
    assert any("BCRS / ICRF" in l for l in header_lines)
    assert any("Post-Newtonian" in l for l in header_lines)

    # Verify JSON content
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "trajectory" in data
    assert len(data["trajectory"]) > 10
    assert data["metadata"]["reference_frame"] == "BCRS / ICRF (IAU 2000/2006)"


def test_cli_interstellar_execution_and_export(tmp_path: Path):
    """Verify CLI interstellar solver execution."""
    csv_path = tmp_path / "proxima.csv"

    parser = build_parser()
    args = parser.parse_args([
        "solve", "interstellar",
        "--target", "proxima_centauri",
        "--accel", "1.0",
        "--output-csv", str(csv_path),
    ])

    exit_code = _cmd_interstellar(args)
    assert exit_code == 0
    assert csv_path.exists()


def test_cli_benchmark_execution():
    """Verify CLI benchmark command runs and reports success."""
    parser = build_parser()
    args = parser.parse_args(["benchmark"])
    exit_code = _cmd_benchmark(args)
    assert exit_code == 0


def test_cli_kerr_execution_and_export(tmp_path: Path):
    """Verify CLI Kerr solver and JSON export."""
    out_json = tmp_path / "kerr_sol.json"
    parser = build_parser()
    args = parser.parse_args([
        "kerr",
        "--spin", "0.9",
        "--mass", "10.0",
        "--inclination", "85.0",
        "--output-json", str(out_json),
    ])
    exit_code = _cmd_kerr(args)
    assert exit_code == 0
    assert out_json.exists()

    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["spin_dimensionless"] == 0.9
    assert data["gravitational_radius_km"] > 14.0
    assert "bardeen_shadow_contour" in data
    assert len(data["bardeen_shadow_contour"]["alpha_m"]) > 10


def test_cli_dsn_shapiro():
    """Verify CLI DSN Shapiro delay calculator."""
    parser = build_parser()
    args = parser.parse_args([
        "dsn", "shapiro",
        "--r1-au", "1.0",
        "--r2-au", "1.524",
        "--angle-deg", "180.0",
    ])
    exit_code = _cmd_dsn(args)
    assert exit_code == 0


def test_cli_dsn_simulate_and_export(tmp_path: Path):
    """Verify CLI DSN tracking simulation and export."""
    out_json = tmp_path / "dsn_arc.json"
    parser = build_parser()
    args = parser.parse_args([
        "dsn", "simulate",
        "--station", "DSS-14",
        "--body", "mars",
        "--duration", "2.0",
        "--step", "1.0",
        "--output-json", str(out_json),
    ])
    exit_code = _cmd_dsn(args)
    assert exit_code == 0
    assert out_json.exists()

    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["station"] == "DSS-14"
    assert len(data["points"]) == 3


def test_cli_manifest_execution_and_export(tmp_path: Path):
    """Verify CLI manifest generator exports W3C JSON-LD."""
    out_json = tmp_path / "manifest.jsonld"
    parser = build_parser()
    args = parser.parse_args([
        "manifest",
        "--id", "CERT-RUN-999",
        "--output", str(out_json),
    ])
    exit_code = _cmd_manifest(args)
    assert exit_code == 0
    assert out_json.exists()

    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["@type"] == "Dataset"
    assert data["identifier"] == "CERT-RUN-999"
    assert "constants" in data
    assert "result_sha256" in data["outputs"]
