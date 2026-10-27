"""
Memory script to check experiments/spinpulse_memory.py: a tiny end-to-end run."""

import csv

import pytest

pytest.importorskip("spin_pulse")

from experiments.spinpulse_memory import main

TINY = ["--seeds", "42", "--window-length", "3", "--length", "80", "--noise-samples", "2"]


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_memory_sweep_writes_one_row_per_level_seed_and_delay(tmp_path):
    out = tmp_path / "memory.csv"
    main(["--levels", "ideal", "sp-ideal", "1", "--out", str(out), *TINY])
    rows = read(out)

    assert len(rows) == 3 * 3  # 3 levels x 3 delays
    capacity = {(r["level"], int(r["delay"])): float(r["capacity"]) for r in rows}
    assert all(0.0 <= c <= 1.0 for c in capacity.values())
    # SpinPulse without noise only adds its tiny intrinsic gate error to the ideal circuit
    for d in range(3):
        assert capacity[("sp-ideal", d)] == pytest.approx(capacity[("ideal", d)], abs=0.02)
    # the realistic chip reports its window duration in us
    assert all(float(r["window_us"]) > 0 for r in rows if r["level"] != "ideal")


def test_memory_sweep_resumes_without_repeating_runs(tmp_path):
    out = tmp_path / "memory.csv"
    main(["--levels", "ideal", "--out", str(out), *TINY])
    main(["--levels", "ideal", "sp-ideal", "--out", str(out), *TINY])
    levels = [r["level"] for r in read(out)]
    assert levels.count("ideal") == 3 and levels.count("sp-ideal") == 3