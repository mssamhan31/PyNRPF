"""Command line: run M9 on a CSV of readings and write the two tables and a summary.

    pynrpf run readings.csv --out results/ [--c 0.7] [--calibration -4.59,2.16]
                            [--site substation_id --timestamp timestamp --net-load net_load_MW --solar solar_MW]

Inputs:  a CSV with the four columns.
Outputs: ``site_days.csv``, ``intervals.csv`` and ``summary.json`` in the output folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import __version__
from .m9 import RELEASE_CALIBRATION, RELEASE_PHI, Calibration
from .m9.decision import DEFAULT_C
from .run import run
from .validate import DEFAULT_COLUMNS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pynrpf",
                                     description="Detect and correct a wrong reverse-power-flow sign with M9.")
    parser.add_argument("--version", action="version", version=f"pynrpf {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="score a CSV of fifteen-minute readings")
    p.add_argument("input", type=Path, help="CSV with site, timestamp, net load (MW) and solar estimate (MW)")
    p.add_argument("--out", type=Path, required=True, help="folder for site_days.csv, intervals.csv and summary.json")
    p.add_argument("--c", type=float, default=DEFAULT_C, help=f"the control; default {DEFAULT_C}")
    p.add_argument("--calibration", type=str, default=None, help="intercept,slope to replace the release calibration")
    p.add_argument("--phi", type=float, default=RELEASE_PHI, help="evidence floor in MW; default the release value")
    for role, default in DEFAULT_COLUMNS.items():
        p.add_argument(f"--{role.replace('_', '-')}", dest=f"col_{role}", default=default,
                       help=f"column name for {role}; default {default}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    calibration = RELEASE_CALIBRATION
    if args.calibration:
        intercept, slope = (float(v) for v in args.calibration.split(","))
        calibration = Calibration(intercept, slope, provenance="command line")
    columns = {role: getattr(args, f"col_{role}") for role in DEFAULT_COLUMNS}
    frame = pd.read_csv(args.input)
    result = run(frame, columns=columns, c=args.c, calibration=calibration, phi=args.phi)
    args.out.mkdir(parents=True, exist_ok=True)
    result.site_days.to_csv(args.out / "site_days.csv", index=False)
    result.intervals.to_csv(args.out / "intervals.csv", index=False)
    summary = dict(result.summary(), c=args.c, phi=args.phi,
                   calibration=dict(intercept=calibration.intercept, slope=calibration.slope))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
