from __future__ import annotations

import argparse
from pathlib import Path

from oracle_review_core import (
    default_annotation_path,
    default_input_path,
    default_output_dir,
    export_reflagged_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the manually reflagged 2023-10-01 to 2024-09-30 oracle dataset."
    )
    parser.add_argument("--input", type=Path, default=default_input_path())
    parser.add_argument("--annotations", type=Path, default=default_annotation_path())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    args = parser.parse_args()

    result = export_reflagged_dataset(
        input_path=args.input,
        annotation_path=args.annotations,
        output_dir=args.output_dir,
    )
    print(f"Wrote CSV: {result.csv_path}")
    print(f"Wrote Parquet: {result.parquet_path}")
    print(f"Wrote summary: {result.summary_path}")
    print(f"Wrote status: {result.status_path}")
    print(f"Wrote checksums: {result.checksum_path}")
    print(
        "Review progress: "
        f"{result.reviewed_site_days}/{result.total_site_days} site-days "
        f"({'complete' if result.complete else 'incomplete'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
