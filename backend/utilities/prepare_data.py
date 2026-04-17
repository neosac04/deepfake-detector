from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def dfdc_metadata_json_to_csv(json_path: str | Path, output_csv: str | Path) -> None:
    """
    Convert DFDC-style metadata.json into a flat CSV.
    Output columns: file,label,original,split
    """
    json_path = Path(json_path)
    output_csv = Path(output_csv)

    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for file_name, meta in raw.items():
        rows.append(
            {
                "file": file_name,
                "label": meta.get("label", ""),
                "original": meta.get("original", ""),
                "split": meta.get("split", ""),
            }
        )
    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True, help="Path to metadata.json")
    parser.add_argument("--output_csv", required=True, help="Path to output CSV")
    args = parser.parse_args()
    dfdc_metadata_json_to_csv(args.json_path, args.output_csv)


if __name__ == "__main__":
    main()
