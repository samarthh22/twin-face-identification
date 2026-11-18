"""Create reproducible, family-disjoint train/validation/test splits."""

import argparse
import json
from pathlib import Path
import random


def make_splits(families, seed=42):
    families = sorted(families)
    if len(families) < 10:
        raise ValueError(
            "At least 10 families are needed: train >= 4, calibration >= 2, assessment >= 2, test >= 2"
        )
    random.Random(seed).shuffle(families)
    test_count = max(2, round(len(families) * 0.2))
    val_count = max(4, round(len(families) * 0.2))
    return {
        "seed": seed,
        "train_fids": sorted(families[test_count + val_count :]),
        "val_fids": sorted(families[test_count : test_count + val_count]),
        "test_fids": sorted(families[:test_count]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Aligned dataset containing FIDs/family/person/images",
    )
    parser.add_argument("--output", type=Path, default=Path("data/splits.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(
            "Split file already exists; use a new output path to avoid changing an established protocol"
        )
    families = [p.name for p in (args.data_root / "FIDs").iterdir() if p.is_dir()]
    result = make_splits(families, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
