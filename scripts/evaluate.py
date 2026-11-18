"""Evaluate a frozen encoder on general identities or twin pairs, optionally with degradation."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import torch

from evaluation.general_faces import encode, evaluate, identification
from evaluation.robustness import transform_for, twin_metric, twin_protocol
from models.general_verifier import GeneralFaceVerifier
from training.general_faces import audit_fiw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["general", "twins"], required=True)
    parser.add_argument("--data-root", required=True, help="Aligned image tree")
    parser.add_argument(
        "--manifest",
        help="Family-disjoint split JSON; required for the general protocol",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/general_selected_model.json"
    )
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--degradations",
        action="store_true",
        help="Also evaluate six degradation types at all five levels",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new file")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.protocol == "general" and not args.manifest:
        parser.error("--manifest is required for general evaluation")
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    verifier = GeneralFaceVerifier(
        config, args.weights or ROOT / config["weights"], args.device
    )
    if args.protocol == "general":
        rows = audit_fiw(args.data_root, args.manifest)["roles"]["test"]
        pairs = None
    else:
        rows, pairs = twin_protocol(
            args.data_root,
            {
                "seed": 42,
                "twin_pairs_per_set": {"genuine_per_identity": 10, "own_twin": 20},
            },
        )
    clean = encode(verifier.model, rows, args.device, batch_size=args.batch_size)
    conditions = [("clean", None, None)]
    if args.degradations:
        conditions += [
            (f"{axis}_{level}", axis, level)
            for axis in ["isotropic", "anisotropic", "motion", "jpeg", "under", "over"]
            for level in range(1, 6)
        ]
    result = {
        "protocol": args.protocol,
        "threshold": config["threshold"],
        "encoder_sha256": config["encoder_sha256"],
        "conditions": {},
        "reference": "clean; only the probe image is degraded",
    }
    for name, axis, level in conditions:
        damaged = (
            clean
            if axis is None
            else encode(
                verifier.model,
                rows,
                args.device,
                batch_size=args.batch_size,
                transform=transform_for(axis, level),
            )
        )
        if pairs is None:
            metric = evaluate(rows, clean, config["threshold"], second=damaged)
            metric["multi_photo_identification"] = identification(
                rows, clean, damaged, gallery_images=config["maximum_enrollment_photos"]
            )
        else:
            metric = twin_metric(
                rows,
                pairs,
                clean,
                damaged,
                config["threshold"],
                config["maximum_enrollment_photos"],
            )
        result["conditions"][name] = metric
        print(
            f"{name}: balanced accuracy {metric['balanced_accuracy']:.4f}", flush=True
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
