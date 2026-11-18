"""Verify a face pair or identify a probe against a gallery of reference photos."""

import argparse
import json
from pathlib import Path

import cv2
import torch

from models.general_verifier import GeneralFaceVerifier

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*")
    parser.add_argument(
        "--gallery", type=Path, help="JSON mapping identity names to 1-3 image paths"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/general_selected_model.json"
    )
    parser.add_argument(
        "--weights", type=Path, help="Override the configured checkpoint path"
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--threads", type=int, default=2, help="CPU worker threads")
    parser.add_argument(
        "--aligned",
        action="store_true",
        help="Inputs are already aligned 112x112 BGR crops",
    )
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if args.describe:
            print(json.dumps(config, indent=2))
            return
        if len(args.images) != (1 if args.gallery else 2):
            parser.error("Supply two photos, or one probe photo with --gallery")
        if args.threads < 1:
            parser.error("--threads must be positive")
        torch.set_num_threads(args.threads)
        cv2.setNumThreads(args.threads)
        aligner = None
        if not args.aligned:
            from preprocessing.align import FaceAligner

            aligner = FaceAligner()

        def read_image(path):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot read image: {path}")
            return aligner.align(image) if aligner else image

        weights = args.weights or Path(config["weights"])
        if not args.weights and not weights.is_absolute():
            weights = ROOT / weights
        verifier = GeneralFaceVerifier(config, weights, args.device)
        if args.gallery:
            raw = json.loads(args.gallery.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not raw:
                raise ValueError("Gallery must be a nonempty JSON object")
            gallery = {}
            for identity, files in raw.items():
                if (
                    not isinstance(files, list)
                    or not 1 <= len(files) <= config["maximum_enrollment_photos"]
                ):
                    raise ValueError(
                        f"{identity}: provide a list of 1-3 reference image paths"
                    )
                gallery[identity] = [
                    read_image(args.gallery.parent / file) for file in files
                ]
            result = verifier.identify(read_image(args.images[0]), gallery)
        else:
            result = verifier.compare(*(read_image(path) for path in args.images))
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
