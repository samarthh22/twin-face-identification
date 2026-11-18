"""Detect one face with OpenCV YuNet and align its five landmarks to 112x112."""

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class FaceAligner:
    def __init__(self, model_path=None):
        path = (
            Path(model_path)
            if model_path
            else ROOT / "checkpoints/face_detection_yunet_2023mar.onnx"
        )
        if not path.is_file():
            raise FileNotFoundError(
                "Missing detector. Run python scripts/download_models.py"
            )
        self.detector = cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.8, 0.3)

    def align(self, image):
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a readable color photograph")
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            raise ValueError("No face detected. Use a clear, front-facing photograph.")
        if len(faces) != 1:
            raise ValueError(
                "Multiple faces detected. Crop the photograph to the intended person."
            )
        points = faces[0][4:14].reshape(5, 2).astype(np.float32)
        eyes = sorted(points[:2], key=lambda point: point[0])
        mouth = sorted(points[3:], key=lambda point: point[0])
        landmarks = np.array([*eyes, points[2], *mouth], dtype=np.float32)
        matrix, _ = cv2.estimateAffinePartial2D(landmarks, REFERENCE, method=cv2.LMEDS)
        if matrix is None:
            raise ValueError("Unable to align facial landmarks")
        return cv2.warpAffine(
            image, matrix, (112, 112), flags=cv2.INTER_LINEAR, borderValue=0
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, target = args.input.resolve(), args.output.resolve()
    if source == target or source in target.parents:
        parser.error("Output must be separate from the input tree")
    if not source.is_dir():
        parser.error("Input must be a directory")
    files = sorted(
        p for p in source.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not files:
        parser.error("No image files found")
    aligner = FaceAligner()
    failures = 0
    for path in files:
        output = (target / path.relative_to(source)).with_suffix(".png")
        try:
            if output.exists():
                raise ValueError(
                    "Output already exists; choose an empty output directory"
                )
            crop = aligner.align(cv2.imread(str(path)))
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), crop):
                raise ValueError("Could not write aligned image")
        except ValueError as error:
            failures += 1
            print(f"FAILED {path}: {error}")
    print(f"Aligned {len(files) - failures}/{len(files)} images; failed {failures}.")
    if failures:
        parser.exit(
            1, "Some images were not aligned; review failures before evaluation.\n"
        )


if __name__ == "__main__":
    main()
