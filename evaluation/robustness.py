"""Frozen-model verification under controlled degradation, including own-twin impostors."""

import hashlib, itertools, random
from pathlib import Path
import cv2
from training.general_faces import digest
from evaluation.general_faces import scores_for, identification
from evaluation.calibration import binary_metrics
from degradation.simulator import (
    apply_blur,
    apply_jpeg_compression,
    apply_illumination_shift,
)


def transform_for(axis, severity):
    def transform(image, row, index):
        seed = int(
            hashlib.sha256(
                (row["file_sha256"] + axis + str(severity) + "42").encode()
            ).hexdigest()[:8],
            16,
        ) % (2**31 - 1)
        if axis in ["isotropic", "anisotropic", "motion"]:
            return apply_blur(
                image, severity, kernel_type=axis, split="eval", seed=seed
            )[0]
        if axis == "jpeg":
            return apply_jpeg_compression(image, severity, split="eval", seed=seed)[0]
        return apply_illumination_shift(image, severity, mode=axis, seed=seed)[0]

    return transform


def twin_protocol(root, config):
    rows = []
    pairs = []
    rng = random.Random(config["seed"])
    groups = sorted(p for p in Path(root).iterdir() if p.is_dir())
    if not groups:
        raise ValueError("No twin groups found")
    for group in groups:
        people = sorted(p for p in group.iterdir() if p.is_dir())
        if len(people) != 2:
            raise ValueError("Twin set requires two identities")
        indices = []
        for person in people:
            ids = []
            for path in sorted(person.iterdir()):
                if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue
                image = cv2.imread(str(path))
                if image is None or image.shape != (112, 112, 3):
                    raise ValueError("Invalid CTTS crop")
                ids.append(len(rows))
                rows.append(
                    {
                        "path": str(path),
                        "identity": group.name + "/" + person.name,
                        "family": group.name,
                        "photo": path.stem,
                        "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                        "file_sha256": digest(path),
                    }
                )
            if len(ids) < 4:
                raise ValueError("Need distinct CTTS enrollment/probe images")
            candidates = [
                (a, b, 1)
                for a, b in itertools.combinations(ids, 2)
                if rows[a]["pixel_sha256"] != rows[b]["pixel_sha256"]
            ]
            rng.shuffle(candidates)
            pairs += candidates[: config["twin_pairs_per_set"]["genuine_per_identity"]]
            indices.append(ids)
        candidates = [
            (a, b, 0)
            for a, b in itertools.product(*indices)
            if rows[a]["pixel_sha256"] != rows[b]["pixel_sha256"]
        ]
        rng.shuffle(candidates)
        pairs += candidates[: config["twin_pairs_per_set"]["own_twin"]]
    return rows, pairs


def twin_metric(rows, pairs, clean, damaged, threshold, enrollment):
    scores, labels = scores_for(clean, pairs, damaged)
    prediction = scores >= threshold
    tar = float(prediction[labels == 1].mean())
    tnr = float((~prediction[labels == 0]).mean())
    return {
        "accuracy": float((prediction == labels).mean()),
        "balanced_accuracy": (tar + tnr) / 2,
        "genuine_acceptance": tar,
        "twin_rejection": tnr,
        "twin_false_acceptance": 1 - tnr,
        "genuine_false_rejection": 1 - tar,
        "threshold": threshold,
        "binary": binary_metrics(scores, labels),
        "identification": identification(
            rows, clean, damaged, gallery_images=enrollment
        ),
        "positive_pairs": int((labels == 1).sum()),
        "negative_pairs": int((labels == 0).sum()),
    }
