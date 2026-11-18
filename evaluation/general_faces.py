"""Binary verification and held-out gallery/probe identity recognition."""

from collections import defaultdict
import cv2
import numpy as np
import torch
from evaluation.calibration import binary_metrics
from training.general_faces import native_tensor, pair_indices


@torch.inference_mode()
def encode(model, rows, device, batch_size=64, transform=None, flip=False):
    model.eval()
    out = []
    for start in range(0, len(rows), batch_size):
        batch = []
        for index, r in enumerate(rows[start : start + batch_size], start):
            image = cv2.imread(r["path"])
            if transform:
                image = transform(image, r, index)
            batch.append(native_tensor(image))
        x = torch.stack(batch).to(device)
        z, _ = model(x)
        if flip:
            flipped, _ = model(torch.flip(x, [3]))
            z = torch.nn.functional.normalize(z + flipped, dim=1)
        out.append(z.cpu().numpy())
    values = np.concatenate(out)
    if not np.isfinite(values).all() or not np.allclose(
        np.linalg.norm(values, axis=1), 1, atol=1e-4
    ):
        raise ValueError("Nonfinite or nonunit embeddings")
    return values


def scores_for(embeddings, pairs, second=None):
    second = embeddings if second is None else second
    return np.array([np.dot(embeddings[a], second[b]) for a, b, _ in pairs]), np.array(
        [y for _, _, y in pairs]
    )


def fit_threshold(scores, labels, role):
    if role != "calibration":
        raise ValueError("Thresholds require calibration role")
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    unique = np.unique(scores)
    edges = np.r_[
        np.nextafter(unique[0], -np.inf),
        (unique[1:] + unique[:-1]) / 2,
        np.nextafter(unique[-1], np.inf),
    ]
    if set(labels) != {0, 1}:
        raise ValueError("Need both binary classes")
    measured = []
    for t in edges:
        tar = np.mean(scores[labels == 1] >= t)
        tnr = np.mean(scores[labels == 0] < t)
        measured.append(((tar + tnr) / 2, -abs(tar - tnr), float(t)))
    return max(measured)[2]


def identification(
    rows, clean, probe=None, gallery_images=1, probe_gallery_images=None
):
    probe = clean if probe is None else probe
    by_id = defaultdict(list)
    for i, row in enumerate(rows):
        by_id[row["identity"]].append(i)
    gallery = []
    probes = []
    templates = []
    counts = []
    for identity, indices in sorted(by_id.items()):
        photos = {}
        for i in indices:
            photos.setdefault(rows[i]["photo"], i)
        reserved = list(photos.values())[
            : min(probe_gallery_images or gallery_images, len(photos) - 1)
        ]
        chosen = reserved[:gallery_images]
        enrolled = {rows[i]["photo"] for i in reserved}
        other = [i for i in indices if rows[i]["photo"] not in enrolled]
        if chosen and other:
            gallery.append(chosen[0])
            probes.extend(other)
            template = clean[chosen].mean(0)
            templates.append(template / max(np.linalg.norm(template), 1e-12))
            counts.append(len(chosen))
    if not gallery or not probes:
        raise ValueError("No eligible gallery/probe identities")
    similarities = probe[probes] @ np.array(templates).T
    rank = similarities.argmax(1)
    correct = np.array(
        [
            rows[gallery[g]]["identity"] == rows[p]["identity"]
            for g, p in zip(rank, probes)
        ]
    )
    macro = np.mean(
        [
            np.mean(
                [
                    correct[j]
                    for j, p in enumerate(probes)
                    if rows[p]["identity"] == rows[g]["identity"]
                ]
            )
            for g in gallery
        ]
    )
    return {
        "rank1": float(correct.mean()),
        "macro_identity_rank1": float(macro),
        "gallery_identities": len(gallery),
        "probes": len(probes),
        "correct_probes": int(correct.sum()),
        "maximum_enrollment_photos": gallery_images,
        "probe_reserved_enrollment_photos": probe_gallery_images or gallery_images,
        "actual_enrollment_min": min(counts),
        "actual_enrollment_max": max(counts),
        "protocol": "deterministic clean enrollment photos; distinct-photo probes; all eligible identities as distractors",
    }


def evaluate(rows, embeddings, threshold, seed=42, second=None):
    pairs = pair_indices(rows, seed)
    scores, labels = scores_for(embeddings, pairs, second)
    prediction = scores >= threshold
    tar = float(prediction[labels == 1].mean())
    tnr = float((~prediction[labels == 0]).mean())
    return {
        "balanced_accuracy": (tar + tnr) / 2,
        "accuracy": float((prediction == labels).mean()),
        "genuine_acceptance": tar,
        "unrelated_rejection": tnr,
        "false_acceptance": 1 - tnr,
        "false_rejection": 1 - tar,
        "threshold": float(threshold),
        "confusion_matrix": [
            [
                int(((labels == 0) & ~prediction).sum()),
                int(((labels == 0) & prediction).sum()),
            ],
            [
                int(((labels == 1) & ~prediction).sum()),
                int(((labels == 1) & prediction).sum()),
            ],
        ],
        "binary": binary_metrics(scores, labels),
        "identification": identification(rows, embeddings, second),
        "positive_pairs": int((labels == 1).sum()),
        "negative_pairs": int((labels == 0).sum()),
    }


def selection_score(result):
    return min(result["balanced_accuracy"], result["identification"]["rank1"])
