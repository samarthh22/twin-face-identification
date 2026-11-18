"""General-face protocol: family-disjoint roles, native AdaFace inputs, no data writes."""

import hashlib, itertools, json, random
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def native_tensor(image):
    if image is None or image.dtype != np.uint8 or image.shape != (112, 112, 3):
        raise ValueError("Expected existing 112x112 uint8 BGR crop")
    return torch.from_numpy(
        (image.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1).copy()
    )


def pair_indices(rows, seed=42, cap=20):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["identity"]].append(i)
    positive = []
    for identity, indices in sorted(groups.items()):
        pairs = [
            (a, b, 1)
            for a, b in itertools.combinations(indices, 2)
            if rows[a]["photo"] != rows[b]["photo"]
            and rows[a]["pixel_sha256"] != rows[b]["pixel_sha256"]
        ]
        rng.shuffle(pairs)
        positive.extend(pairs[:cap])
    if not positive:
        raise ValueError("No distinct-photo genuine pairs")
    negative = set()
    attempts = 0
    while len(negative) < len(positive):
        a, b = rng.sample(range(len(rows)), 2)
        attempts += 1
        if rows[a]["family"] != rows[b]["family"]:
            negative.add((min(a, b), max(a, b), 0))
        if attempts > 1000000:
            raise ValueError("Insufficient distinct unrelated pairs")
    return positive + sorted(negative)


def audit_fiw(data_root, manifest, seed=42):
    meta = json.loads(Path(manifest).read_text())
    validation = sorted(meta["val_fids"])
    random.Random(seed).shuffle(validation)
    half = len(validation) // 2
    families = {
        "train": meta["train_fids"],
        "calibration": validation[:half],
        "assessment": validation[half:],
        "test": meta["test_fids"],
    }
    flat = [f for group in families.values() for f in group]
    if len(set(flat)) != len(flat):
        raise ValueError("Family overlap")
    roles = {}
    seen = {}
    excluded = []
    invalid = []
    for role, groups in families.items():
        rows = []
        for family in sorted(groups):
            root = Path(data_root) / "FIDs" / family
            if not root.is_dir():
                raise FileNotFoundError(str(root))
            for person in sorted(p for p in root.iterdir() if p.is_dir()):
                for path in sorted(person.iterdir()):
                    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                        continue
                    image = cv2.imread(str(path))
                    if image is None or image.shape != (112, 112, 3):
                        invalid.append(str(path))
                        continue
                    pixel = hashlib.sha256(image.tobytes()).hexdigest()
                    identity = family + "/" + person.name
                    if pixel in seen:
                        old = seen[pixel]
                        if old["split"] != role or old["identity"] != identity:
                            raise ValueError(
                                "Duplicate pixels cross identity or split: " + str(path)
                            )
                        excluded.append(str(path))
                        continue
                    row = {
                        "path": str(path),
                        "identity": identity,
                        "family": family,
                        "photo": path.stem.split("_")[0],
                        "split": role,
                        "pixel_sha256": pixel,
                        "file_sha256": digest(path),
                    }
                    seen[pixel] = row
                    rows.append(row)
        roles[role] = rows
    if invalid:
        raise ValueError("Unreadable/wrong-sized FIW files: " + str(len(invalid)))
    summary = {
        role: {
            "families": len(families[role]),
            "identities": len({r["identity"] for r in rows}),
            "images": len(rows),
            "pairs": len(pair_indices(rows, seed)),
        }
        for role, rows in roles.items()
    }
    return {
        "roles": roles,
        "families": families,
        "summary": summary,
        "source_manifest_sha256": digest(manifest),
        "same_identity_exact_duplicates_excluded": len(excluded),
        "invalid_files": len(invalid),
        "seed": seed,
    }


class IdentityDataset(Dataset):
    def __init__(self, rows, seed=42):
        self.rows = rows
        self.labels = {
            identity: i
            for i, identity in enumerate(sorted({r["identity"] for r in rows}))
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = cv2.imread(row["path"])
        if image is None:
            raise FileNotFoundError(row["path"])
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
        if random.random() < 0.3:
            image = np.clip(
                image.astype(np.float32) * random.uniform(0.9, 1.1)
                + random.uniform(-5, 5),
                0,
                255,
            ).astype(np.uint8)
        return native_tensor(image), self.labels[row["identity"]]
