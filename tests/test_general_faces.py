import numpy as np
import pytest
from training.general_faces import pair_indices, native_tensor
from evaluation.general_faces import fit_threshold, identification


def test_native_bgr_contract():
    im = np.zeros((112, 112, 3), dtype=np.uint8)
    im[:, :, 0] = 255
    t = native_tensor(im)
    assert t[0].min() == 1 and t[2].max() == -1
    with pytest.raises(ValueError):
        native_tensor(im[:100])


def test_distinct_photo_pairs_and_unrelated_negatives():
    rows = [
        {
            "identity": f"F{k}/M1",
            "family": f"F{k}",
            "photo": str(p),
            "pixel_sha256": str((k, p)),
        }
        for k in range(3)
        for p in range(3)
    ]
    rows.append({**rows[0], "pixel_sha256": "newcrop"})
    pairs = pair_indices(rows)
    assert len({(a, b) for a, b, _ in pairs}) == len(pairs)
    assert sum(y for _, _, y in pairs) == len(pairs) // 2
    for a, b, y in pairs:
        if y:
            assert (
                rows[a]["identity"] == rows[b]["identity"]
                and rows[a]["photo"] != rows[b]["photo"]
            )
        else:
            assert rows[a]["family"] != rows[b]["family"]


def test_calibration_role_and_gallery_distractors():
    with pytest.raises(ValueError):
        fit_threshold([0.8, 0.1], [1, 0], "test")
    threshold = fit_threshold([0.9, 0.85, 0.2, 0.1], [1, 1, 0, 0], "calibration")
    assert 0.2 < threshold < 0.85
    rows = [
        {"identity": name, "photo": str(i)} for name in ["a", "b"] for i in range(2)
    ]
    emb = np.array([[1, 0], [1, 0], [0, 1], [1, 0]], dtype=float)
    result = identification(rows, emb)
    assert result["rank1"] == 0.5 and result["gallery_identities"] == 2


def test_multiphoto_enrollment_keeps_distinct_probes_and_same_distractors():
    rows = [
        {"identity": name, "photo": str(i)} for name in ["a", "b"] for i in range(4)
    ]
    emb = np.array([[1, 0]] * 4 + [[0, 1]] * 4, dtype=float)
    multi = identification(rows, emb, gallery_images=3)
    control = identification(rows, emb, gallery_images=1, probe_gallery_images=3)
    assert multi["probes"] == control["probes"] == 2
    assert multi["gallery_identities"] == control["gallery_identities"] == 2
    assert multi["rank1"] == control["rank1"] == 1


def test_deployment_pair_rule_and_template_mean():
    import torch
    from models.general_verifier import GeneralFaceVerifier

    verifier = GeneralFaceVerifier.__new__(GeneralFaceVerifier)
    verifier.config = {"threshold": 0.7, "maximum_enrollment_photos": 3}
    verifier.embed = lambda values: torch.nn.functional.normalize(
        torch.tensor(values, dtype=torch.float32), dim=1
    )
    assert verifier.compare([1, 0], [0, 1])["prediction"] == "different_identity"
    assert verifier.compare([1, 0], [1, 0])["prediction"] == "same_identity"
    result = verifier.identify([1, 0], {"a": [[1, 0], [0.9, 0.1]], "b": [[0, 1]]})
    assert result["rank1_identity"] == "a" and result["gallery_identities"] == 2
