import numpy as np
import pytest
import torch

from degradation.simulator import apply_degradation
from preprocessing.align import FaceAligner
from scripts.prepare_splits import make_splits
from training.general_faces import digest
from utils.checkpoints import load_encoder


def test_family_splits_are_disjoint_and_repeatable():
    families = [f"F{i:03}" for i in range(20)]
    split = make_splits(families)
    assert split == make_splits(reversed(families))
    groups = [set(split[key]) for key in ("train_fids", "val_fids", "test_fids")]
    assert set.union(*groups) == set(families)
    assert all(
        not a.intersection(b) for i, a in enumerate(groups) for b in groups[i + 1 :]
    )
    with pytest.raises(ValueError, match="10 families"):
        make_splits(families[:3])


def test_checkpoint_prefixes_and_integrity(tmp_path):
    model = torch.nn.Linear(2, 2)
    state = {"model." + key: value for key, value in model.state_dict().items()}
    state["head.weight"] = torch.ones(5, 2)
    weights = tmp_path / "encoder.ckpt"
    torch.save({"state_dict": state}, weights)
    loaded = torch.nn.Linear(2, 2)
    expected = digest(weights)
    load_encoder(loaded, weights, expected)
    assert torch.equal(model.weight, loaded.weight)
    with weights.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        load_encoder(loaded, weights, expected)
    with pytest.raises(FileNotFoundError, match="download_models"):
        load_encoder(loaded, tmp_path / "absent.ckpt", expected)


@pytest.mark.parametrize("axis", ["blur", "jpeg", "illumination"])
def test_degradations_repeat_and_preserve_input(axis):
    image = np.random.default_rng(42).integers(0, 256, (112, 112, 3), dtype=np.uint8)
    original = image.copy()
    first, _ = apply_degradation(image, axis=axis, severity=3, seed=42)
    second, _ = apply_degradation(image, axis=axis, severity=3, seed=42)
    assert np.array_equal(first, second)
    assert np.array_equal(image, original)
    assert first.shape == image.shape and first.dtype == image.dtype
    assert not np.array_equal(first, image)


@pytest.mark.parametrize(
    "faces,message", [(None, "No face"), (np.zeros((2, 15)), "Multiple faces")]
)
def test_alignment_rejects_ambiguous_photographs(faces, message):
    class Detector:
        def setInputSize(self, size):
            pass

        def detect(self, image):
            return None, faces

    aligner = FaceAligner.__new__(FaceAligner)
    aligner.detector = Detector()
    with pytest.raises(ValueError, match=message):
        aligner.align(np.zeros((112, 112, 3), dtype=np.uint8))
