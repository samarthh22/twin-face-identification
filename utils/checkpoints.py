"""Load hash-verified public or locally trained encoder checkpoints."""

from pathlib import Path

import torch

from training.general_faces import digest


def load_encoder(model, weights, expected_sha256, weights_format="adaface_checkpoint"):
    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing weights: {path}. Run python scripts/download_models.py"
        )
    if digest(path) != expected_sha256:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if weights_format == "adaface_checkpoint":
        state = {
            key.removeprefix("model."): value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("model.")
        }
    elif weights_format == "state_dict":
        state = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format: {weights_format}")
    model.load_state_dict(state, strict=True)
    return model
