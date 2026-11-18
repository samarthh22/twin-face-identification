"""Locked general-face verifier and multi-photo cosine-template identification."""

import torch
from torch.nn import functional as F
from models.adaface_backbone import build_model
from training.general_faces import native_tensor
from utils.checkpoints import load_encoder


class GeneralFaceVerifier:
    def __init__(self, config, weights, device="cpu"):
        if config["input_contract"] != "BGR uint8 112x112 /127.5 -1":
            raise ValueError("Unsupported preprocessing")
        self.model = build_model("ir_50").to(device)
        load_encoder(
            self.model,
            weights,
            config["encoder_sha256"],
            config.get("weights_format", "state_dict"),
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.device = device
        self.config = config

    @torch.inference_mode()
    def embed(self, images):
        values, _ = self.model(
            torch.stack([native_tensor(im) for im in images]).to(self.device)
        )
        if not torch.isfinite(values).all() or not torch.allclose(
            values.norm(dim=1), torch.ones(len(images), device=self.device), atol=1e-4
        ):
            raise ValueError("Invalid embeddings")
        return values

    def compare(self, first, second):
        z = self.embed([first, second])
        score = float((z[0] * z[1]).sum())
        threshold = self.config["threshold"]
        return {
            "cosine_similarity": score,
            "threshold": threshold,
            "prediction": "same_identity"
            if score >= threshold
            else "different_identity",
            "score_is_probability": False,
            "threshold_source": self.config.get(
                "threshold_source", "Independent calibration families"
            ),
        }

    def identify(self, probe, gallery):
        identities = sorted(gallery)
        if not identities:
            raise ValueError("Empty gallery")
        templates = []
        for identity in identities:
            photos = gallery[identity]
            if not 1 <= len(photos) <= self.config["maximum_enrollment_photos"]:
                raise ValueError(
                    "Gallery requires one to three reference photos per identity"
                )
            templates.append(F.normalize(self.embed(photos).mean(0), dim=0))
        score = torch.stack(templates) @ self.embed([probe])[0]
        index = int(score.argmax())
        return {
            "rank1_identity": identities[index],
            "cosine_similarity": float(score[index]),
            "gallery_identities": len(identities),
            "protocol": "closed-set identification; highest cosine to normalized reference-image mean",
        }
