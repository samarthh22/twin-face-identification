"""Optional identity fine-tuning with independent calibration and model selection."""

import argparse
import copy
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from evaluation.general_faces import (
    encode,
    evaluate,
    fit_threshold,
    scores_for,
    selection_score,
)
from models.adaface_backbone import build_model
from training.general_faces import IdentityDataset, audit_fiw, digest, pair_indices
from utils.checkpoints import load_encoder


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def measure(model, roles, device):
    embeddings = encode(model, roles["calibration"], device)
    scores, labels = scores_for(embeddings, pair_indices(roles["calibration"]))
    threshold = fit_threshold(scores, labels, "calibration")
    assessment = encode(model, roles["assessment"], device)
    return evaluate(roles["assessment"], assessment, threshold), threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/general_selected_model.json"
    )
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Two epochs, at most two batches each; excludes test evaluation",
    )
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error("epochs, patience, and batch-size must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    torch.backends.cudnn.benchmark = False
    config = json.loads(args.config.read_text(encoding="utf-8"))
    weights = args.weights or ROOT / config["weights"]
    dataset_info = audit_fiw(args.data_root, args.manifest)
    roles = dataset_info["roles"]
    write(args.output / "dataset_summary.json", dataset_info["summary"])
    model = build_model("ir_50").to(args.device)
    load_encoder(model, weights, config["encoder_sha256"], config["weights_format"])
    baseline, best_threshold = measure(model, roles, args.device)
    best = copy.deepcopy(baseline)
    best_epoch = stale = 0
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    dataset = IdentityDataset(roles["train"])
    if len(dataset) < args.batch_size:
        parser.error("Training set is smaller than batch-size; reduce --batch-size")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    head = nn.Linear(512, len(dataset.labels), bias=False).to(args.device)
    clean = encode(model, roles["train"], args.device)
    identities = np.array([row["identity"] for row in roles["train"]])
    with torch.no_grad():
        for identity, label in dataset.labels.items():
            mean = torch.tensor(
                clean[identities == identity].mean(0), device=args.device
            )
            head.weight[label].copy_(F.normalize(mean, dim=0))
    model.requires_grad_(False)
    trainable = list(model.body[-3:].parameters()) + list(
        model.output_layer.parameters()
    )
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [{"params": trainable, "lr": 1e-5}, {"params": head.parameters(), "lr": 3e-4}],
        weight_decay=1e-4,
    )
    epochs = 2 if args.smoke else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_path = args.output / "best.pth"
    torch.save(model.state_dict(), best_path)
    history = []
    for epoch in range(1, epochs + 1):
        model.eval()  # Keep running BN statistics fixed on the small identity dataset.
        head.train()
        losses = []
        for step, (images, labels) in enumerate(loader):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            if epoch <= 2 and not args.smoke:
                with torch.no_grad():
                    embeddings, _ = model(images)
            else:
                embeddings, _ = model(images)
            logits = 32 * F.linear(embeddings, F.normalize(head.weight, dim=1))
            with torch.no_grad():
                target, _ = teacher(images)
            loss = (
                F.cross_entropy(logits, labels)
                + 0.5 * (1 - (embeddings * target).sum(1)).mean()
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable + list(head.parameters()), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
            if args.smoke and step == 1:
                break
        measured, threshold = measure(model, roles, args.device)
        improved = selection_score(measured) > selection_score(best) + 1e-5
        if improved:
            best, best_epoch, best_threshold, stale = measured, epoch, threshold, 0
            torch.save(model.state_dict(), best_path)
        elif epoch > 2:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "assessment": measured,
                "improved": improved,
            }
        )
        write(args.output / "history.json", history)
        print(
            f"Epoch {epoch}: loss={history[-1]['loss']:.5f}, assessment={selection_score(measured):.4f}",
            flush=True,
        )
        scheduler.step()
        if stale >= args.patience:
            break
    model.load_state_dict(
        torch.load(best_path, map_location=args.device, weights_only=True), strict=True
    )
    replay, _ = measure(model, roles, args.device)
    if abs(selection_score(replay) - selection_score(best)) > 1e-6:
        raise ValueError("Reloaded model did not reproduce selection metrics")
    selected = {
        **config,
        "weights": str(best_path.resolve()),
        "weights_format": "state_dict",
        "encoder_sha256": digest(best_path),
        "threshold": best_threshold,
        "threshold_source": "Independent calibration families from the supplied split manifest",
    }
    write(args.output / "selected_model.json", selected)
    result = {
        "smoke": args.smoke,
        "best_epoch": best_epoch,
        "baseline": baseline,
        "selected": best,
        "manifest_sha256": digest(args.manifest),
        "threshold": best_threshold,
    }
    if not args.smoke:
        test = encode(model, roles["test"], args.device)
        result["test"] = evaluate(roles["test"], test, best_threshold)
    write(args.output / "results.json", result)
    print(f"Saved encoder and inference configuration to {args.output}")


if __name__ == "__main__":
    main()
