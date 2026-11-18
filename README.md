# Twin Face Identification

Face verification and gallery identification for highly similar identities, with controlled tests for blur, JPEG compression, and illumination changes.

**Mini-project · LNMIIT Jaipur · Communication and Computer Engineering**

**Team:** Ananya Surana and Samarth Gupta

**Guide:** Dr. Gaurav Verma

The application aligns a face photograph, extracts a normalized 512-dimensional AdaFace embedding, and compares it using cosine similarity. It supports:

- **Verification:** compare two photographs and predict `same_identity` or `different_identity`.
- **Identification:** match a probe to a named gallery with one to three reference photographs per person.
- **Robustness evaluation:** measure verification and identification under six degradation types at five severity levels, including an own-twin impostor protocol.
- **Optional fine-tuning:** train on family-disjoint data, calibrate on separate families, and select the encoder using an independent assessment split.

The default encoder is the public **AdaFace IR-50 WebFace4M** pretrained model. Running the application does not require training, a DGX server, private project files, or a research dataset.

## Quick start

Use Python **3.11–3.13** and Git. A CPU is sufficient for inference; a supported GPU is useful for training. Allow approximately 3 GB for the environment and downloaded model, with additional space for your photographs and training outputs. Internet access is required for installation and the initial model download; inference then works offline.

```bash
git clone https://github.com/samarthh22/twin-face-identification.git
cd twin-face-identification
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Install and download the models:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_models.py
```

The downloader retrieves the encoder (approximately 597 MB) and YuNet face detector from their public publishers and checks their SHA-256 hashes. Interrupted downloads are retried; rerunning setup skips verified files. Model binaries are stored locally in `checkpoints/` and are not committed to Git.

If PowerShell blocks activation, run `.venv\Scripts\python.exe` in place of `python` for the remaining commands.

## Compare two faces

Place two clear photographs in a local `photos/` folder, with one face per photograph:

```bash
python recognize_general.py photos/person_a.jpg photos/person_b.jpg
```

The command detects and aligns each face automatically and prints JSON containing `cosine_similarity`, `threshold`, and `prediction`. A similarity score is **not a probability**. Images without a detectable face, or with multiple faces, are rejected with a clear error.

If your inputs are already landmark-aligned 112×112 face crops:

```bash
python recognize_general.py photos/aligned_a.png photos/aligned_b.png --aligned
```

Use `--device cuda` with a CUDA-enabled PyTorch installation, or `--device mps` on supported Apple hardware. CPU is the default. Use `--describe` to inspect the model configuration without loading weights.

## Identify a person

Create `photos/gallery.json` with one to three reference photographs per identity. Image paths are resolved relative to the gallery JSON file:

```json
{
  "Person A": ["person_a_1.jpg", "person_a_2.jpg"],
  "Person B": ["person_b_1.jpg"]
}
```

```bash
python recognize_general.py photos/probe.jpg --gallery photos/gallery.json
```

Identification returns the closest enrolled identity. This is **closed-set identification**: it assumes the probe belongs to somebody in the gallery and does not reject unknown people. Use distinct photographs for enrollment and testing.

## Align a dataset

The alignment command preserves the input directory structure and writes lossless 112×112 PNG crops into a separate output directory:

```bash
python -m preprocessing.align --input data/raw --output data/aligned
```

Use an empty output directory. Review any failed detections before training or evaluation. Do not merely resize unaligned photographs and pass them as aligned inputs.

## Train or evaluate on your own data

Research datasets are not bundled. Use appropriately obtained datasets or your own photographs. No dataset is needed for the inference commands above. Reproducing a benchmark's exact numbers requires the same images, alignment, split manifest, and model; a newly generated split is a new experiment.

### General identities and optional fine-tuning

Arrange aligned photographs in this structure (the `FIDs` directory name is required):

```text
data/aligned/FIDs/
  family_001/
    person_001/
      P001.png
      P002.png
    person_002/
      P003.png
      P004.png
  family_002/
    ...
```

Keep all related identities in the same family folder. Use at least ten families and multiple distinct photographs per identity. Filenames sharing the prefix before the first underscore are treated as crops from the same source photograph: use names such as `P001_face1.png` and `P002_face1.png` for different source photographs. For simple independent images, `001.png`, `002.png`, etc. also work.

```bash
python scripts/prepare_splits.py --data-root data/aligned --output data/splits.json
python scripts/train.py --data-root data/aligned --manifest data/splits.json --output runs/training --device cpu
```

The training command uses an identity classifier with a feature-preservation loss, a two-epoch classifier warm-up, and early stopping. The pretrained encoder remains selected unless assessment performance improves. Validation families are split into calibration and assessment roles; the test families are scored after model selection. Exact pixel duplicates across identities or splits are rejected.

For a short integration check, add `--smoke --batch-size 4` and use a fresh output directory. Smoke mode does not score the test set. Full fine-tuning is substantially faster on a GPU. Each run exports `best.pth`, `selected_model.json`, and local JSON metrics inside its output folder.

Use a trained encoder for inference:

```bash
python recognize_general.py photos/person_a.jpg photos/person_b.jpg --config runs/training/selected_model.json
```

Evaluate the public pretrained model on test families:

```bash
python scripts/evaluate.py --protocol general --data-root data/aligned --manifest data/splits.json --output runs/general.json
```

Pass `--config runs/training/selected_model.json` to evaluate a trained model. Add `--degradations` for the complete robustness sweep.

### Twin evaluation

Arrange aligned crops as `data/twins/<twin_pair>/<identity>/<photo>.png`, with exactly two identities per twin-pair folder and at least four distinct photographs per identity:

```bash
python scripts/evaluate.py --protocol twins --data-root data/twins --output runs/twins.json --degradations
```

This samples up to ten genuine pairs per identity and twenty own-twin impostor pairs per twin set, using seed 42. It keeps the reference images clean and degrades only probe images. The clean verification threshold remains fixed throughout evaluation. Results include balanced accuracy, false acceptance/rejection, ROC AUC, EER, and gallery identification. Low-FAR metrics are left unsupported when there are too few negative pairs.

## Project layout

```text
recognize_general.py     Verification and gallery identification CLI
configs/                Portable inference configuration
models/                 AdaFace backbone, verifier, and third-party license notices
preprocessing/          Face detection and alignment
degradation/            Deterministic blur, JPEG, and illumination simulation
training/               Dataset handling and split validation
evaluation/             Calibration, verification, identification, and robustness metrics
scripts/                Model download, split preparation, training, and evaluation
utils/                  Verified checkpoint loading
tests/                  Runtime and protocol regression tests
```

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Unit tests run without downloading models or obtaining datasets. Real inference additionally requires the setup download and face photographs.

## Interpretation and attribution

The bundled threshold was calibrated on clean FIW family data. Accuracy on a new camera, alignment method, demographic group, or degraded image distribution may differ. In particular, visually similar twins can produce high false acceptance rates. This model predicts identity similarity; it does not establish whether people are biologically related. Calibrate thresholds on independent development data for a new dataset, never on its test labels.

The backbone implementation and pretrained model come from [AdaFace](https://github.com/mk-minchul/AdaFace), by Minchul Kim, Anil K. Jain, and Xiaoming Liu (CVPR 2022). Its MIT notice is preserved in `models/ADAFACE_LICENSE.txt`. Input normalization follows the publisher's BGR convention.

Face detection uses [OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet). See `models/YUNET_LICENSE.txt` for its MIT notice. The local application uses YuNet landmark alignment; the original AdaFace examples use MTCNN, so alignment-dependent benchmark results should not be assumed identical.

Only this README is maintained as project documentation. Downloaded weights, private images, generated metrics, caches, and local environments are excluded through `.gitignore`.
