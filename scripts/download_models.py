"""Download public models with SHA-256 verification and atomic installation."""

import argparse
import hashlib
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "encoder": {
        "path": "checkpoints/adaface_ir50_webface4m.ckpt",
        "url": "https://drive.usercontent.google.com/download?id=1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN&export=download&confirm=t",
        "sha256": "52cca7c64808fea6f44f9b9aee2b0e091bf96c1ab4f6e31bedcdf5d77009b4f8",
    },
    "detector": {
        "path": "checkpoints/face_detection_yunet_2023mar.onnx",
        "url": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    },
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_model(name):
    spec = MODELS[name]
    target = ROOT / spec["path"]
    if target.exists() and sha256(target) == spec["sha256"]:
        print(f"Already verified: {target.name}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {name} from its public publisher...", flush=True)
    for attempt in range(4):
        try:
            size = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "face-identification-setup"}
            if size:
                headers["Range"] = f"bytes={size}-"
            request = urllib.request.Request(spec["url"], headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                resume = size > 0 and response.status == 206
                if resume and not response.headers.get("Content-Range", "").startswith(
                    f"bytes {size}-"
                ):
                    raise ValueError(
                        "Download server returned an unexpected byte range"
                    )
                if not resume:
                    size = 0
                with partial.open("ab" if resume else "wb") as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                        size += len(block)
                        if size % (50 * 1024 * 1024) == 0:
                            print(f"  {size // (1024 * 1024)} MiB", flush=True)
            if sha256(partial) != spec["sha256"]:
                partial.unlink(missing_ok=True)
                raise ValueError(
                    "Downloaded file failed SHA-256 verification; the host may have returned an error page."
                )
            partial.replace(target)
            break
        except OSError:
            if attempt == 3:
                raise
            print(f"Connection interrupted; retrying ({attempt + 1}/3)...", flush=True)
            time.sleep(2)
    print(f"Verified: {target.name}")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=MODELS)
    args = parser.parse_args()
    for name in [args.only] if args.only else MODELS:
        download_model(name)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f"Model setup failed: {error}. Check your connection and retry.")
