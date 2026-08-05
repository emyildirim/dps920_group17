"""Download and extract the team driving dataset from Google Drive."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import gdown

FILE_ID = "18eGWZ25Gu00CrdBEzYPy1bDvhWF2gn0L"
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "dataset"
ZIP_PATH = REPO_ROOT / "dataset.zip"


def find_driving_log(root: Path) -> Path | None:
    matches = list(root.rglob("driving_log.csv"))
    return matches[0] if matches else None


def normalize_layout(extracted_root: Path) -> None:
    """Ensure dataset/driving_log.csv and dataset/IMG exist at the top level."""
    log = find_driving_log(extracted_root)
    if log is None:
        raise FileNotFoundError("driving_log.csv was not found inside the downloaded archive.")

    source_dir = log.parent
    target_log = DATASET_DIR / "driving_log.csv"
    target_img = DATASET_DIR / "IMG"

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if source_dir.resolve() != DATASET_DIR.resolve():
        if target_log.exists():
            target_log.unlink()
        shutil.copy2(log, target_log)

        source_img = source_dir / "IMG"
        if not source_img.is_dir():
            raise FileNotFoundError(f"IMG folder missing next to {log}")
        if target_img.exists():
            shutil.rmtree(target_img)
        shutil.copytree(source_img, target_img)

        # Remove nested extract folder if it is not the final dataset dir.
        if extracted_root.resolve() != DATASET_DIR.resolve() and extracted_root.exists():
            shutil.rmtree(extracted_root, ignore_errors=True)


def main() -> int:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    existing = find_driving_log(DATASET_DIR)
    if existing is not None and (existing.parent / "IMG").is_dir():
        print(f"Dataset already present: {existing}")
        return 0

    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print(f"Downloading dataset to {ZIP_PATH} ...")
    gdown.download(url, str(ZIP_PATH), quiet=False)

    extract_tmp = DATASET_DIR / "_extract"
    if extract_tmp.exists():
        shutil.rmtree(extract_tmp)
    extract_tmp.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        archive.extractall(extract_tmp)

    normalize_layout(extract_tmp)
    print(f"Ready: {DATASET_DIR / 'driving_log.csv'}")
    print(f"Images: {DATASET_DIR / 'IMG'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
