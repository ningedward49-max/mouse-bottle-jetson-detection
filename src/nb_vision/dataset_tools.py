"""Audit reviewed YOLO labels and create deterministic dataset splits."""

import argparse
import random
import shutil
from pathlib import Path

from nb_vision.contracts import validate_yolo_line


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLIT_NAMES = ("train", "val", "test")


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def audit_labels(image_dir: Path, label_dir: Path) -> list[str]:
    """Return all missing, unmatched, or invalid-label problems without changing data."""
    problems: list[str] = []
    image_by_stem = {path.stem: path for path in _image_files(image_dir)}
    label_files = sorted(label_dir.glob("*.txt"))

    for image in image_by_stem.values():
        if image.stem not in {label.stem for label in label_files}:
            problems.append(f"missing label: {image.name}")

    for label in label_files:
        if label.stem not in image_by_stem:
            problems.append(f"missing image: {label.name}")
            continue
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                validate_yolo_line(line)
            except ValueError as error:
                problems.append(f"{label.name}:{line_number}: {error}")

    return problems


def _paired_files(input_dir: Path) -> list[tuple[Path, Path]]:
    images = _image_files(input_dir)
    labels = {path.stem: path for path in input_dir.glob("*.txt")}
    image_stems = {path.stem for path in images}

    missing_labels = [image.name for image in images if image.stem not in labels]
    missing_images = [label.name for stem, label in labels.items() if stem not in image_stems]
    if missing_labels or missing_images:
        details = [*(f"missing label: {name}" for name in missing_labels), *(f"missing image: {name}" for name in missing_images)]
        raise ValueError("; ".join(details))
    return [(image, labels[image.stem]) for image in images]


def _split_sizes(total: int) -> dict[str, int]:
    if total < 10:
        raise ValueError("at least 10 reviewed image-label pairs are required")
    val_count = max(1, round(total * 0.1))
    test_count = max(1, round(total * 0.1))
    return {"train": total - val_count - test_count, "val": val_count, "test": test_count}


def split_pairs(input_dir: Path, output_dir: Path, seed: int = 49) -> dict[str, int]:
    """Copy reviewed image-label pairs to an 80/10/10 YOLO dataset split."""
    pairs = _paired_files(input_dir)
    sizes = _split_sizes(len(pairs))
    random.Random(seed).shuffle(pairs)

    start = 0
    for split in SPLIT_NAMES:
        end = start + sizes[split]
        image_output = output_dir / "images" / split
        label_output = output_dir / "labels" / split
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)
        for image, label in pairs[start:end]:
            shutil.copy2(image, image_output / image.name)
            shutil.copy2(label, label_output / label.name)
        start = end
    return sizes


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--images", type=Path, required=True)
    audit.add_argument("--labels", type=Path, required=True)
    split = commands.add_parser("split")
    split.add_argument("--input", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=49)
    args = parser.parse_args()

    if args.command == "audit":
        problems = audit_labels(args.images, args.labels)
        if problems:
            raise SystemExit("\n".join(problems))
        print("Label audit passed")
    else:
        print(split_pairs(args.input, args.output, args.seed))


if __name__ == "__main__":
    _main()
