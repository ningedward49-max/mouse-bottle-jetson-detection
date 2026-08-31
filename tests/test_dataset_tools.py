from pathlib import Path

from nb_vision.dataset_tools import audit_labels, split_pairs


def test_audit_reports_missing_and_invalid_labels(tmp_path: Path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "a.jpg").write_bytes(b"image")
    (images / "b.jpg").write_bytes(b"image")
    (labels / "a.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.2 0.2\n",
        encoding="utf-8",
    )

    problems = audit_labels(images, labels)

    assert any("a.txt:2" in problem for problem in problems)
    assert any("missing label: b.jpg" in problem for problem in problems)


def test_split_keeps_image_and_label_together(tmp_path: Path):
    source, output = tmp_path / "source", tmp_path / "dataset"
    source.mkdir()
    for index in range(10):
        (source / f"{index}.jpg").write_bytes(b"image")
        (source / f"{index}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )

    counts = split_pairs(source, output, seed=49)

    assert counts == {"train": 8, "val": 1, "test": 1}
    assert sum(1 for _ in (output / "images").glob("**/*.jpg")) == 10
    assert sum(1 for _ in (output / "labels").glob("**/*.txt")) == 10
