"""GPU pre-labeling helpers for the fixed project classes."""


def coco_to_project_class(name: str) -> int | None:
    """Map COCO class names to this project's fixed YOLO class IDs."""
    return {"mouse": 0, "bottle": 1}.get(name)


def write_yolo_labels(result, output_path):
    """Write only mouse/bottle detections from one Ultralytics result."""
    h, w = result.orig_shape
    lines = []
    for box in result.boxes:
        name = result.names[int(box.cls.item())]
        class_id = coco_to_project_class(name)
        if class_id is None:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        lines.append(f"{class_id} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from ultralytics import YOLO
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.pt")
    for result in model(args.images, device=0, conf=0.25, stream=True, verbose=False):
        write_yolo_labels(result, args.output / f"{Path(result.path).stem}.txt")
