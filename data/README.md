# Data review checklist

Before training, every reviewed image must have a same-name YOLO `.txt` file.

- Keep class order fixed: `0=mouse`, `1=bottle`.
- Give every visible mouse and bottle its own tight rectangular bounding box.
- Delete false automatic detections, correct wrong classes and add every missed object.
- An image with no target must use an intentionally empty `.txt` file.
- Keep near-duplicate burst photos in the same train/validation/test split.
- Run the audit command before splitting; do not train while it reports problems.
