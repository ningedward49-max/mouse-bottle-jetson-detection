import json
from pathlib import Path
from PIL import Image

NAMES = ("mouse", "bottle")

raw = Path("raw")
labels = Path("labeled/labels")
for image in raw.glob("*.jpg"):
    w, h = Image.open(image).size
    shapes = []
    label_file = labels / f"{image.stem}.txt"
    for line in label_file.read_text(encoding="utf-8").splitlines():
        cls, x, y, bw, bh = map(float, line.split())
        x1, y1 = (x - bw / 2) * w, (y - bh / 2) * h
        x2, y2 = (x + bw / 2) * w, (y + bh / 2) * h
        shapes.append({"label": NAMES[int(cls)], "points": [[x1, y1], [x2, y2]], "group_id": None, "description": "", "shape_type": "rectangle", "flags": {}, "mask": None})
    payload = {"version": "4.0.2", "flags": {}, "shapes": shapes, "imagePath": image.name, "imageData": None, "imageHeight": h, "imageWidth": w}
    (raw / f"{image.stem}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
