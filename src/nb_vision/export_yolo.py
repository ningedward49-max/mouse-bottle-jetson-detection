import json
from pathlib import Path

IDS = {"mouse": 0, "bottle": 1}
raw = Path("raw")
out = Path("labeled/final_labels")
out.mkdir(parents=True, exist_ok=True)
for annotation in raw.glob("*.json"):
    data = json.loads(annotation.read_text(encoding="utf-8"))
    w, h = data["imageWidth"], data["imageHeight"]
    lines = []
    for shape in data["shapes"]:
        if shape["label"] not in IDS or shape["shape_type"] != "rectangle":
            continue
        xs = [point[0] for point in shape["points"]]
        ys = [point[1] for point in shape["points"]]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        lines.append(f"{IDS[shape['label']]} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {abs(x2-x1)/w:.6f} {abs(y2-y1)/h:.6f}")
    (out / f"{annotation.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
