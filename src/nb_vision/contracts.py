"""Shared data contracts for labels and ROS2 detection messages."""

import json


CLASS_NAMES = ("mouse", "bottle")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


def validate_yolo_line(line: str) -> tuple[int, float, float, float, float]:
    """Parse and validate one normalized YOLO detection-label line."""
    values = line.split()
    if len(values) != 5:
        raise ValueError("YOLO label needs five values")

    try:
        class_id = int(values[0])
        x, y, width, height = (float(value) for value in values[1:])
    except ValueError as error:
        raise ValueError("YOLO label contains a non-numeric value") from error

    if class_id not in range(len(CLASS_NAMES)):
        raise ValueError("invalid class id")
    if not all(0.0 <= value <= 1.0 for value in (x, y, width, height)):
        raise ValueError("coordinates must be normalized")
    if width == 0.0 or height == 0.0:
        raise ValueError("box size must be positive")

    return class_id, x, y, width, height


def detection_packet(fps: float, objects: list[dict]) -> str:
    """Return the fixed JSON payload sent through the ROS2 topic."""
    return json.dumps({"speed_fps": round(fps, 1), "objects": objects})
