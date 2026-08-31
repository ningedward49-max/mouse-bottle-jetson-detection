from __future__ import annotations

import importlib.util
from pathlib import Path


NODE_PATH = Path(__file__).parents[1] / "src" / "ros2_yolo_node.py"


def load_node_module():
    spec = importlib.util.spec_from_file_location("ros2_yolo_node", NODE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_supports_teammate_compatible_command_line():
    node = load_node_module()

    args = node.parse_args(
        [
            "--model",
            "nb/model/best.pt",
            "--camera",
            "0",
            "--conf",
            "0.8",
            "--imgsz",
            "416",
            "--width",
            "640",
            "--height",
            "480",
            "--fps",
            "30",
            "--device",
            "0",
        ]
    )

    assert args.model == "nb/model/best.pt"
    assert args.camera == 0
    assert args.confidence == 0.8
    assert args.imgsz == 416
    assert args.width == 640
    assert args.height == 480
    assert args.fps == 30
    assert args.device == "0"
