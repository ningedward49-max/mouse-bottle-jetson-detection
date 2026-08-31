#!/usr/bin/env python3
"""Run the NB mouse-and-bottle YOLO model and publish detections over ROS2.

The command-line arguments intentionally match the teammate's existing
``ros2_yolo_node.py`` invocation so the model can be swapped without changing
the camera setup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to YOLO .pt model")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--conf", dest="confidence", type=float, default=0.8)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="0", help="YOLO device, normally 0 on Jetson")
    parser.add_argument("--topic", default="/yolo_detections")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # These imports stay here so --help and argument parsing also work on the
    # Windows training computer, which intentionally does not have ROS2.
    import cv2
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from ultralytics import YOLO

    class NbMouseBottleYoloNode(Node):
        def __init__(self) -> None:
            super().__init__("nb_mouse_bottle_yolo_node")
            self.publisher = self.create_publisher(String, args.topic, 10)
            self.model = YOLO(str(model_path))
            self.camera = cv2.VideoCapture(args.camera)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            self.camera.set(cv2.CAP_PROP_FPS, args.fps)
            if not self.camera.isOpened():
                raise RuntimeError(f"Cannot open camera index {args.camera}")
            self.timer = self.create_timer(1.0 / args.fps, self.detect_and_publish)
            self.get_logger().info(
                f"YOLO node started. Publishing to {args.topic}; "
                f"camera={args.camera}, {args.width}x{args.height}@{args.fps} FPS"
            )

        def detect_and_publish(self) -> None:
            ok, frame = self.camera.read()
            if not ok:
                self.get_logger().warning("Camera frame read failed")
                return
            result = self.model(
                frame, conf=args.confidence, imgsz=args.imgsz, device=args.device, verbose=False
            )[0]
            detections: list[dict[str, object]] = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": result.names[class_id],
                            "confidence": round(float(box.conf[0].item()), 4),
                            "bbox_xyxy": [round(float(value), 1) for value in box.xyxy[0].tolist()],
                        }
                    )
            self.publisher.publish(String(data=json.dumps({"detections": detections})))
            cv2.imshow("NB mouse and bottle detection", result.plot())
            cv2.waitKey(1)

        def close(self) -> None:
            self.camera.release()
            cv2.destroyAllWindows()

    rclpy.init(args=None)
    node = NbMouseBottleYoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
