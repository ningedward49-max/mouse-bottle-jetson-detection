# Mouse and Bottle Object Detection on Jetson

An individual object-detection experiment for **Group Experiment 1: Object Detection and Recognition**. The project detects two desktop-object classes, **mouse** and **bottle**, using a custom YOLOv8n model trained on self-collected images and deployed on a Jetson platform with ROS 2 publication.

## Project overview

The complete workflow was implemented from data collection to embedded deployment:

1. Capture desktop images containing a mouse, a bottle, or both objects under different viewpoints, lighting conditions, distances, backgrounds, and partial occlusions.
2. Review bounding-box annotations in X-AnyLabeling and export them to YOLO format.
3. Validate the dataset and split it into training, validation, and test subsets.
4. Train YOLOv8n on a Windows computer with an NVIDIA GPU.
5. Deploy the final `best.pt` model to Jetson for live USB-camera inference.
6. Display bounding boxes, class names, confidence values, and FPS, then publish the detections through ROS 2.

| Class ID | Class name | Description |
| ---: | --- | --- |
| 0 | `mouse` | Computer mouse visible on the desktop |
| 1 | `bottle` | Water bottle visible on the desktop |

## Dataset status

The final dataset contains **168 self-collected images** and **206 reviewed bounding boxes**. Each image/label pair is stored in the standard YOLO directory structure.

| Split | Images | Mouse boxes | Bottle boxes | Total boxes |
| --- | ---: | ---: | ---: | ---: |
| Train | 134 | 88 | 70 | 158 |
| Validation | 17 | 12 | 12 | 24 |
| Test | 17 | 14 | 10 | 24 |
| **Total** | **168** | **114** | **92** | **206** |

The class order is fixed in [`config/data.yaml`](config/data.yaml): `0 = mouse`, `1 = bottle`.

## Training

YOLOv8n was selected because it is compact enough for real-time embedded inference while still providing strong accuracy for this two-class task. Training used the pretrained `yolov8n.pt` checkpoint as the initial model.

**Training environment**

- Windows development computer with an NVIDIA RTX 4060 GPU
- YOLOv8n detector
- 100 epochs, image size 640, batch size 8, GPU device 0

```bash
yolo detect train model=yolov8n.pt data=config/data.yaml epochs=100 imgsz=640 batch=8 device=0 project=runs name=nb_mouse_bottle exist_ok=True
```

**Final validation metrics**

| Metric | Result |
| --- | ---: |
| Precision | 99.1% |
| Recall | 91.7% |
| mAP@0.5 | 91.5% |
| mAP@0.5:0.95 | 81.0% |

Training curves, confusion matrices, and validation prediction images are produced in `runs/detect/runs/nb_mouse_bottle/`.

## Jetson and ROS 2 inference

The deployment entry point is [`src/ros2_yolo_node.py`](src/ros2_yolo_node.py). It opens the USB camera, runs YOLO inference, displays the annotated image, and publishes a JSON string message on the ROS 2 topic **`/yolo_detections`**.

- ROS 2 node name: `nb_mouse_bottle_yolo_node`
- Message type: `std_msgs/String`
- Published fields for each detection: `class_id`, `class_name`, `confidence`, and `bbox_xyxy`

Example payload:

```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "mouse",
      "confidence": 0.97,
      "bbox_xyxy": [120.5, 84.0, 416.2, 340.8]
    }
  ]
}
```

For the prepared Jetson environment, use:

```bash
source ~/qyy_env/bin/activate
python ~/nb/src/ros2_yolo_node.py --model ~/nb/model/best.pt --camera 0 --conf 0.8 --imgsz 416 --width 640 --height 480 --fps 30 --device 0
```

The inference size is reduced from the 640-pixel training size to `--imgsz 416` on Jetson to improve real-time speed. The confidence threshold is set to `0.8` to reduce weak detections; this is a deliberate precision-recall trade-off.

See [Jetson deployment instructions](docs/jetson-deployment.md) for the packaged deployment workflow.

## On-device evaluation

The formal assignment acceptance count used 20 individual object presentations:

| Test measure | Result | Requirement | Status |
| --- | ---: | ---: | --- |
| Object-level recognition accuracy | 17 / 20 = **85.0%** | >= 80% | Passed |
| Lowest recorded displayed Jetson FPS | **16.8 FPS** | >= 5 FPS | Passed |
| Recorded frame-level accuracy | 22 / 25 = **88.0%** | Supplementary system measure | Recorded |

The saved test evidence includes 25 on-device screenshots, a live detection video, and three preserved missed-detection cases. The missed cases were two white-mouse examples and one black-mouse example; they are retained for error analysis rather than excluded from the result record.

## Repository structure

```text
config/                 YOLO data configuration
data/                   Data-processing notes and intermediate metadata
docs/                   Annotation and Jetson deployment instructions
src/                    ROS 2 node and self-written data tools
  nb_vision/            Annotation conversion, validation, and export utilities
tests/                  Lightweight checks for project code
runs/                   Local training outputs (generated during training)
```

Key self-written files:

- `src/ros2_yolo_node.py` - Jetson camera inference and ROS 2 publisher.
- `src/nb_vision/xany_converter.py` - X-AnyLabeling JSON to YOLO conversion.
- `src/nb_vision/dataset_tools.py` - Dataset validation and splitting support.
- `src/nb_vision/export_yolo.py` - Final YOLO dataset export.
- `src/nb_vision/contracts.py` - Shared class-ID contract.

## Documentation

- [Annotation workflow](docs/labeling-workflow.md)
- [Jetson deployment guide](docs/jetson-deployment.md)
- [Windows requirements](requirements-windows.txt)

