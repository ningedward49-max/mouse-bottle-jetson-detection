# NB Mouse and Bottle Detection: Design

## Goal

Build an individual object-detection project for two desktop-object classes:

- `0: mouse`
- `1: bottle`

The completed system will train on the student's own photographs, run in real time on an NVIDIA Jetson, display bounding boxes, class names, confidence scores and end-to-end FPS, and publish current detections through ROS2.

## Project identity

| Item | Value |
| --- | --- |
| GitHub repository | `nb-mouse-bottle-vision` |
| Jetson project directory | `~/nb_mouse_bottle_vision` |
| ROS2 node | `nb_mouse_bottle_detector` |
| ROS2 topic | `/nb/objects` |
| Detection classes | `mouse`, `bottle` |

## Data workflow

The source images are the student's own 100–200+ photos, including individual objects, both objects together and partial occlusions. A third-party MIT-licensed utility may generate initial labels using COCO pre-trained YOLO weights, but every generated label must be reviewed manually in LabelImg.

Each visible mouse or bottle receives a separate rectangle. Labels use YOLO detection format and keep the fixed class order `mouse`, then `bottle`. The dataset is split approximately 80% train, 10% validation and 10% test. Near-duplicate images from a single burst stay in the same split to avoid leakage.

## Repository scope

The repository stores project source code, configuration, documentation, data-set statistics, training configuration and real test records. It ignores raw image collections, generated datasets, model weights, virtual environments, videos and other large binaries by default. These artifacts are retained locally for submission and can be shared separately if required.

The project records meaningful milestones as Git commits: repository setup, labelling configuration, dataset validation, training configuration, ROS2 detector, and experiment records.

## Training workflow

Training runs on the student's Windows computer using its NVIDIA GPU and Ultralytics YOLOv8n. Training begins from `yolov8n.pt` and uses the student's reviewed two-class data set. The selected output is the student's own `best.pt`, accompanied by training metrics and a validation summary.

## Jetson runtime

The Jetson runtime loads the copied model, opens a configurable camera, runs inference per frame, draws YOLO detections and computes FPS from full frame-loop timing. It publishes JSON over `std_msgs/String` to `/nb/objects` with this shape:

```json
{
  "speed_fps": 12.4,
  "objects": [
    {"name": "mouse", "score": 0.91, "xyxy": [80, 95, 313, 238]}
  ]
}
```

It reports clear errors for missing model files, unavailable cameras and unexpected ROS2/YOLO startup failures. The chosen confidence threshold is configurable, allowing tuning only after real validation.

## Verification and evidence

Before deployment, scripts validate class IDs, normalized bounding-box values, image-label pairing and split counts. The node is syntax-checked and has focused non-hardware tests for message construction and configuration. On Jetson, verification covers camera output, visible detections, displayed FPS of at least 5, ROS2 topic discovery and `ros2 topic echo` data.

The final experiment log records 20 real trials across both classes with actual/predicted classes, confidence, FPS, correct/incorrect result and notes. It calculates accuracy from those records and keeps representative error cases for the report.

## Out of scope

This project does not reuse another student's dataset or trained weights, alter shared Jetson system packages, or commit virtual environments, CUDA/ROS installations or large generated artifacts.
