# NB Mouse and Bottle Vision

An individual YOLOv8 and ROS2 object-detection experiment for two desktop-object classes:

| ID | Class |
| --- | --- |
| 0 | mouse |
| 1 | bottle |

The dataset uses the student's own photographs. The model is trained on Windows with an NVIDIA GPU and deployed to Jetson. The Jetson ROS2 node is named `nb_mouse_bottle_detector` and publishes `/nb/objects`.

## Project stages

1. [Label your own images](docs/labeling-workflow.md) using automatic pre-labels followed by complete human review.
2. Validate and split the reviewed YOLO data set.
3. Train the student-owned model on Windows NVIDIA GPU.
4. Deploy the model to Jetson and publish detections to ROS2.
5. Complete 20 real trials, calculate accuracy and record error cases.

## Data status

Fill this table only after the reviewed data set has been split:

| Split | Images | Mouse boxes | Bottle boxes |
| --- | ---: | ---: | ---: |
| Train | pending real split | pending real count | pending real count |
| Validation | pending real split | pending real count | pending real count |
| Test | pending real split | pending real count | pending real count |

Raw photos, generated labels, dataset copies, models, virtual environments and videos are ignored by Git. Keep them locally as assignment deliverables.
