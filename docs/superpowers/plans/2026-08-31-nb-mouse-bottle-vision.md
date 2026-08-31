# NB Mouse and Bottle Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, student-owned mouse-and-bottle detection workflow from reviewed labels through Jetson ROS2 runtime and experiment evidence.

**Architecture:** A small Python package owns the class contract, YOLO-label validation/splitting, experiment-result calculations, and ROS message construction. A Jetson entry-point imports that package and combines camera capture, Ultralytics inference, overlay drawing and a ROS2 publisher. The upstream MIT labeling utility is only used as a separately attributed pre-labeling tool; student data, labels, model and runtime node remain this project’s work.

**Tech Stack:** Python 3.10+, pytest, Ultralytics YOLOv8, OpenCV, ROS2 Humble (`rclpy`, `std_msgs`), NVIDIA CUDA on Windows/Jetson, Git and GitHub.

**Spec:** `docs/superpowers/specs/2026-08-31-nb-mouse-bottle-vision-design.md`

## Global Constraints

- Use only student-captured mouse and bottle photographs and the student-trained `best.pt`; do not use another student’s data or weights.
- Keep the exact class mapping `0: mouse`, `1: bottle` everywhere.
- Use project identity `nb`: Jetson directory `~/nb_mouse_bottle_vision`, ROS node `nb_mouse_bottle_detector`, ROS2 topic `/nb/objects`.
- The repository must not commit raw images, generated datasets, model weights, videos, virtual environments or CUDA/ROS installations.
- Pre-labels from the MIT-licensed upstream tool must be manually reviewed before training; label each visible object separately.
- Record 20 real test trials and do not manufacture accuracy, FPS or error results.
- Do not remove or modify shared Jetson system packages.

---

## File structure

| Path | Responsibility |
| --- | --- |
| `.gitignore` | Exclude generated/heavy/private artifacts while retaining small configs and logs. |
| `requirements-windows.txt` | Windows-side dependencies for labeling, validation and training. |
| `config/data.yaml` | Fixed YOLO class mapping and dataset paths. |
| `src/nb_vision/contracts.py` | Class constants, YOLO-label validation, detection serialization and test-log metrics. |
| `src/nb_vision/dataset_tools.py` | Deterministic split creation and dataset auditing CLI. |
| `src/nb_vision/jetson_node.py` | Jetson ROS2 camera detector and CLI configuration. |
| `tests/` | Hardware-independent tests for contracts, split/audit and runtime message creation. |
| `scripts/train_windows.ps1` | Reproducible Windows GPU training command wrapper. |
| `experiment/test_results.csv` | Student-entered 20-trial result record (not fabricated). |
| `README.md` | Setup, labeling, training, deployment and evidence instructions. |

### Task 1: Establish the repository contract

**Files:**
- Create: `.gitignore`
- Create: `requirements-windows.txt`
- Create: `config/data.yaml`
- Create: `src/nb_vision/__init__.py`
- Create: `src/nb_vision/contracts.py`
- Create: `tests/test_contracts.py`

**Interfaces:**
- Produces: `CLASS_NAMES: tuple[str, str]`, `CLASS_TO_ID: dict[str, int]`, `validate_yolo_line(line: str) -> tuple[int, float, float, float, float]`, `detection_packet(fps: float, objects: list[dict]) -> str`.
- Consumes: no project interfaces.

- [ ] **Step 1: Write the failing contract tests**

```python
from nb_vision.contracts import CLASS_NAMES, detection_packet, validate_yolo_line

def test_class_contract_is_stable():
    assert CLASS_NAMES == ("mouse", "bottle")

def test_valid_yolo_line_is_normalized_and_parsed():
    assert validate_yolo_line("1 0.5 0.2 0.1 0.3") == (1, 0.5, 0.2, 0.1, 0.3)

def test_invalid_yolo_line_is_rejected():
    try:
        validate_yolo_line("2 0.5 0.2 0.1 0.3")
    except ValueError as error:
        assert "class id" in str(error)
    else:
        raise AssertionError("invalid class must fail")

def test_detection_packet_has_required_fields():
    assert detection_packet(12.44, [{"name": "mouse", "score": 0.91, "xyxy": [1, 2, 3, 4]}]) == (
        '{"speed_fps": 12.4, "objects": [{"name": "mouse", "score": 0.91, "xyxy": [1, 2, 3, 4]}]}'
    )
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_contracts.py -v`

Expected: FAIL because `nb_vision.contracts` does not exist.

- [ ] **Step 3: Implement minimal contract code and configuration**

```python
# src/nb_vision/contracts.py
import json

CLASS_NAMES = ("mouse", "bottle")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}

def validate_yolo_line(line: str) -> tuple[int, float, float, float, float]:
    values = line.split()
    if len(values) != 5:
        raise ValueError("YOLO label needs five values")
    class_id = int(values[0])
    if class_id not in range(len(CLASS_NAMES)):
        raise ValueError("invalid class id")
    x, y, width, height = (float(value) for value in values[1:])
    if not all(0.0 <= value <= 1.0 for value in (x, y, width, height)):
        raise ValueError("coordinates must be normalized")
    if width == 0.0 or height == 0.0:
        raise ValueError("box size must be positive")
    return class_id, x, y, width, height

def detection_packet(fps: float, objects: list[dict]) -> str:
    return json.dumps({"speed_fps": round(fps, 1), "objects": objects}, ensure_ascii=False)
```

Create `config/data.yaml` with `path: ../dataset`, the `images/train`, `images/val`, `images/test` paths and names `0: mouse`, `1: bottle`. Add `ultralytics>=8.3`, `opencv-python>=4.10`, `pytest>=8.0`, and `labelImg>=1.8` to `requirements-windows.txt`. Ignore `raw/`, `labeled/`, `dataset/`, `models/`, `runs/`, `video/`, `.venv/`, `__pycache__/` and `*.pt`.

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_contracts.py -v`

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add .gitignore requirements-windows.txt config/data.yaml src/nb_vision tests/test_contracts.py
git commit -m "feat: establish two-class detection contract"
```

### Task 2: Add auditable dataset splitting and validation

**Files:**
- Create: `src/nb_vision/dataset_tools.py`
- Create: `tests/test_dataset_tools.py`
- Create: `data/README.md`

**Interfaces:**
- Consumes: `validate_yolo_line` and `CLASS_NAMES` from `nb_vision.contracts`.
- Produces: `audit_labels(image_dir: Path, label_dir: Path) -> list[str]` and `split_pairs(input_dir: Path, output_dir: Path, seed: int = 49) -> dict[str, int]`.

- [ ] **Step 1: Write failing audit/split tests**

```python
from pathlib import Path
from nb_vision.dataset_tools import audit_labels, split_pairs

def test_audit_reports_missing_and_invalid_labels(tmp_path: Path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir(); labels.mkdir()
    (images / "a.jpg").write_bytes(b"image")
    (images / "b.jpg").write_bytes(b"image")
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.2 0.2\n", encoding="utf-8")
    problems = audit_labels(images, labels)
    assert any("a.txt:2" in problem for problem in problems)
    assert any("missing label: b.jpg" in problem for problem in problems)

def test_split_keeps_image_and_label_together(tmp_path: Path):
    source, output = tmp_path / "source", tmp_path / "dataset"
    source.mkdir()
    for index in range(10):
        (source / f"{index}.jpg").write_bytes(b"image")
        (source / f"{index}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    counts = split_pairs(source, output, seed=49)
    assert counts == {"train": 8, "val": 1, "test": 1}
    assert sum(1 for path in output.glob("images/**/*.jpg")) == 10
    assert sum(1 for path in output.glob("labels/**/*.txt")) == 10
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_dataset_tools.py -v`

Expected: FAIL because `nb_vision.dataset_tools` does not exist.

- [ ] **Step 3: Implement the deterministic utility**

Implement `audit_labels` to match every `.jpg`/`.jpeg`/`.png` image to a same-stem `.txt`, validate every nonempty label line with `validate_yolo_line`, and return descriptive problems without silently correcting data. Implement `split_pairs` to shuffle sorted same-stem image/label pairs with `random.Random(seed)`, allocate 80/10/10 for 10 or more samples, and copy each pair to `images/<split>` and `labels/<split>`. Reject an unpaired source image or label with `ValueError`.

Expose a CLI:

```text
python -m nb_vision.dataset_tools audit --images labeled/images --labels labeled/labels
python -m nb_vision.dataset_tools split --input labeled/pairs --output dataset --seed 49
```

Write `data/README.md` defining the review checklist: correct class, tight box, every visible object labeled, empty image uses an empty `.txt`, and adjacent burst photos remain in one split.

- [ ] **Step 4: Run focused and full tests**

Run: `python -m pytest tests/test_dataset_tools.py -v; python -m pytest -v`

Expected: PASS; the second command includes all contract tests.

- [ ] **Step 5: Commit**

```bash
git add src/nb_vision/dataset_tools.py tests/test_dataset_tools.py data/README.md
git commit -m "feat: validate and split reviewed yolo labels"
```

### Task 3: Prepare and review the student’s labels

**Files:**
- Create: `docs/labeling-workflow.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: fixed class order in `config/data.yaml`; audit CLI from Task 2.
- Produces: `labeled/` containing reviewed image/label pairs, then a locally generated `dataset/` that passes the audit.

- [ ] **Step 1: Document the exact pre-label workflow**

Document the Windows commands below, with no image uploads required:

```powershell
git clone https://github.com/hanlinji4002/desk-object-labeling.git work/desk-object-labeling
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r work/desk-object-labeling/requirements.txt
python work/desk-object-labeling/tools/auto_label.py --images raw --out labeled --classes mouse bottle --model yolov8x.pt --conf 0.25
labelImg labeled/images
```

State that the upstream tool is MIT-licensed, only creates pre-labels, and must not substitute for a review. In LabelImg select YOLO format, load `mouse` then `bottle` as the class list, preserve `0=mouse`, `1=bottle`, and save label files beside/under the corresponding labels directory.

- [ ] **Step 2: Run the student-facing pre-label and review process**

Copy the student’s photos into ignored `raw/`; do not commit them. Run the documented pre-label command. In LabelImg, review every frame, remove false positives, correct every incorrect box/class, add missed boxes, and verify that a frame with both objects contains two annotations.

Expected: every raw image has an intentionally reviewed label file.

- [ ] **Step 3: Validate and split the reviewed data**

Run: `python -m nb_vision.dataset_tools audit --images labeled/images --labels labeled/labels`

Expected: no reported problems. Then run the split CLI and ensure dataset counts are approximately 80/10/10. Copy the resulting dataset-level counts and class totals into `README.md`; do not invent counts before completion.

- [ ] **Step 4: Commit documentation and real dataset metadata only**

```bash
git add docs/labeling-workflow.md README.md config/data.yaml
git commit -m "docs: record reviewed dataset preparation workflow"
```

### Task 4: Add reproducible Windows training and result capture

**Files:**
- Create: `scripts/train_windows.ps1`
- Create: `docs/training-workflow.md`
- Create: `experiment/training_summary.md`

**Interfaces:**
- Consumes: valid local `dataset/` and `config/data.yaml` from Tasks 2–3.
- Produces: ignored `runs/nb_mouse_bottle/weights/best.pt` plus a committed real summary of the chosen model and validation metrics.

- [ ] **Step 1: Write the training-wrapper behavior check**

Create `tests/test_training_script.py` that reads the script and asserts it contains all four exact contract values: `yolov8n.pt`, `config/data.yaml`, `epochs 80`, and `imgsz 640`.

- [ ] **Step 2: Run test to verify failure**

Run: `python -m pytest tests/test_training_script.py -v`

Expected: FAIL because the script does not exist.

- [ ] **Step 3: Implement Windows training wrapper**

Create `scripts/train_windows.ps1`:

```powershell
param([int]$Epochs = 80, [int]$ImageSize = 640, [int]$Batch = 8)
$ErrorActionPreference = "Stop"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU not available'; print(torch.cuda.get_device_name(0))"
yolo detect train model=yolov8n.pt data=config/data.yaml epochs=$Epochs imgsz=$ImageSize batch=$Batch device=0 project=runs name=nb_mouse_bottle exist_ok=True
```

Write the training document with the activation command, GPU check, run command, expected `best.pt` path and a rule to record actual `mAP50`, `precision`, `recall`, epoch count and data counts in `experiment/training_summary.md`. The summary file starts as a clearly labeled template with blank fields, not invented metrics.

- [ ] **Step 4: Verify test and run actual training**

Run: `python -m pytest tests/test_training_script.py -v`

Expected: PASS. Then run `.\scripts\train_windows.ps1`, inspect `runs/nb_mouse_bottle/weights/best.pt`, and complete only the actual values in `experiment/training_summary.md`.

- [ ] **Step 5: Commit configuration and actual summary, excluding weights**

```bash
git add scripts/train_windows.ps1 docs/training-workflow.md experiment/training_summary.md tests/test_training_script.py
git commit -m "feat: add reproducible windows training workflow"
```

### Task 5: Implement the Jetson ROS2 runtime

**Files:**
- Create: `src/nb_vision/jetson_node.py`
- Create: `tests/test_runtime_contract.py`
- Create: `requirements-jetson.txt`
- Create: `docs/jetson-deployment.md`

**Interfaces:**
- Consumes: `detection_packet` from `nb_vision.contracts`, copied local `model.pt`, a camera index, ROS2 Humble and Ultralytics.
- Produces: `CameraObjectNode` with node name `nb_mouse_bottle_detector`, topic `/nb/objects`, and a `build_object(name: str, score: float, xyxy: list[float]) -> dict` helper.

- [ ] **Step 1: Write the failing hardware-independent test**

```python
from nb_vision.jetson_node import build_object

def test_build_object_rounds_and_converts_box_coordinates():
    assert build_object("bottle", 0.876, [1.8, 2.1, 30.9, 42.0]) == {
        "name": "bottle", "score": 0.88, "xyxy": [1, 2, 30, 42]
    }
```

- [ ] **Step 2: Run test to verify failure**

Run: `python -m pytest tests/test_runtime_contract.py -v`

Expected: FAIL because `nb_vision.jetson_node` does not exist.

- [ ] **Step 3: Implement the node with explicit startup checks**

Implement `build_object` and a `CameraObjectNode(Node)` that (1) takes `--model`, `--camera-id`, `--score-limit` and `--device` parameters, (2) raises a clear `FileNotFoundError` if the model is absent, (3) opens `cv2.VideoCapture(camera_id)` and raises `RuntimeError` if it fails, (4) calls `YOLO(model)(frame, conf=score_limit, device=device, verbose=False)`, (5) emits a packet through `detection_packet`, (6) publishes `std_msgs.msg.String`, (7) draws `prediction.plot()` plus the full-loop FPS, and (8) exits on `q` while reliably releasing the camera/destroying windows/shutting down rclpy.

Use exactly:

```python
super().__init__("nb_mouse_bottle_detector")
self.output = self.create_publisher(String, "/nb/objects", 10)
```

Set `requirements-jetson.txt` to `ultralytics>=8.3` and document that Jetson torch/torchvision must come from the JetPack-compatible source rather than a generic x86 wheel.

- [ ] **Step 4: Run tests and syntax validation**

Run: `python -m pytest tests/test_runtime_contract.py -v; python -m py_compile src/nb_vision/jetson_node.py`

Expected: PASS and no syntax output. On Jetson after copying the source and `model.pt`, run `python -c "import torch, rclpy, ultralytics, cv2; print(torch.cuda.is_available())"`; expected final value is `True`.

- [ ] **Step 5: Commit**

```bash
git add src/nb_vision/jetson_node.py tests/test_runtime_contract.py requirements-jetson.txt docs/jetson-deployment.md
git commit -m "feat: add nb jetson ros2 detector"
```

### Task 6: Record real acceptance evidence and finish GitHub delivery

**Files:**
- Create: `experiment/test_results.csv`
- Create: `experiment/error_cases.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: running node from Task 5 and output from `ros2 topic echo /nb/objects`.
- Produces: real 20-test evidence with computed accuracy and documented error cases.

- [ ] **Step 1: Create an honest 20-trial log template**

Create a CSV header and exactly 20 numbered blank rows:

```csv
id,actual_class,predicted_class,confidence,fps,correct,notes
1,,,,,,
```

Use IDs 1–10 for `mouse` trials and 11–20 for `bottle` trials, but leave all result fields blank until the real Jetson test. Document changing distance, angle, orientation and background.

- [ ] **Step 2: Run and validate the Jetson demonstration**

In Jetson terminal 1:

```bash
cd ~/nb_mouse_bottle_vision
source vision_env/bin/activate
source /opt/ros/humble/setup.bash
python -m nb_vision.jetson_node --model model.pt --camera-id 0 --score-limit 0.55 --device 0
```

In Jetson terminal 2:

```bash
source /opt/ros/humble/setup.bash
ros2 topic list
ros2 topic echo /nb/objects
```

Expected: visible detection boxes/class/confidence/FPS, topic `/nb/objects`, and continuous JSON messages. Save screenshots/video locally, not in Git unless the file is demonstrably small.

- [ ] **Step 3: Fill real measurements and calculate accuracy**

Enter every actual observation. In `README.md`, calculate `correct rows / 20 * 100` using the CSV values and state whether accuracy is at least 80% and displayed FPS is at least 5. Add only real error-case details to `experiment/error_cases.md`.

- [ ] **Step 4: Run final repository verification**

Run: `python -m pytest -v; git status --short; git log --oneline --decorate -10`

Expected: all tests pass; no models/raw images/virtual environment appear in status; the log contains the staged milestone commits.

- [ ] **Step 5: Commit and publish**

```bash
git add experiment README.md
git commit -m "docs: record real jetson acceptance results"
gh repo create nb-mouse-bottle-vision --private --source . --remote origin --push
```

Run `gh auth status` first and ensure it shows the student’s GitHub account. Do not publish the repository until all sensitive or large artifacts are excluded.

## Plan self-review

- Spec coverage: class identity (Tasks 1 and 5); own data and reviewed labels (Task 3); split/audit (Task 2); Windows NVIDIA training and own model (Task 4); Jetson detection, display and ROS2 publishing (Task 5); 20 truthful trials, accuracy/FPS evidence and GitHub delivery (Task 6).
- Placeholder scan: no implementation placeholders remain.
- Interface consistency: `CLASS_NAMES`/`validate_yolo_line` flow from Task 1 to Task 2; `detection_packet` flows from Task 1 to Task 5; `build_object` is defined and tested within Task 5.
